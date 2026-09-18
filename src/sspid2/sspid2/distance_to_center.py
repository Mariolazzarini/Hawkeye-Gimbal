# distance_to_center.py w/enhanced visualization + EMA filter
import rclpy
import cv2
import math
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from ultralytics import YOLO
import numpy as np
from collections import deque


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.sub = self.create_subscription(Image, '/rgb', self.listener_callback, qos_profile)
        self.pub = self.create_publisher(Point, '/target_coord', qos_profile)
        self.pub_error_raro = self.create_publisher(Float64, '/controller/error', qos_profile)

        # ---------------- METRICS PUBLISHERS ----------------
        self.pub_target_availability = self.create_publisher(Float64, '/metrics/target_availability', qos_profile)
        self.pub_fps_60s = self.create_publisher(Float64, '/metrics/fps_60s', qos_profile)
        self.pub_inter_frame_dt = self.create_publisher(Float64, '/metrics/inter_frame_dt', qos_profile)
        self.pub_target_confidence = self.create_publisher(Float64, '/metrics/target_confidence', qos_profile)
        self.pub_id_switch_event = self.create_publisher(Float64, '/metrics/id_switch_event', qos_profile)

        # ---------------- METRICS STATE ----------------
        self._last_frame_time_sec = None
        self._fps_window_sec = 60.0
        self._frame_times_60s = deque()
        self._availability_60s = deque()

        self.bridge = CvBridge()

        # YOLO model
        self.model = YOLO('config/yolov8m.pt')

        # ---- SPEED / RESOLUTION CONTROL ----
        self.TARGET_WIDTH = 640
        self.PROCESS_EVERY_N_FRAMES = 1
        self.frame_skip_counter = 0
        self.SHOW_DEBUG = True

        # ---------------- VISUAL STYLE ----------------
        self.VIZ_LINE_THICKNESS = 1
        self.VIZ_TEXT_THICKNESS = 1
        self.VIZ_FONT_SCALE_MAIN = 0.55
        self.VIZ_FONT_SCALE_SMALL = 0.45
        self.VIZ_CIRCLE_RADIUS = 3
        self.VIZ_LINE_TYPE = cv2.LINE_AA

        # ---------------- OVERLAY TOGGLES ----------------
        self.OVERLAY_ON = True
        self.OVERLAY_BOXES = True
        self.OVERLAY_ERROR_LINE = True
        self.OVERLAY_TARGET_DOT = True
        self.OVERLAY_TEXT = True
        self.OVERLAY_HELP = True

        # --- TRACKING STATE ---
        self.locked_id = None
        self.last_known_pos = None   # raw center (still useful for Re-ID)
        self.patience_counter = 0
        self.PATIENCE_LIMIT = 5
        self.REID_DISTANCE_LIMIT = 150

        # --- EMA FILTER ---
        self.EMA_ALPHA = 0.28                # weight for newest measurement
        self._filtered_center = None         # (fx, fy) smoothed center in 640px coords

        self.available_ids = []

    # ------------------------------------------------------------
    # Helper: Exponential Moving Average for center coordinates
    # ------------------------------------------------------------
    def _apply_ema(self, cx_raw, cy_raw, reset=False):
        """
        Update and return the filtered center using exponential decay.
        If reset=True (e.g., new ID), the filter reinitialises to the raw value.
        """
        if reset or self._filtered_center is None:
            self._filtered_center = (float(cx_raw), float(cy_raw))
        else:
            fx = self.EMA_ALPHA * cx_raw + (1.0 - self.EMA_ALPHA) * self._filtered_center[0]
            fy = self.EMA_ALPHA * cy_raw + (1.0 - self.EMA_ALPHA) * self._filtered_center[1]
            self._filtered_center = (fx, fy)
        return self._filtered_center

    # ------------------------------------------------------------
    # Original helper methods (unchanged except where noted)
    # ------------------------------------------------------------
    def get_center(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _draw_boxes_thin(self, frame, detections_xyxy, detections_ids=None):
        if detections_xyxy is None:
            return
        for i, box in enumerate(detections_xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), self.VIZ_LINE_THICKNESS, self.VIZ_LINE_TYPE)
            if detections_ids is not None:
                obj_id = int(detections_ids[i])
                cv2.putText(frame, f"ID {obj_id}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_SMALL,
                            (0, 255, 0), self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)

    def _draw_overlay(self, frame, cx=None, cy=None, obj_id=None, status=None):
        h, w = frame.shape[:2]
        if cx is not None and cy is not None:
            color = (0, 255, 0) if status == "TRACKING" else (0, 0, 255)
            if status == "RECOVERED":
                color = (0, 255, 255)
            if self.OVERLAY_TARGET_DOT:
                cv2.circle(frame, (int(cx), int(cy)), self.VIZ_CIRCLE_RADIUS, color, -1, self.VIZ_LINE_TYPE)
            if self.OVERLAY_ERROR_LINE:
                cv2.line(frame, (int(cx), int(cy)), (w // 2, h // 2), color, self.VIZ_LINE_THICKNESS, self.VIZ_LINE_TYPE)
            if self.OVERLAY_TEXT and obj_id is not None and status is not None:
                cv2.putText(frame, f"ID: {obj_id} [{status}]", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_MAIN, color,
                            self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)
        if self.OVERLAY_HELP:
            cv2.putText(frame, "Keys: q quit | a/d switch ID | v overlay | b boxes | e line | t text | h help",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_SMALL,
                        (255, 255, 255), self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)

    def draw_info(self, frame, cx, cy, dist, obj_id, status="TRACKING"):
        color = (0, 255, 0) if status == "TRACKING" else (0, 0, 255)
        if status == "RECOVERED":
            color = (0, 255, 255)
        cv2.circle(frame, (int(cx), int(cy)), self.VIZ_CIRCLE_RADIUS, color, -1, self.VIZ_LINE_TYPE)
        h, w = frame.shape[:2]
        cv2.line(frame, (int(cx), int(cy)), (w // 2, h // 2), color, self.VIZ_LINE_THICKNESS, self.VIZ_LINE_TYPE)
        cv2.putText(frame, f"ID: {obj_id} [{status}]", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_MAIN, color, self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)
        if status == "LOST":
            cv2.putText(frame, f"Drop in: {self.PATIENCE_LIMIT - self.patience_counter}",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_SMALL,
                        (0, 0, 255), self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)
        cv2.putText(frame, "Keys: q quit | a/d switch ID | v overlay | b boxes | e line | t text | h help",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_SMALL, (255, 255, 255),
                    self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)

    def publish_no_target(self):
        pt = Point()
        pt.x = 0.0; pt.y = 0.0; pt.z = 0.0
        self.pub.publish(pt)
        msg = Float64()
        msg.data = 0.0
        self.pub_error_raro.publish(msg)

    def _resize_to_target_width(self, frame):
        orig_h, orig_w = frame.shape[:2]
        if orig_w == self.TARGET_WIDTH:
            return frame
        scale = self.TARGET_WIDTH / float(orig_w)
        new_h = int(orig_h * scale)
        return cv2.resize(frame, (self.TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------
    # Main callback (modified to use EMA in all tracking branches)
    # ------------------------------------------------------------
    def listener_callback(self, msg: Image):
        try:
            # ---------------- METRICS (timing/FPS) ----------------
            now_sec = self.get_clock().now().nanoseconds / 1e9
            dt = None
            if self._last_frame_time_sec is not None:
                dt = now_sec - self._last_frame_time_sec
                if dt < 0.0:
                    dt = None
            self._last_frame_time_sec = now_sec
            if dt is not None:
                dt_msg = Float64()
                dt_msg.data = float(dt)
                self.pub_inter_frame_dt.publish(dt_msg)

            self._frame_times_60s.append(now_sec)
            cutoff = now_sec - self._fps_window_sec
            while self._frame_times_60s and self._frame_times_60s[0] < cutoff:
                self._frame_times_60s.popleft()
            fps_msg = Float64()
            fps_msg.data = float(len(self._frame_times_60s)) / self._fps_window_sec
            self.pub_fps_60s.publish(fps_msg)

            # ---- FRAME SKIPPING ----
            self.frame_skip_counter += 1
            if self.frame_skip_counter < self.PROCESS_EVERY_N_FRAMES:
                return
            self.frame_skip_counter = 0

            # ---- IMAGE & YOLO ----
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_proc = self._resize_to_target_width(frame)

            CUSTOM_TRACKER = "config/custom_tracker.yaml"
            results = self.model.track(
                frame_proc, persist=True, tracker=CUSTOM_TRACKER,
                conf=0.35, imgsz=self.TARGET_WIDTH, classes=[0], verbose=False
            )

            # ---- PARSE DETECTIONS ----
            current_detections = {}    # id -> (cx, cy) raw
            current_conf = {}
            fallback_candidates = []   # list of ((cx, cy), conf)
            self.available_ids = []

            if len(results) > 0:
                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                    if r.boxes.id is not None:
                        ids = r.boxes.id.cpu().numpy().astype(int)
                        for i, obj_id in enumerate(ids):
                            center = self.get_center(boxes[i])
                            current_detections[obj_id] = center
                            if confs is not None and i < len(confs):
                                current_conf[obj_id] = float(confs[i])
                            self.available_ids.append(obj_id)
                    else:
                        for i in range(len(boxes)):
                            center = self.get_center(boxes[i])
                            c = float(confs[i]) if (confs is not None and i < len(confs)) else float('nan')
                            fallback_candidates.append((center, c))

            self.available_ids.sort()

            # ---- TRACKING LOGIC (EMA integrated) ----
            status = "IDLE"
            target_active = False
            strict_active = False
            tracked_conf = None

            if self.locked_id is not None:
                # ----- A) locked_id present -----
                if self.locked_id in current_detections:
                    self.patience_counter = 0
                    cx_raw, cy_raw = current_detections[self.locked_id]
                    self.last_known_pos = (cx_raw, cy_raw)

                    # --- APPLY EMA FILTER ---
                    fx, fy = self._apply_ema(cx_raw, cy_raw)
                    self.publish_target(fx, fy, frame_proc)

                    status = "TRACKING"
                    target_active = True
                    strict_active = True
                    tracked_conf = current_conf.get(self.locked_id, None)

                # ----- A2) fallback candidates (no IDs) -----
                elif len(fallback_candidates) > 0 and self.last_known_pos is not None:
                    best = min(fallback_candidates, key=lambda pc: self.distance(self.last_known_pos, pc[0]))
                    (cx_raw, cy_raw), c = best
                    self.patience_counter = 0
                    self.last_known_pos = (cx_raw, cy_raw)

                    fx, fy = self._apply_ema(cx_raw, cy_raw)
                    self.publish_target(fx, fy, frame_proc)

                    status = "TRACKING"
                    target_active = True
                    strict_active = True
                    tracked_conf = c

                # ----- B) Re-ID -----
                else:
                    found_replacement = False
                    if self.last_known_pos is not None:
                        best_match_id = None
                        min_dist = float('inf')
                        for cand_id, cand_pos in current_detections.items():
                            dist = self.distance(self.last_known_pos, cand_pos)
                            if dist < self.REID_DISTANCE_LIMIT and dist < min_dist:
                                min_dist = dist
                                best_match_id = cand_id
                        if best_match_id is not None:
                            self.get_logger().warn(f"ID Switch detected! {self.locked_id} -> {best_match_id}")
                            ev = Float64()
                            ev.data = 1.0
                            self.pub_id_switch_event.publish(ev)

                            self.locked_id = best_match_id
                            self.patience_counter = 0
                            cx_raw, cy_raw = current_detections[self.locked_id]
                            self.last_known_pos = (cx_raw, cy_raw)

                            # Reset EMA because we switched to a new ID
                            fx, fy = self._apply_ema(cx_raw, cy_raw, reset=True)
                            self.publish_target(fx, fy, frame_proc)

                            status = "RECOVERED"
                            found_replacement = True
                            target_active = True
                            strict_active = True
                            tracked_conf = current_conf.get(self.locked_id, None)

                    # ----- C) lost (patience window) -----
                    if not found_replacement:
                        self.patience_counter += 1
                        if self.patience_counter <= self.PATIENCE_LIMIT and self.last_known_pos is not None:
                            # Hold last filtered position (no new raw measurement)
                            if self._filtered_center is not None:
                                fx, fy = self._filtered_center
                                self.publish_target(fx, fy, frame_proc)
                            else:
                                # Fallback to raw if filter uninitialised (should not happen)
                                lx, ly = self.last_known_pos
                                self.publish_target(lx, ly, frame_proc)

                            if self.SHOW_DEBUG and self.OVERLAY_ON:
                                self.draw_info(frame_proc, fx if self._filtered_center else lx,
                                               fy if self._filtered_center else ly,
                                               0, self.locked_id, status="LOST")
                            target_active = True
                            strict_active = False
                            tracked_conf = None
                        else:
                            # Totally lost
                            self.locked_id = None
                            self.last_known_pos = None
                            self._filtered_center = None
                            self.publish_no_target()
                            target_active = False
                            strict_active = False
                            tracked_conf = None
                            if self.SHOW_DEBUG and self.OVERLAY_ON:
                                cv2.putText(frame_proc, "NO TARGET - STABILIZING", (10, 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_MAIN,
                                            (0, 0, 255), self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)

            # ---- Auto-lock if nothing locked and IDs exist ----
            elif self.available_ids:
                self.locked_id = self.available_ids[0]
                self.get_logger().info(f"Auto-locking to ID: {self.locked_id}")
                # Filter will be initialised on the next frame's detection automatically
            else:
                self.publish_no_target()
                if self.SHOW_DEBUG and self.OVERLAY_ON:
                    cv2.putText(frame_proc, "NO TARGET - STABILIZING", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, self.VIZ_FONT_SCALE_MAIN,
                                (0, 0, 255), self.VIZ_TEXT_THICKNESS, self.VIZ_LINE_TYPE)

            # ---------------- METRICS (availability + confidence) ----------------
            self._availability_60s.append((now_sec, 1 if strict_active else 0))
            cutoff = now_sec - self._fps_window_sec
            while self._availability_60s and self._availability_60s[0][0] < cutoff:
                self._availability_60s.popleft()
            if self._availability_60s:
                avail_pct = 100.0 * (sum(v for _, v in self._availability_60s) / len(self._availability_60s))
            else:
                avail_pct = 0.0
            avail_msg = Float64()
            avail_msg.data = float(avail_pct)
            self.pub_target_availability.publish(avail_msg)

            if strict_active and tracked_conf is not None and not math.isnan(tracked_conf):
                conf_msg = Float64()
                conf_msg.data = float(tracked_conf)
                self.pub_target_confidence.publish(conf_msg)

            # ---------------- VISUALIZATION ----------------
            if self.SHOW_DEBUG:
                if not self.OVERLAY_ON:
                    display = frame_proc
                else:
                    display = frame_proc.copy()
                    if self.OVERLAY_BOXES and len(results) > 0 and results[0].boxes is not None:
                        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
                        self._draw_boxes_thin(display, boxes_xyxy, ids)

                    if status in ["TRACKING", "RECOVERED"] and target_active and self.last_known_pos is not None:
                        # Show the FILTERED center on the overlay
                        if self._filtered_center is not None:
                            fx, fy = self._filtered_center
                        else:
                            fx, fy = self.last_known_pos  # fallback
                        self._draw_overlay(display, cx=fx, cy=fy, obj_id=self.locked_id, status=status)
                    else:
                        self._draw_overlay(display, cx=None, cy=None)

                cv2.imshow("YOLOv8 Robust Tracking (640px)", display)
                key = cv2.waitKey(1) & 0xFF
                self.handle_keys(key)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    # ------------------------------------------------------------
    # publish_target uses the coordinates passed (already filtered)
    # ------------------------------------------------------------
    def publish_target(self, cx, cy, frame_proc):
        h, w = frame_proc.shape[:2]
        dx = float(cx) - (w / 2.0)
        dy = float(cy) - (h / 2.0)
        dist = math.sqrt(dx * dx + dy * dy)
        pt = Point()
        pt.x = dx
        pt.y = -dy
        pt.z = 1.0
        self.pub.publish(pt)
        msg = Float64()
        msg.data = float(dist)
        self.pub_error_raro.publish(msg)

    # ------------------------------------------------------------
    # handle_keys: reset EMA when manually switching ID
    # ------------------------------------------------------------
    def handle_keys(self, key):
        if key == 255:
            return
        if key == ord('q'):
            rclpy.shutdown()
            cv2.destroyAllWindows()
            return
        if key == ord('v'):
            self.OVERLAY_ON = not self.OVERLAY_ON
            self.get_logger().info(f"OVERLAY_ON = {self.OVERLAY_ON}")
            return
        if key == ord('b'):
            self.OVERLAY_BOXES = not self.OVERLAY_BOXES
            self.get_logger().info(f"OVERLAY_BOXES = {self.OVERLAY_BOXES}")
            return
        if key == ord('e'):
            self.OVERLAY_ERROR_LINE = not self.OVERLAY_ERROR_LINE
            self.get_logger().info(f"OVERLAY_ERROR_LINE = {self.OVERLAY_ERROR_LINE}")
            return
        if key == ord('t'):
            self.OVERLAY_TEXT = not self.OVERLAY_TEXT
            self.get_logger().info(f"OVERLAY_TEXT = {self.OVERLAY_TEXT}")
            return
        if key == ord('h'):
            self.OVERLAY_HELP = not self.OVERLAY_HELP
            self.get_logger().info(f"OVERLAY_HELP = {self.OVERLAY_HELP}")
            return
        elif key == ord('d') or key == ord('a'):
            if not self.available_ids:
                self.get_logger().warn("No targets available to switch.")
                return
            current_index = -1
            if self.locked_id in self.available_ids:
                current_index = self.available_ids.index(self.locked_id)
            if key == ord('d'):
                new_index = (current_index + 1) % len(self.available_ids)
                action = "Next"
            else:
                new_index = (current_index - 1) % len(self.available_ids)
                action = "Previous"
            self.locked_id = self.available_ids[new_index]
            self.patience_counter = 0
            # Reset EMA for the new ID
            self._filtered_center = None
            self.get_logger().info(f"Manual Switch ({action}) -> Tracking ID: {self.locked_id}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()