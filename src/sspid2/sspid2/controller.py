###################
#  controller.py  # 
###################
#
# State-Space PD (SS-PD) gimbal controller.
#
# NAMING NOTE
# -----------
# The ROS 2 package (`sspid2`), this class (`SSPID`) and the `/pid_log` topic
# keep the historical "PID" name so that previously recorded bags and result
# files remain readable. The control law reported in the paper is a *PD* law:
# ki is 0.0 in every experiment. The integral channel of the observer is kept
# only for backwards compatibility and is NOT validated (see SSPID below); the
# node warns at start-up if ki != 0.

# Standard ROS 2 and numerical imports
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3
from std_msgs.msg import Float64MultiArray, String

# ============================================================================
#  ACTIVE GAIN SET
# ----------------------------------------------------------------------------
#  These are only the *defaults*; every gain can be overridden with ROS 2
#  parameters (see README). Swap this block with one of the alternatives below
#  to reproduce a specific experiment.
#
#  TODO(authors): confirm the mapping between these blocks and the experiment
#  numbers in the paper before tagging the release. The labels below are the
#  ones that were used in the working tree and have NOT been cross-checked
#  against the published tables.
# ============================================================================

# >>> Multi-objective (equal weights, w = [1, 1, 1, 1])
kp      = 8.63482313
ki      = 0.0
kd      = 0.15000789
omega_o = 600.29943956

# ---- Alternative gain sets -------------------------------------------------
#
# # Debug / smoke-test only (NOT a paper result)
# kp = 1.0;          ki = 0.0;  kd = 0.1;        omega_o = 100.0
#
# # Ziegler-Nichols baseline
# kp = 11.36;        ki = 0.0;  kd = 0.639;      omega_o = 500.0
#
# # MAE only                       w = [1, 0, 0, 0]
# kp = 13.221292;    ki = 0.0;  kd = 0.22545;    omega_o = 487.1265
#
# # MAE + Volatility               w = [1, 1, 0, 0]
# kp = 12.362949;    ki = 0.0;  kd = 0.122547;   omega_o = 932.0433
#
# # MAE + Zero-crossings           w = [1, 0, 0, 1]
# kp = 12.867482;    ki = 0.0;  kd = 0.135862;   omega_o = 956.2599
#
# # MAE + Acceleration             w = [1, 0, 1, 0]
# #   (this set was previously labelled "MAE + Jerk"; the quantity is the mean
# #    absolute SECOND difference of the radial error, i.e. an acceleration)
# kp = 12.032110;    ki = 0.0;  kd = 0.147699;   omega_o = 901.5414
#
# # Multi-objective, reweighted    w = [1, 0.5, 0.5, 1]
# kp = 10.23482313;  ki = 0.0;  kd = 0.12000789; omega_o = 818.29943956
# ----------------------------------------------------------------------------

# ---- Configuration Parameters ----------------------------------------------
# Camera and gimbal limits used to convert pixel error to angular error.
#
# NOTE: these must match the rendered image produced by the perception node
# (`distance_to_center.py` resizes every frame to TARGET_WIDTH preserving the
# aspect ratio). If you change the Isaac Sim camera resolution, change them
# here as well -- the geometry is currently duplicated in two places.
FRAME_WIDTH    = 640.0
FRAME_HEIGHT   = 360.0
FOCAL_LENGTH   = 18.1476   # mm, USD `focalLength`
HORIZ_APERTURE = 20.955    # mm, USD `horizontalAperture`

# Isaac Sim / USD derive the vertical field of view from the aspect ratio of
# the rendered image. The USD *default* verticalAperture (15.2908 mm) belongs
# to a ~4:3 sensor and does not describe a 640x360 (16:9) render: using it made
# the pitch angular error ~30 % larger than the yaw one for the same pixel
# offset, even though NPSO assigns identical gains to both axes.
VERT_APERTURE  = HORIZ_APERTURE * (FRAME_HEIGHT / FRAME_WIDTH)   # = 11.787 mm
# To reproduce the original (anisotropic) scaling, replace the line above with:
#   VERT_APERTURE = 15.2908

CONTROL_PERIOD = 0.001   # s, nominal control period (dt used by the observer)
MAX_VEL_TRACK  = 1.5     # rad/s
MAX_VEL_STAB   = 3.0     # rad/s
TARGET_TIMEOUT = 2.0     # time before confirming target loss
STATUS_DECIMATION = 100  # publish /control and console logs every N ticks
# -----------------------------------------------------------------------------


class SSPID:
    # State-space controller with an observer-based realization.
    # It estimates the error, its derivative, and its integral.
    #
    # With ki = 0 (the configuration used in the paper) this is a pure SS-PD.
    # The third observer state integrates the innovation (y - x_hat[1]), not
    # the tracking error, so the integral channel does NOT behave like a
    # classical I term. Do not enable ki without re-deriving the observer.
    # ------------------------------------------------------------------
    def __init__(self, kp: float, ki: float, kd: float,
                 omega_o: float, dt: float = CONTROL_PERIOD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # Observer bandwidth determines the observer coefficients.
        # Forward-Euler stability of the double pole at -omega_o requires
        # omega_o * dt < 2  (i.e. omega_o < 2000 rad/s at dt = 1 ms).
        self.omega_o = omega_o
        self.beta1 = omega_o ** 2   # Lo[0]
        self.beta2 = 2.0 * omega_o  # Lo[1]

        # Estimated state vector: [error_dot, error, integral_error]
        self.x_hat = np.zeros(3)

        # Input that was ACTUALLY applied to the plant on the previous sample.
        # The observer must be driven with the applied (saturated) input, not
        # with the raw control law output, otherwise the estimates diverge
        # while the command is saturated.
        self.u_prev = 0.0

        # Observer dynamics matrix
        self._A_obs = np.array([
            [0.0,     -self.beta1, 0.0],
            [1.0,     -self.beta2, 0.0],
            [0.0,     -1.0,        0.0]
        ])
        # Observer input vector
        self._Be = np.array([1.0, 0.0, 0.0])
        # Observer output injection vector
        self._Lo = np.array([self.beta1, self.beta2, 1.0])

    # ------------------------------------------------------------------
    def reset(self):
        # Reset the observer state to zero.
        self.x_hat = np.zeros(3)
        self.u_prev = 0.0

    # ------------------------------------------------------------------
    def compute(self, error: float) -> float:
        # Measured output for the observer.
        y = -error
        # ---- Observer ----
        # Euler integration of the observer dynamics, driven by the control
        # action applied on the previous sample.
        #
        # FIX: this used to be `u_prev = +self._Ko_dot_xhat()`, i.e. the
        # NEGATIVE of the control action that was actually applied, which made
        # the observer's internal model inconsistent with the plant input.
        x_hat_dot = (self._A_obs @ self.x_hat + self._Be * self.u_prev + self._Lo * y)
        self.x_hat = self.x_hat + self.dt * x_hat_dot
        # ---- Control law ----
        # Control action is computed from the estimated states.
        u = -self._Ko_dot_xhat()

        # Provisional: the caller must call set_applied_input() with the
        # saturated command so the next observer update sees the real input.
        self.u_prev = u
        return u

    # ------------------------------------------------------------------
    def set_applied_input(self, u_applied: float):
        # Report the command that was really sent to the gimbal (after
        # saturation) so the observer stays consistent with the plant.
        self.u_prev = float(u_applied)

    def _Ko_dot_xhat(self) -> float:  # multiplying Ko with x_hat
        # Combine derivative, proportional, and integral gains.
        return (self.kd * self.x_hat[0] + self.kp * self.x_hat[1] + self.ki * self.x_hat[2])


class GimbalControllerNode(Node):
    def __init__(self):
        super().__init__('gimbal_controller')

        # Declare ROS 2 parameters for yaw and pitch gains.
        self.declare_parameter('kp_yaw',            kp)
        self.declare_parameter('ki_yaw',            ki)
        self.declare_parameter('kd_yaw',            kd)
        self.declare_parameter('omega_o_yaw',       omega_o)
        self.declare_parameter('kp_pitch',          kp)
        self.declare_parameter('ki_pitch',          ki)
        self.declare_parameter('kd_pitch',          kd)
        self.declare_parameter('omega_o_pitch',     omega_o)

        # Read parameter values.
        _kp_y   = self.get_parameter('kp_yaw').value
        _ki_y   = self.get_parameter('ki_yaw').value
        _kd_y   = self.get_parameter('kd_yaw').value
        _oo_y   = self.get_parameter('omega_o_yaw').value
        _kp_p   = self.get_parameter('kp_pitch').value
        _ki_p   = self.get_parameter('ki_pitch').value
        _kd_p   = self.get_parameter('kd_pitch').value
        _oo_p   = self.get_parameter('omega_o_pitch').value

        # The published results use a PD law; warn loudly if that is not the case.
        if _ki_y != 0.0 or _ki_p != 0.0:
            self.get_logger().warn(
                'ki != 0: the observer integral channel is NOT the classical I '
                'term and has not been validated. The paper reports SS-PD (ki = 0).'
            )
        for name, value in (('omega_o_yaw', _oo_y), ('omega_o_pitch', _oo_p)):
            if value * CONTROL_PERIOD >= 2.0:
                self.get_logger().warn(
                    f'{name} = {value:.1f} rad/s violates the forward-Euler '
                    f'stability limit omega_o * dt < 2 at dt = {CONTROL_PERIOD} s.'
                )

        # Create SS-PID controllers for tracking mode.
        self.sspid_yaw   = SSPID(kp=_kp_y, ki=_ki_y, kd=_kd_y, omega_o=_oo_y)
        self.sspid_pitch = SSPID(kp=_kp_p, ki=_ki_p, kd=_kd_p, omega_o=_oo_p)

        # Create softer SS-PID controllers for stabilization mode.
        self.sspid_soft_yaw   = SSPID(kp=8.0, ki=0.0, kd=0.1, omega_o=10.0)
        self.sspid_soft_pitch = SSPID(kp=8.0, ki=0.0, kd=0.1, omega_o=10.0)

        # State variables.
        self.error = [0.0, 0.0, 0.0]
        self.current_orientation = [0.0, 0.0, 0.0]
        self.target_orientation  = [90.0, 0.0, 0.0]
        self.default_orientation = [90.0, 0.0, 0.0]
        self.last_target_time = None

        # Subscribers.
        self.create_subscription(Point,   '/target_coord',        self.target_callback,      10)
        self.create_subscription(Vector3, '/current_orientation', self.orientation_callback, 10)

        # Publishers.
        self.publisher     = self.create_publisher(Twist,             '/cam_vel', 10)
        self.log_publisher = self.create_publisher(Float64MultiArray, '/pid_log', 10)
        self.logger_info   = self.create_publisher(String,            '/control', 10)

        # Main control loop at 1 kHz.
        self.create_timer(CONTROL_PERIOD, self.main_loop)
        self._log_counter = 0

        # Loop-period monitor: the observer assumes dt = CONTROL_PERIOD, so a
        # systematic deviation invalidates the meaning of omega_o.
        self._last_loop_time = None
        self._period_acc = 0.0
        self._period_n = 0

    # ------------------------------------------------------------------
    def target_callback(self, msg):
        # Pixel error and target-detection flag.
        self.error[0] = msg.x
        self.error[1] = msg.y
        self.error[2] = msg.z

    def orientation_callback(self, msg):
        # Current gimbal orientation.
        self.current_orientation[0] = msg.x
        self.current_orientation[1] = msg.y
        self.current_orientation[2] = msg.z

    def pixel_to_angle(self, error_x, error_y):
        # Convert pixel error to angular line-of-sight error.
        pixel_size_x = HORIZ_APERTURE / FRAME_WIDTH
        pixel_size_y = VERT_APERTURE  / FRAME_HEIGHT
        sensor_error_x = error_x * pixel_size_x
        sensor_error_y = error_y * pixel_size_y
        angular_error_x_rad = np.arctan(sensor_error_x / FOCAL_LENGTH)
        angular_error_y_rad = np.arctan(sensor_error_y / FOCAL_LENGTH)
        return angular_error_x_rad, angular_error_y_rad

    # ------------------------------------------------------------------
    def _monitor_period(self):
        now = time.perf_counter()
        if self._last_loop_time is not None:
            self._period_acc += now - self._last_loop_time
            self._period_n += 1
            if self._period_n >= 5000:
                mean_dt = self._period_acc / self._period_n
                if abs(mean_dt - CONTROL_PERIOD) > 0.2 * CONTROL_PERIOD:
                    self.get_logger().warn(
                        f'Measured control period {mean_dt * 1e3:.3f} ms deviates '
                        f'from the nominal {CONTROL_PERIOD * 1e3:.3f} ms assumed by '
                        f'the observer. omega_o is scaled accordingly.'
                    )
                self._period_acc = 0.0
                self._period_n = 0
        self._last_loop_time = now

    # ------------------------------------------------------------------
    def main_loop(self):
        self._monitor_period()
        # If no target is detected (flag = 0), run stabilization mode.
        if abs(self.error[2]) == 0.0:   # No target detected
            current_time = time.time()
            if self.last_target_time is not None:
                # After a timeout, return to the default orientation.
                if (current_time - self.last_target_time) > TARGET_TIMEOUT:
                    self.target_orientation = self.default_orientation.copy()
                    self.last_target_time = None
                    self.sspid_yaw.reset()
                    self.sspid_pitch.reset()
            self.stabilization_loop()
        else:                           # Target detected
            # Keep the last known orientation while the target is visible.
            self.target_orientation[0] = self.current_orientation[0]
            self.target_orientation[2] = self.current_orientation[2]
            self.last_target_time = time.time()
            self.sspid_soft_yaw.reset()
            self.sspid_soft_pitch.reset()
            self.control_loop()

    # ------------------------------------------------------------------
    def _publish_status(self, mode, vel_pitch, vel_yaw, ctrl):
        # Published at STATUS_DECIMATION Hz-divided rate: building and sending
        # a String at 1 kHz prevented the control timer from meeting its period.
        # use <<ros2 topic echo /control --field data >> to see these logs
        msg = String()
        msg.data = (
            f'\n\n[{mode}]'
            f'\nOffset:     ({self.error[0]:.3f}, {self.error[1]:.3f}) px'
            f'\nVelocity:   ({vel_pitch:.3f}, {vel_yaw:.3f}) rad/s'
            f'\nParameters: (Kp:{ctrl.kp:.3f}, Ki:{ctrl.ki:.1f}, '
            f'Kd:{ctrl.kd:.4f}, wo:{ctrl.omega_o:.1f})'
        )
        self.logger_info.publish(msg)

    # ------------------------------------------------------------------
    def stabilization_loop(self):
        # Compute angular error between target and current orientation.
        # The orientation topic is in degrees; the controllers work in radians.
        error_pitch = math.radians(self.target_orientation[0] - self.current_orientation[0])
        error_yaw   = math.radians(self.target_orientation[2] - self.current_orientation[2])

        # Soft SS-PID controllers for smooth stabilization.
        vel_yaw   = self.sspid_soft_yaw.compute(error_yaw)
        vel_pitch = self.sspid_soft_pitch.compute(error_pitch)

        # Saturate, then tell each observer what was really applied.
        vel_yaw   = min(MAX_VEL_STAB, max(-MAX_VEL_STAB, vel_yaw))
        vel_pitch = min(MAX_VEL_STAB, max(-MAX_VEL_STAB, vel_pitch))
        self.sspid_soft_yaw.set_applied_input(vel_yaw)
        self.sspid_soft_pitch.set_applied_input(vel_pitch)

        # Publish bounded angular velocities.
        # Convention of /cam_vel: angular.x = pitch rate, angular.y = yaw rate.
        msg = Twist()
        msg.angular.x = vel_pitch
        msg.angular.y = vel_yaw
        self.publisher.publish(msg)

        # Log status periodically.
        self._log_counter += 1
        if self._log_counter % STATUS_DECIMATION == 0:
            self.get_logger().info(
                f'\n [STABILIZATION MODE]'
                f'\n ---------------------------------------------------'
                f'\n |      | Angle Error |   Velocity   | Current Pos.|'
                f'\n ---------------------------------------------------'
                f'\n |Yaw:  |  {error_yaw:.3f} rad  | {vel_yaw:.3f} rad/s |  {self.current_orientation[2]:.3f} deg  |'
                f'\n |Pitch:|  {error_pitch:.3f} rad  | {vel_pitch:.3f} rad/s |  {self.current_orientation[0]:.3f} deg  |'
                f'\n ---------------------------------------------------\n'
            )
            self._publish_status('STABILIZATION MODE', vel_pitch, vel_yaw, self.sspid_soft_pitch)

    # ------------------------------------------------------------------
    def control_loop(self):
        # Convert pixel error to angular error.
        yaw_error, pitch_error = self.pixel_to_angle(self.error[0], self.error[1])

        # SS-PD control with velocity saturation.
        vel_yaw   = min(MAX_VEL_TRACK, max(-MAX_VEL_TRACK, self.sspid_yaw.compute(yaw_error)))
        vel_pitch = min(MAX_VEL_TRACK, max(-MAX_VEL_TRACK, self.sspid_pitch.compute(pitch_error)))

        # Close the loop around the observer with the saturated command.
        self.sspid_yaw.set_applied_input(vel_yaw)
        self.sspid_pitch.set_applied_input(vel_pitch)

        # Publish bounded angular velocities.
        msg = Twist()
        msg.angular.x =  vel_pitch
        msg.angular.y = -vel_yaw
        self.publisher.publish(msg)

        # Publish numerical log for the evaluator.
        # NOTE: /pid_log is only published in tracking mode, so every metric
        # derived from it is scoped to the samples in which a target was being
        # tracked. See the "Data and Results" section of the README.
        log_msg = Float64MultiArray()
        log_msg.data = [
            float(self.error[0]),                         # Pixel error x [px]
            float(self.error[1]),                         # Pixel error y [px]
            float(yaw_error),                             # Angular error yaw [rad]
            float(pitch_error),                           # Angular error pitch [rad]
            float(vel_yaw),                               # Velocity yaw [rad/s]
            float(vel_pitch),                             # Velocity pitch [rad/s]
            float(self.sspid_yaw.x_hat[2]),               # Observer integral state yaw
            float(self.sspid_pitch.x_hat[2]),             # Observer integral state pitch
            float(self.error[2]),                         # Target detected flag
        ]
        self.log_publisher.publish(log_msg)

        # Log status periodically.
        self._log_counter += 1
        if self._log_counter % STATUS_DECIMATION == 0:
            self.get_logger().info(
                f'\n[TRACKING MODE]'
                f'\n   Offset:    ({self.error[0]:.3f}, {self.error[1]:.3f}) px'
                f'\n   Velocity:  ({vel_pitch:.3f}, {vel_yaw:.3f}) rad/s'
                f'\n   Parameters:(Kp:{self.sspid_yaw.kp:.3f}, Ki:{self.sspid_yaw.ki:.1f}, '
                f'Kd:{self.sspid_yaw.kd:.4f}, wo:{self.sspid_yaw.omega_o:.1f})\n'
            )
            self._publish_status('TRACKING MODE', vel_pitch, vel_yaw, self.sspid_yaw)


def main(args=None):
    # Standard ROS 2 node startup.
    rclpy.init(args=args)
    node = GimbalControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
