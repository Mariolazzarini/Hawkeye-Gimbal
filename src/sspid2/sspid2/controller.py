###################
#  controller.py  # SSPID2 
###################

import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3 
from std_msgs.msg import Float64MultiArray, String

# Multi-objective
kp      = 10.23482313;   
ki      = 0.000000000;   
kd      = 0.120007890;   
omega_o = 818.29943956;   

# # Multi-objective-2
# kp      = 10.23482313;   
# ki      = 0.000000000;   
# kd      = 0.120007890;   
# omega_o = 818.29943956;   

# # Ziegler-Nichols
# kp = 11.36
# ki = 0.0
# kd = 0.639  
# omega_o = 500.0;      

# # only-MAE
# kp      = 13.221292
# ki      = 0.000000
# kd      = 0.22545
# omega_o = 487.1265   

# # MAE + Jerk
# kp      = 12.032110
# ki      = 0.000000
# kd      = 0.147699
# omega_o = 901.5414

# # MAE + Zero-crossings
# kp      = 12.867482
# ki      = 0.000000
# kd      = 0.135862
# omega_o = 956.2599

# MAE + Volatility
# kp      = 12.362949
# ki      = 0.000000
# kd      = 0.122547
# omega_o = 932.0433

# # Multi-objective
# kp      = 8.63482313;   
# ki      = 0.000000000;   
# kd      = 0.150007890;   
# omega_o = 600.29943956;   

# ---- Configuration Parameters -------------------------------
FRAME_WIDTH    = 640.0 
FRAME_HEIGHT   = 360.0
FOCAL_LENGTH   = 18.1476
HORIZ_APERTURE = 20.955
VERT_APERTURE  = 15.2908
MAX_VEL_TRACK  = 1.5     # rad/s
MAX_VEL_STAB   = 3.0     # rad/s
TARGET_TIMEOUT = 2.0     # time before confirming target loss
# -------------------------------------------------------------

class SSPID:
    # ------------------------------------------------------------------
    def __init__(self, kp: float, ki: float, kd: float,
                 omega_o: float, dt: float = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.omega_o = omega_o
        self.beta1 = omega_o ** 2   # Lo[0]
        self.beta2 = 2.0 * omega_o  # Lo[1]

        self.x_hat = np.zeros(3)

        self._A_obs = np.array([
            [0.0,     -self.beta1, 0.0],
            [1.0,     -self.beta2, 0.0],
            [0.0,     -1.0,        0.0]
        ])
        self._Be = np.array([1.0, 0.0, 0.0])
        self._Lo = np.array([self.beta1, self.beta2, 1.0])

    # ------------------------------------------------------------------
    def reset(self):
        self.x_hat = np.zeros(3)

    # ------------------------------------------------------------------
    def compute(self, error: float) -> float:
        y = -error  
        # ---- Observer ----
        u_prev = self._Ko_dot_xhat()  
        x_hat_dot = (self._A_obs @ self.x_hat + self._Be * u_prev + self._Lo * y)
        self.x_hat = self.x_hat + self.dt * x_hat_dot
        # ---- Control law ----
        u = -self._Ko_dot_xhat()  

        return u 

    def _Ko_dot_xhat(self) -> float: # multiplying Ko with x_hat
        return (self.kd*self.x_hat[0] + self.kp*self.x_hat[1] + self.ki*self.x_hat[2])


class GimbalControllerNode(Node):
    def __init__(self):
        super().__init__('gimbal_controller')

        self.declare_parameter('kp_yaw',            kp)
        self.declare_parameter('ki_yaw',            ki)
        self.declare_parameter('kd_yaw',            kd)
        self.declare_parameter('omega_o_yaw',       omega_o)
        self.declare_parameter('kp_pitch',          kp)
        self.declare_parameter('ki_pitch',          ki)
        self.declare_parameter('kd_pitch',          kd)
        self.declare_parameter('omega_o_pitch',     omega_o)

        _kp_y   = self.get_parameter('kp_yaw').value
        _ki_y   = self.get_parameter('ki_yaw').value
        _kd_y   = self.get_parameter('kd_yaw').value
        _oo_y   = self.get_parameter('omega_o_yaw').value
        _kp_p   = self.get_parameter('kp_pitch').value
        _ki_p   = self.get_parameter('ki_pitch').value
        _kd_p   = self.get_parameter('kd_pitch').value
        _oo_p   = self.get_parameter('omega_o_pitch').value

        self.sspid_yaw   = SSPID(kp=_kp_y, ki=_ki_y, kd=_kd_y, omega_o=_oo_y)
        self.sspid_pitch = SSPID(kp=_kp_p, ki=_ki_p, kd=_kd_p, omega_o=_oo_p)

        self.sspid_soft_yaw   = SSPID(kp=8.0, ki=0.0, kd=0.1, omega_o=10.0)
        self.sspid_soft_pitch = SSPID(kp=8.0, ki=0.0, kd=0.1, omega_o=10.0)

        self.error = [0.0, 0.0, 0.0]
        self.current_orientation = [0.0, 0.0, 0.0]
        self.target_orientation  = [90.0, 0.0, 0.0]
        self.default_orientation = [90.0, 0.0, 0.0]
        self.last_target_time = None

        self.create_subscription(Point,   '/target_coord',        self.target_callback,      10)
        self.create_subscription(Vector3, '/current_orientation', self.orientation_callback, 10)
        self.publisher     = self.create_publisher(Twist,             '/cam_vel', 10)
        self.log_publisher = self.create_publisher(Float64MultiArray, '/pid_log', 10)
        self.logger_info   = self.create_publisher(String,            '/control', 10)

        self.create_timer(0.001, self.main_loop)
        self._log_counter = 0

    # ------------------------------------------------------------------
    def target_callback(self, msg):
        self.error[0] = msg.x
        self.error[1] = msg.y
        self.error[2] = msg.z

    def orientation_callback(self, msg):
        self.current_orientation[0] = msg.x
        self.current_orientation[1] = msg.y
        self.current_orientation[2] = msg.z

    def pixel_to_angle(self, error_x, error_y):
        pixel_size_x = HORIZ_APERTURE / FRAME_WIDTH
        pixel_size_y = VERT_APERTURE  / FRAME_HEIGHT
        sensor_error_x = error_x * pixel_size_x
        sensor_error_y = error_y * pixel_size_y
        angular_error_x_rad = np.arctan(sensor_error_x / FOCAL_LENGTH)
        angular_error_y_rad = np.arctan(sensor_error_y / FOCAL_LENGTH)
        return angular_error_x_rad, angular_error_y_rad

    # ------------------------------------------------------------------
    def main_loop(self):
        if abs(self.error[2]) == 0.0:   # No target detected
            current_time = time.time()
            if self.last_target_time is not None:
                if (current_time - self.last_target_time) > TARGET_TIMEOUT: # Maintain last target-detected orientation for a couple seconds before switching back 
                                                                            # to default orientation, to avoid rapid moving back and forth when target is lost for a decimal of seconds.
                    self.target_orientation = self.default_orientation.copy()
                    self.last_target_time = None
                    self.sspid_yaw.reset()
                    self.sspid_pitch.reset()
            self.stabilization_loop()
        else:                           # Target detected
            self.target_orientation[0] = self.current_orientation[0] # The target orientation is updated to the current orientation every time the target is in camera, 
            self.target_orientation[2] = self.current_orientation[2] # so that when the target is lost, it will maintain the last orientation instead of snapping back to default.
                                                                     # to avoid rapid changes when target is lost for just a decimal of a second.
            self.last_target_time = time.time()
            self.sspid_soft_yaw.reset()
            self.sspid_soft_pitch.reset()
            self.control_loop()

    # ------------------------------------------------------------------
    def stabilization_loop(self):
        error_pitch = math.radians(self.target_orientation[0] - self.current_orientation[0])
        error_yaw   = math.radians(self.target_orientation[2] - self.current_orientation[2])

        vel_yaw   = self.sspid_soft_yaw.compute(error_yaw)
        vel_pitch = self.sspid_soft_pitch.compute(error_pitch)

        msg = Twist()
        msg.angular.x =  min(MAX_VEL_STAB, max(-MAX_VEL_STAB, vel_pitch))
        msg.angular.y =  min(MAX_VEL_STAB, max(-MAX_VEL_STAB, vel_yaw))
        self.publisher.publish(msg)

        self._log_counter += 1
        if self._log_counter % 100 == 0:
            self.get_logger().info(
                f'\n 🔄 [STABILIZATION MODE]'
                f'\n ---------------------------------------------------'
                f'\n |      | Angle Error |   Velocity   | Current Pos.|' 
                f'\n ---------------------------------------------------'          
                f'\n |Yaw:  |  {error_yaw:.3f} deg  | {vel_yaw:.3f} rad/s |  {self.current_orientation[2]:.3f} deg  |'
                f'\n |Pitch:|  {error_pitch:.3f} deg  | {vel_pitch:.3f} rad/s |  {self.current_orientation[0]:.3f} deg  |'
                f'\n ---------------------------------------------------\n'
            )

        msg_text = ( # use <<ros2 topic echo /control --field data >> to see these logs
            f'\n\nOffset:   ({self.error[0]:.3f}, {self.error[1]:.3f}) px'
            f'\nVelocity:   ({vel_pitch:.3f}, {vel_yaw:.3f}) rad/s'
            f'\nParameters: (Kp:{self.sspid_pitch.kp:.3f}, Ki:{self.sspid_pitch.ki:.1f}, Kd:{self.sspid_pitch.kd:.4f}, ωo:{self.sspid_pitch.omega_o:.1f})'
        )
        msg = String()
        msg.data = msg_text
        self.logger_info.publish(msg)

    # ------------------------------------------------------------------
    def control_loop(self):
        yaw_error, pitch_error = self.pixel_to_angle(self.error[0], self.error[1])
        vel_yaw   = min(MAX_VEL_TRACK, max(-MAX_VEL_TRACK, self.sspid_yaw.compute(yaw_error)))
        vel_pitch = min(MAX_VEL_TRACK, max(-MAX_VEL_TRACK, self.sspid_pitch.compute(pitch_error)))

        msg = Twist()
        msg.angular.x =  vel_pitch
        msg.angular.y = -vel_yaw
        self.publisher.publish(msg)

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

        self._log_counter += 1
        if self._log_counter % 100 == 0:
            self.get_logger().info(
                f'\n🎯 [TRACKING MODE]'
                f'\n   Offset:    ({self.error[0]:.3f}, {self.error[1]:.3f}) px'
                f'\n   Velocity:  ({vel_pitch:.3f}, {vel_yaw:.3f}) rad/s'
                f'\n   Parameters:(Kp:{self.sspid_yaw.kp:.3f}, Ki:{self.sspid_yaw.ki:.1f}, Kd:{self.sspid_yaw.kd:.4f}, ωo:{self.sspid_yaw.omega_o:.1f})\n'
            )

        msg_text = ( # use <<ros2 topic echo /control --field data >> to see these logs
            f'\n\nOffset:   ({self.error[0]:.3f}, {self.error[1]:.3f}) px'
            f'\nVelocity:   ({vel_pitch:.3f}, {vel_yaw:.3f}) rad/s'
            f'\nParameters: (Kp:{self.sspid_yaw.kp:.3f}, Ki:{self.sspid_yaw.ki:.1f}, Kd:{self.sspid_yaw.kd:.4f}, ωo:{self.sspid_yaw.omega_o:.1f})'
        )
        msg = String()
        msg.data = msg_text
        self.logger_info.publish(msg)

def main(args=None):
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