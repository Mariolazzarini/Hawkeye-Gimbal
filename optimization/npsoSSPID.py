########################################################################
#  npsoSSPID.py                                                        #
#  Offline NPSO tuning of the State-Space PD (SS-PD) gimbal controller #
########################################################################

import rclpy, random, time, subprocess, signal, json
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# ─── Sampling / evaluation timing ──────────────────────────────────────────
Ts = 0.001  # controller sampling period, Sec. 4.4 (Eq. 9)
EVAL_DURATION_S = 100.0 # set this to match your actual experimental setup
EVAL_MAX_SAMPLES = int(EVAL_DURATION_S / Ts)

# ─── Search space Omega and velocity limits (Eq. 32) ───────────────────────
KP_RANGE = (7.0, 14.0)
KD_RANGE = (0.0, 0.70)
OMEGA_RANGE = (20.0, 1000.0)

V_MAX = np.array([0.7, 0.07, 100.0])   # componentwise |v| <= vmax (Eq. 29)
V_MIN = -V_MAX

# ─── Fitness normalization constants (Eq. 26) ──────────────────────────────
C_MAE = 50.0   # px
C_VOL = 5.0    # px
C_ACC = 2.0    # px
C_ZC = 1000.0  # crossings

# ─── PSO coefficient schedule (Eq. 31) ─────────────────────────────────────
def _pso_coefficients(k: int, n_it: int):
    """w_k, c1_k, c2_k as functions of the normalized iteration index
    rho_k = k / n_it (Eq. 31)."""
    rho = k / n_it
    w = 0.9 - 0.5 * rho
    c1 = 2.0 * (1.0 - rho ** 1.5)
    c2 = 2.0 * (rho ** 1.5)
    return w, c1, c2


# ─── Single-axis State-Space PD controller (Eqs. 11-20) ───────────────────
class SSPD:
    """
    Single-axis state-space PD controller, following the observer-based
    state-feedback structure of Tan et al. (Cite 6) simplified to the
    two-state, integral-free, PD.

    This class is kept in sync with, and documents exactly, the control
    law implemented by the ROS 2 'gimbal_controller' node that is
    launched during the closed-loop fitness evaluation below. It is not
    itself used to simulate the plant (the real Isaac Sim closed loop
    is what actually gets evaluated), but is provided so both
    implementations stay verifiably consistent, and so the controller
    can be exercised/unit-tested standalone if desired.
    """

    def __init__(self, kp: float, kd: float, omega_o: float, dt: float = Ts):
        self.kp = kp
        self.kd = kd
        self.omega_o = omega_o
        self.dt = dt

        # Observer feedback coefficients (Eq. 14): poles both at -omega_o
        self.beta1 = omega_o ** 2      # beta_1 = omega_o^2
        self.beta2 = 2.0 * omega_o     # beta_2 = 2*omega_o

        self.x1 = 0.0  # estimate of y_dot   (x_bar_1, Eq. 13)
        self.x2 = 0.0  # estimate of y       (x_bar_2, Eq. 13)
        self.u_prev = 0.0  # u[k-1], stored explicitly (Eq. 16)

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.u_prev = 0.0

    def compute(self, alpha: float) -> float:
        """
        alpha: angular line-of-sight error for this axis (Eq. 7).
        Returns u[k] (Eq. 20).
        """
        # Eq. (12): r_i[k] = 0, y_i[k] = -alpha_i[k]
        y = -alpha

        # Observer update (Eq. 16 / 18), Euler-integrated at Ts, using
        # the control action applied in the previous sample.
        x1_dot = self.u_prev + self.beta1 * (y - self.x2)
        x2_dot = self.x1 + self.beta2 * (y - self.x2)
        self.x1 += self.dt * x1_dot
        self.x2 += self.dt * x2_dot

        # Control law (Eq. 20): u[k] = -KD*x1[k] - KP*x2[k]
        u = -self.kd * self.x1 - self.kp * self.x2

        self.u_prev = u  # becomes u[k-1] for the next sample
        return u


# ─── Particle ───────────────────────────────────────────────────────────────
@dataclass
class Particle:
    kp: float
    kd: float
    omega_o: float
    # Velocity
    v_kp: float = 0.0
    v_kd: float = 0.0
    v_omega_o: float = 0.0
    # Personal best
    b_kp: float = 0.0
    b_kd: float = 0.0
    b_omega_o: float = 20.0
    b_fitness: float = float('inf')
    # Current state
    fitness: float = float('inf')
    fitness_components: Optional[dict] = None
    iteration: int = 0
    particle_id: int = 0


# ─── Data collector ─────────────────────────────────────────────────────────
class DataCollectorNode(Node):
    """
    Subscribes to the signed image-plane pixel errors [e_x, e_y] published
    on /pid_log by the 'gimbal_controller' node (see controller.py),
    matching e_t as defined in Eq. (5) of Tan et al. (Cite 6). 
    """

    def __init__(self):
        super().__init__('data_collector')
        self.data = []
        self.max_samples = EVAL_MAX_SAMPLES
        self.collecting = False
        self.subscription = self.create_subscription(
            Float64MultiArray, '/pid_log', self.log_callback, 1000)

    def log_callback(self, msg):
        if self.collecting and len(self.data) < self.max_samples:
            # data[0] = e_x,t [px], data[1] = e_y,t [px] (Eq. 5)
            self.data.append([msg.data[0], msg.data[1]])

    def start_collection(self):
        self.data = []
        self.collecting = True
        print('   -> Starting data collection...')

    def stop_collection(self):
        self.collecting = False
        print(f'   -> Collection completed: {len(self.data)} samples')
        return np.array(self.data)


# ─── Report ─────────────────────────────────────────────────────────────────
class OptimizationReport:
    def __init__(self, num_particles, num_iterations):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.start_time = datetime.now()
        self.all_particles: List[Particle] = []
        self.best_per_iteration: List[Particle] = []

    def _copy(self, p: Particle) -> Particle:
        return Particle(
            kp=p.kp, kd=p.kd, omega_o=p.omega_o,
            fitness=p.fitness,
            fitness_components=p.fitness_components.copy() if p.fitness_components else None,
            iteration=p.iteration,
            particle_id=p.particle_id
        )

    def add_particle(self, p: Particle):
        self.all_particles.append(self._copy(p))

    def add_iteration_best(self, p: Particle):
        self.best_per_iteration.append(self._copy(p))

    def get_sorted_particles(self) -> List[Particle]:
        return sorted(self.all_particles, key=lambda p: p.fitness)

    def to_dict(self):
        sorted_particles = self.get_sorted_particles()

        def _p_dict(p, rank=None):
            d = {
                'fitness': p.fitness,
                'iteration': p.iteration,
                'particle_id': p.particle_id,
                'kp': p.kp,
                'kd': p.kd,
                'omega_o': p.omega_o,
                'fitness_components': p.fitness_components
            }
            if rank is not None:
                d['rank'] = rank
            return d

        return {
            'METADATA': {
                'num_particles': self.num_particles,
                'num_iterations_total_rounds': self.num_iterations,
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'total_evaluations': len(self.all_particles),
            },
            'BEST_PARTICLE': _p_dict(sorted_particles[0]) if sorted_particles else None,
            'TOP_10_PARTICLES': [
                _p_dict(p, rank=idx + 1)
                for idx, p in enumerate(sorted_particles[:10])
            ],
        }


# ─── Main optimiser ───────────────────────────────────────────────────────────
class NPSO_SSPD_Optimizer:
    """
    Implements the following offline NPSO search procedure:
    - 20 particles, 11 total evaluation rounds -> 220 fitness evaluations
      (round 0 = initial random sampling, rounds 1..10 = updates using
      the schedule rho_k = k / N_IT with N_IT = 11, Eq. 31).
    - eta = [Kp, Kd, omega_o]^T (Eq. 11); search space Omega (Eq. 32).
    - Position update uses the global best (Eq. 30), not the particle's
      own previous position.
    """

    NUM_PARTICLES = 20
    N_IT = 11  # total iteration count used inside rho_k = k/N_IT (Eq. 31)
    NUM_UPDATE_ROUNDS = N_IT - 1

    def __init__(self):
        self.kp_range = KP_RANGE
        self.kd_range = KD_RANGE
        self.omega_range = OMEGA_RANGE

        self.best_global_particle: Optional[Particle] = None
        self.best_global_fitness = float('inf')

        self.report = OptimizationReport(self.NUM_PARTICLES, self.N_IT)

        rclpy.init()
        self.collector_node = DataCollectorNode()

    # ── Particle factory ──────────────────────────────────────────────────
    def _create_particle(self) -> Particle:
        return Particle(
            kp=random.uniform(*self.kp_range),
            kd=random.uniform(*self.kd_range),
            omega_o=random.uniform(*self.omega_range),
            v_kp=random.uniform(V_MIN[0], V_MAX[0]),
            v_kd=random.uniform(V_MIN[1], V_MAX[1]),
            v_omega_o=random.uniform(V_MIN[2], V_MAX[2]),
        )

    # ── Launch controller process ─────────────────────────────────────────
    def _launch_controller(self, particle: Particle):
        cmd = [
            'ros2', 'run', 'sspid2', 'controller_node',
            '--ros-args',
            '-p', f'kp_yaw:={particle.kp}',
            '-p', f'kd_yaw:={particle.kd}',
            '-p', f'omega_o_yaw:={particle.omega_o}',
            '-p', f'kp_pitch:={particle.kp}',
            '-p', f'kd_pitch:={particle.kd}',
            '-p', f'omega_o_pitch:={particle.omega_o}',
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        )

    # ── Fitness function (Eqs. 22-28) ───────────────────────────────────────
    def _fitness(self, data: np.ndarray, particle: Particle) -> float:
        if len(data) < 3:
            particle.fitness_components = {}
            return 10.0  # highest penalty if insufficient data collected

        ex = data[:, 0]
        ey = data[:, 1]
        ep = np.sqrt(ex ** 2 + ey ** 2)  # radial tracking error

        # 1. Mean absolute error (Eq. 22)
        j_mae = np.mean(ep)

        # 2. Volatility: std of first difference of radial error (Eq. 23)
        d_ep = np.diff(ep)
        j_vol = np.std(d_ep)

        # 3. Acceleration-like term: mean |second difference| (Eq. 24)
        d2_ep = ep[2:] - 2 * ep[1:-1] + ep[:-2]
        j_acc = np.mean(np.abs(d2_ep))

        # 4. Zero-crossings of e_x and e_y (Eq. 25)
        sign_x = np.sign(ex)
        sign_y = np.sign(ey)
        n_zc_x = np.sum(sign_x[:-1] != sign_x[1:])
        n_zc_y = np.sum(sign_y[:-1] != sign_y[1:])
        j_zc = n_zc_x + n_zc_y

        # Normalization (Eq. 26): J_hat_m = min(1, J_m / c_m)
        j_mae_hat = min(1.0, j_mae / C_MAE)
        j_vol_hat = min(1.0, j_vol / C_VOL)
        j_acc_hat = min(1.0, j_acc / C_ACC)
        j_zc_hat = min(1.0, j_zc / C_ZC)

        particle.fitness_components = {
            'mae': j_mae_hat,
            'volatility': j_vol_hat,
            'acceleration': j_acc_hat,
            'zero_crossings': j_zc_hat,
        }

        # Scalar fitness (Eq. 27) -- weights are set in optimize()/main()
        total = (self.w_mae * j_mae_hat + self.w_vol * j_vol_hat +
                 self.w_acc * j_acc_hat + self.w_zc * j_zc_hat)
        return total

    # ── Collect closed-loop data for one candidate parameter vector ───────
    def _collect_real_data(self, particle, particle_num, total, round_idx, max_retries=3):
        info = (f"\n--------------------------------------------------\n"
                f"          Particle {particle_num}/{total} - Round {round_idx}/{self.N_IT - 1}\n\n"
                f" Kp={particle.kp:.4f}, Kd={particle.kd:.4f}, omega_o={particle.omega_o:.2f}")
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   RETRY {attempt}/{max_retries - 1}: Restarting...")
                time.sleep(5.0)
            print(f"   {info}")
            controller_process = None
            try:
                controller_process = self._launch_controller(particle)
                time.sleep(4.0)  # allow the controller / sim to settle
                self.collector_node.start_collection()
                start_time = time.time()
                while (time.time() - start_time < EVAL_DURATION_S) and \
                      (len(self.collector_node.data) < self.collector_node.max_samples):
                    rclpy.spin_once(self.collector_node, timeout_sec=0.001)
                data = self.collector_node.stop_collection()
                if len(data) > 0:
                    return data
                print("   Insufficient data collected. Retrying...")
            except Exception as e:
                print(f"   Error: {e}")
                if attempt == max_retries - 1:
                    return np.array([])
            finally:
                if controller_process is not None:
                    try:
                        controller_process.terminate()
                        controller_process.wait(timeout=2.0)
                    except Exception:
                        pass
                subprocess.run(['killall', '-9', 'gimbal_controller'],
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                time.sleep(3.0)
        return np.array([])

    # ── NPSO update (Eqs. 29-31) ────────────────────────────────────────────
    def _update_particle(self, particle: Particle, w: float, c1: float, c2: float):
        gb = self.best_global_particle
        r1 = [random.random(), random.random(), random.random()]
        r2 = [random.random(), random.random(), random.random()]

        def _vel(v, p_best, x, g_best, r1_i, r2_i):
            # Eq. (29): elementwise, using the particle's OWN position x
            return w * v + c1 * r1_i * (p_best - x) + c2 * r2_i * (g_best - x)

        def _clamp(x, lo, hi):
            return max(lo, min(hi, x))

        # Velocity update (uses own current position for both pulls)
        v_kp = _clamp(_vel(particle.v_kp, particle.b_kp, particle.kp, gb.kp, r1[0], r2[0]),
                       V_MIN[0], V_MAX[0])
        v_kd = _clamp(_vel(particle.v_kd, particle.b_kd, particle.kd, gb.kd, r1[1], r2[1]),
                       V_MIN[1], V_MAX[1])
        v_omega_o = _clamp(_vel(particle.v_omega_o, particle.b_omega_o, particle.omega_o,
                                 gb.omega_o, r1[2], r2[2]),
                            V_MIN[2], V_MAX[2])
        particle.v_kp, particle.v_kd, particle.v_omega_o = v_kp, v_kd, v_omega_o

        # Position update (Eq. 30): eta^(k+1) = Proj_Omega(g^(k) + v^(k+1))
        particle.kp = _clamp(gb.kp + v_kp, *self.kp_range)
        particle.kd = _clamp(gb.kd + v_kd, *self.kd_range)
        particle.omega_o = _clamp(gb.omega_o + v_omega_o, *self.omega_range)

    def _set_personal_best(self, particle: Particle):
        particle.b_kp = particle.kp
        particle.b_kd = particle.kd
        particle.b_omega_o = particle.omega_o
        particle.b_fitness = particle.fitness

    def _make_global_best(self, particle: Particle, round_idx: int) -> Particle:
        return Particle(
            kp=particle.kp, kd=particle.kd, omega_o=particle.omega_o,
            fitness=particle.fitness, iteration=round_idx,
            particle_id=particle.particle_id
        )

    # ── Main optimisation loop ────────────────────────────────────────────
    def optimize(self, w_mae=1.0, w_vol=1.0, w_zc=1.0, w_acc=1.0) -> Particle:
        """
        w_mae, w_vol, w_acc, w_zc: fitness weights (Eq. 27), e.g.:
          Experiment 2 (MAE only):             1, 0,   0,   0
          Experiment 3 (MAE + Volatility):     0, 1,   0,   0
          Experiment 4 (MAE + Zero-Crossings): 0, 0,   1,   0
          Experiment 5 (MAE + Acceleration):   0, 0,   0,   1
          Experiment 6 (equal weights):        1, 1,   1,   1
          Experiment 7 (reweighted):           1, 0.5, 0.5, 1
        """
        self.w_mae, self.w_vol, self.w_acc, self.w_zc = w_mae, w_vol, w_acc, w_zc

        particles = [self._create_particle() for _ in range(self.NUM_PARTICLES)]

        print(f"\n{'=' * 50}")
        print("NPSO OPTIMIZATION - STATE-SPACE PD GIMBAL TUNING")
        print(f"Particles: {self.NUM_PARTICLES}, Total rounds: {self.N_IT} "
              f"(1 initial + {self.NUM_UPDATE_ROUNDS} updates)")
        print(f"Weights: wMAE={w_mae}, wVol={w_vol}, wAcc={w_acc}, wZC={w_zc}")
        print(f"{'=' * 50}")

        try:
            for round_idx in range(self.N_IT):  # 0 .. N_IT-1  (11 total rounds)
                if round_idx == 0:
                    print(f"\n{'#' * 50}\n   INITIAL ROUND (0/{self.N_IT - 1})\n{'#' * 50}")
                else:
                    # Coefficients used to move FROM round_idx-1 TO round_idx
                    # use rho_{round_idx-1} = (round_idx-1)/N_IT (Eq. 31)
                    w, c1, c2 = _pso_coefficients(round_idx - 1, self.N_IT)
                    print(f"\n{'#' * 50}\n   ROUND {round_idx}/{self.N_IT - 1}\n{'#' * 50}")
                    print(f"  PSO params: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")

                for i, particle in enumerate(particles):
                    particle.iteration = round_idx
                    particle.particle_id = i + 1

                    if round_idx > 0:
                        self._update_particle(particle, w, c1, c2)

                    data = self._collect_real_data(particle, i + 1, self.NUM_PARTICLES, round_idx)
                    particle.fitness = self._fitness(data, particle)
                    print(f" Fitness = {particle.fitness:.5f}")
                    self.report.add_particle(particle)

                    if round_idx == 0 or particle.fitness < particle.b_fitness:
                        self._set_personal_best(particle)

                    if particle.fitness < self.best_global_fitness:
                        self.best_global_fitness = particle.fitness
                        self.best_global_particle = self._make_global_best(particle, round_idx)
                        print(f"   NEW GLOBAL BEST! Fitness = {particle.fitness:.5f}")

                round_best = min(particles, key=lambda p: p.fitness)
                self.report.add_iteration_best(round_best)

                print(f"\n{'=' * 50}")
                print(f"   Round {round_idx} summary:")
                print(f"       Round best:  Fitness = {round_best.fitness:.5f}")
                print(f"       Global best: Fitness = {self.best_global_fitness:.5f}")
                print(f"{'=' * 50}")

            return self.best_global_particle

        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            return self.best_global_particle

    def cleanup(self):
        print("\nCleaning up...")
        try:
            self.collector_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        subprocess.run(['killall', '-9', 'gimbal_controller'],
                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print("Done")
        time.sleep(1.0)
