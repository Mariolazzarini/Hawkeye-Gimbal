########################################################################
#  npsoSSPID.py                                                        #
#  Offline NPSO tuning of the State-Space PD (SS-PD) gimbal controller #
########################################################################

import rclpy, random, time, subprocess, signal
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────
Ts = 0.001
EVAL_DURATION_S = 30.0  # 30s to fit the deadline
EVAL_MAX_SAMPLES = int(EVAL_DURATION_S / Ts)

SETTLE_BEFORE_S = 4.0
CLEANUP_AFTER_S = 3.0
PARTICLE_DURATION = SETTLE_BEFORE_S + EVAL_DURATION_S + CLEANUP_AFTER_S

CONTROLLER_EXECUTABLE = 'controller_node'

KP_RANGE = (7.0, 14.0)
KD_RANGE = (0.0, 0.70)
OMEGA_RANGE = (20.0, 1000.0)
V_MAX = np.array([0.7, 0.07, 100.0])
V_MIN = -V_MAX

C_MAE = 50.0
C_VOL = 5.0
C_ACC = 2.0
C_ZC = 1000.0

DEFAULT_SEED: Optional[int] = None
EVAL_SCENARIO: str = "scenario_1_in_sample"

# ─── PSO Schedule ──────────────────────────────────────────────────────────
def _pso_coefficients(k: int, n_it: int):
    rho = k / n_it
    w = 0.9 - 0.5 * rho
    c1 = 2.0 * (1.0 - rho ** 1.5)
    c2 = 2.0 * (rho ** 1.5)
    return w, c1, c2

# ─── SS-PD Controller Model ────────────────────────────────────────────────
class SSPD:
    def __init__(self, kp: float, kd: float, omega_o: float, dt: float = Ts,
                 u_max: Optional[float] = None):
        self.kp, self.kd, self.omega_o, self.dt, self.u_max = kp, kd, omega_o, dt, u_max
        self.beta1 = omega_o ** 2
        self.beta2 = 2.0 * omega_o
        self.x1, self.x2, self.u_prev = 0.0, 0.0, 0.0

    def reset(self):
        self.x1, self.x2, self.u_prev = 0.0, 0.0, 0.0

    def compute(self, alpha: float, dt: float = None) -> float:
        dt = self.dt if dt is None else dt
        
        # Substepping to maintain Euler stability for large sim dt
        n_sub = max(1, int(np.ceil(dt / 0.002)))
        sub_dt = dt / n_sub
        y = -alpha
        u = self.u_prev
        
        for _ in range(n_sub):
            x1_dot = self.u_prev + self.beta1 * (y - self.x2)
            x2_dot = self.x1 + self.beta2 * (y - self.x2)
            self.x1 += sub_dt * x1_dot
            self.x2 += sub_dt * x2_dot
            
            u = -self.kd * self.x1 - self.kp * self.x2
            if self.u_max is not None:
                u = min(self.u_max, max(-self.u_max, u))
            self.u_prev = u
            
        return u

# ─── Data Structures ───────────────────────────────────────────────────────
@dataclass
class Particle:
    kp: float
    kd: float
    omega_o: float
    v_kp: float = 0.0
    v_kd: float = 0.0
    v_omega_o: float = 0.0
    b_kp: float = 0.0
    b_kd: float = 0.0
    b_omega_o: float = 0.0
    b_fitness: float = float('inf')
    fitness: float = float('inf')
    fitness_components: Optional[dict] = None
    iteration: int = 0
    particle_id: int = 0

# ─── ROS 2 Data Collector ──────────────────────────────────────────────────
class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.data = []
        self.max_samples = EVAL_MAX_SAMPLES
        self.collecting = False
        self.subscription = self.create_subscription(
            Float64MultiArray, '/pid_log', self.log_callback, 1000)
        self.clock = self.get_clock()

    def log_callback(self, msg):
        if self.collecting and len(self.data) < self.max_samples:
            self.data.append([msg.data[0], msg.data[1]])

    def start_collection(self):
        self.data = []
        self.collecting = True
        print('   -> Starting data collection...')

    def stop_collection(self):
        self.collecting = False
        print(f'   -> Collection completed: {len(self.data)} samples')
        return np.array(self.data)

# ─── Optimization Report ───────────────────────────────────────────────────
class OptimizationReport:
    def __init__(self, num_particles, num_iterations, seed=None, scenario=None):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.seed = seed
        self.scenario = scenario
        self.start_time = datetime.now()
        self.weights = None
        self.interrupted = False
        self.all_particles: List[Particle] = []
        self.best_per_iteration: List[Particle] = []

    def _copy(self, p: Particle) -> Particle:
        return Particle(
            kp=p.kp, kd=p.kd, omega_o=p.omega_o,
            fitness=p.fitness,
            fitness_components=p.fitness_components.copy() if p.fitness_components else None,
            iteration=p.iteration, particle_id=p.particle_id
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
                'fitness': p.fitness, 'iteration': p.iteration, 'particle_id': p.particle_id,
                'kp': p.kp, 'kd': p.kd, 'omega_o': p.omega_o,
                'fitness_components': p.fitness_components
            }
            if rank is not None: d['rank'] = rank
            return d

        return {
            'METADATA': {
                'num_particles': self.num_particles,
                'num_iterations_total_rounds': self.num_iterations,
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'total_evaluations': len(self.all_particles),
                'interrupted': self.interrupted, 'weights': self.weights,
                'seed': self.seed, 'scenario': self.scenario,
                'eval_duration_sec': EVAL_DURATION_S, 'sampling_period_sec': Ts,
                'search_space': {'kp': list(KP_RANGE), 'kd': list(KD_RANGE), 'omega_o': list(OMEGA_RANGE)},
                'normalizers': {'mae': C_MAE, 'volatility': C_VOL, 'acceleration': C_ACC, 'zero_crossings': C_ZC},
            },
            'BEST_PARTICLE': _p_dict(sorted_particles[0]) if sorted_particles else None,
            'TOP_10_PARTICLES': [_p_dict(p, rank=idx + 1) for idx, p in enumerate(sorted_particles[:10])],
        }

# ─── Main Optimiser ────────────────────────────────────────────────────────
class NPSO_SSPD_Optimizer:
    def __init__(self, num_particles: int = 20, num_iterations: int = 10,
                 seed: Optional[int] = None, scenario: Optional[str] = None):
        if num_particles < 1 or num_iterations < 1:
            raise ValueError('num_particles and num_iterations must be >= 1')

        self.NUM_PARTICLES = int(num_particles)
        self.NUM_UPDATE_ROUNDS = int(num_iterations)
        self.N_IT = self.NUM_UPDATE_ROUNDS + 1

        self.kp_range, self.kd_range, self.omega_range = KP_RANGE, KD_RANGE, OMEGA_RANGE

        self.seed = seed if seed is not None else DEFAULT_SEED
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        else:
            print("WARNING: no seed set. Runs will not be reproducible.")

        self.scenario = scenario if scenario is not None else EVAL_SCENARIO
        self.w_mae, self.w_vol, self.w_acc, self.w_zc = 1.0, 1.0, 1.0, 1.0
        self.best_global_particle: Optional[Particle] = None
        self.best_global_fitness = float('inf')
        self.best_history = []

        self.report = OptimizationReport(self.NUM_PARTICLES, self.N_IT, seed=self.seed, scenario=self.scenario)
        rclpy.init()
        self.collector_node = DataCollectorNode()

    def _create_particle(self) -> Particle:
        return Particle(
            kp=random.uniform(*self.kp_range), kd=random.uniform(*self.kd_range), omega_o=random.uniform(*self.omega_range),
            v_kp=random.uniform(V_MIN[0], V_MAX[0]), v_kd=random.uniform(V_MIN[1], V_MAX[1]), v_omega_o=random.uniform(V_MIN[2], V_MAX[2])
        )

    def _launch_controller(self, particle: Particle):
        cmd = [
            'ros2', 'run', 'sspid2', CONTROLLER_EXECUTABLE,
            '--ros-args', '-p', 'use_sim_time:=true',
            '-p', f'kp_yaw:={particle.kp}', '-p', f'kd_yaw:={particle.kd}', '-p', f'omega_o_yaw:={particle.omega_o}',
            '-p', f'kp_pitch:={particle.kp}', '-p', f'kd_pitch:={particle.kd}', '-p', f'omega_o_pitch:={particle.omega_o}',
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

    @staticmethod
    def _kill_stale_controllers():
        subprocess.run(['pkill', '-9', '-f', CONTROLLER_EXECUTABLE], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def _fitness(self, data: np.ndarray, particle: Particle) -> float:
        if len(data) < 3:
            particle.fitness_components = {}
            return 10.0

        ex, ey = data[:, 0], data[:, 1]
        ep = np.sqrt(ex ** 2 + ey ** 2)

        j_mae = np.mean(ep)
        j_vol = np.std(np.diff(ep))
        j_acc = np.mean(np.abs(ep[2:] - 2 * ep[1:-1] + ep[:-2]))
        
        sign_x = _sign_no_zero(ex)
        sign_y = _sign_no_zero(ey)
        j_zc = np.sum(sign_x[:-1] != sign_x[1:]) + np.sum(sign_y[:-1] != sign_y[1:])

        j_mae_hat = min(1.0, j_mae / C_MAE)
        j_vol_hat = min(1.0, j_vol / C_VOL)
        j_acc_hat = min(1.0, j_acc / C_ACC)
        j_zc_hat = min(1.0, j_zc / C_ZC)

        particle.fitness_components = {
            'mae': j_mae_hat, 'volatility': j_vol_hat, 'acceleration': j_acc_hat,
            'zero_crossings': j_zc_hat, 'samples': int(len(data)),
        }

        return (self.w_mae * j_mae_hat + self.w_vol * j_vol_hat + self.w_acc * j_acc_hat + self.w_zc * j_zc_hat)

    def _collect_real_data(self, particle, particle_num, total, round_idx, max_retries=3):
        info = (f"\n--------------------------------------------------\n"
                f"          Particle {particle_num}/{total} - Round {round_idx}/{self.N_IT - 1}\n\n"
                f" Kp={particle.kp:.4f}, Kd={particle.kd:.4f}, omega_o={particle.omega_o:.2f}")
        for attempt in range(max_retries):
            if attempt > 0: time.sleep(5.0)
            print(f"   {info}")
            controller_process = None
            try:
                controller_process = self._launch_controller(particle)
                time.sleep(SETTLE_BEFORE_S)
                self.collector_node.start_collection()
                
                start_time = self.collector_node.clock.now()
                while ((self.collector_node.clock.now() - start_time).nanoseconds / 1e9 < EVAL_DURATION_S) and \
                      (len(self.collector_node.data) < self.collector_node.max_samples):
                    rclpy.spin_once(self.collector_node, timeout_sec=0.001)
                
                data = self.collector_node.stop_collection()
                if len(data) > 0: return data
            except Exception as e:
                if attempt == max_retries - 1: return np.array([])
            finally:
                if controller_process is not None:
                    try:
                        controller_process.terminate()
                        controller_process.wait(timeout=2.0)
                    except Exception:
                        try: controller_process.kill()
                        except Exception: pass
                self._kill_stale_controllers()
                time.sleep(CLEANUP_AFTER_S)
        return np.array([])

    def _update_particle(self, particle: Particle, w: float, c1: float, c2: float, gb: Optional[Particle]):
        if gb is None: return

        particle.kp = gb.kp
        particle.kd = gb.kd
        particle.omega_o = gb.omega_o
        
        particle.v_kp = random.uniform(V_MIN[0], V_MAX[0])
        particle.v_kd = random.uniform(V_MIN[1], V_MAX[1])
        particle.v_omega_o = random.uniform(V_MIN[2], V_MAX[2])

        r1 = [random.random(), random.random(), random.random()]
        r2 = [random.random(), random.random(), random.random()]

        def _vel(v, p_best, x, g_best, r1_i, r2_i):
            return w * v + c1 * r1_i * (p_best - x) + c2 * r2_i * (g_best - x)

        def _clamp(x, lo, hi): return max(lo, min(hi, x))

        v_kp = _clamp(_vel(particle.v_kp, particle.b_kp, particle.kp, gb.kp, r1[0], r2[0]), V_MIN[0], V_MAX[0])
        v_kd = _clamp(_vel(particle.v_kd, particle.b_kd, particle.kd, gb.kd, r1[1], r2[1]), V_MIN[1], V_MAX[1])
        v_omega_o = _clamp(_vel(particle.v_omega_o, particle.b_omega_o, particle.omega_o, gb.omega_o, r1[2], r2[2]), V_MIN[2], V_MAX[2])
        
        particle.v_kp, particle.v_kd, particle.v_omega_o = v_kp, v_kd, v_omega_o
        particle.kp = _clamp(gb.kp + v_kp, *self.kp_range)
        particle.kd = _clamp(gb.kd + v_kd, *self.kd_range)
        particle.omega_o = _clamp(gb.omega_o + v_omega_o, *self.omega_range)

    def _set_personal_best(self, particle: Particle):
        particle.b_kp, particle.b_kd, particle.b_omega_o, particle.b_fitness = particle.kp, particle.kd, particle.omega_o, particle.fitness

    def _make_global_best(self, particle: Particle, round_idx: int) -> Particle:
        return Particle(kp=particle.kp, kd=particle.kd, omega_o=particle.omega_o, fitness=particle.fitness, iteration=round_idx, particle_id=particle.particle_id)

    def optimize(self, w_mae=1.0, w_vol=1.0, w_acc=1.0, w_zc=1.0) -> Particle:
        self.w_mae, self.w_vol, self.w_acc, self.w_zc = w_mae, w_vol, w_acc, w_zc
        self.report.weights = {'mae': w_mae, 'volatility': w_vol, 'acceleration': w_acc, 'zero_crossings': w_zc}

        particles = [self._create_particle() for _ in range(self.NUM_PARTICLES)]

        print(f"\n{'=' * 50}")
        print("NPSO OPTIMIZATION - STATE-SPACE PD GIMBAL TUNING")
        print(f"Particles: {self.NUM_PARTICLES}, Total rounds: {self.N_IT}")
        print(f"Weights: wMAE={w_mae}, wVol={w_vol}, wAcc={w_acc}, wZC={w_zc}")
        print(f"Seed: {self.seed}, Scenario: {self.scenario}")
        print(f"{'=' * 50}")

        w = c1 = c2 = 0.0
        round_gbest = None
        try:
            for round_idx in range(self.N_IT):
                if round_idx == 0:
                    print(f"\n{'#' * 50}\n   INITIAL ROUND (0/{self.N_IT - 1})\n{'#' * 50}")
                else:
                    w, c1, c2 = _pso_coefficients(round_idx - 1, self.N_IT)
                    print(f"\n{'#' * 50}\n   ROUND {round_idx}/{self.N_IT - 1}\n{'#' * 50}")
                    print(f"  PSO params: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
                    round_gbest = self.best_global_particle

                for i, particle in enumerate(particles):
                    particle.iteration = round_idx
                    particle.particle_id = i + 1

                    if round_idx > 0:
                        self._update_particle(particle, w, c1, c2, round_gbest)

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
                self.best_history.append((round_idx, self.best_global_fitness))

                print(f"\n{'=' * 50}")
                print(f"   Round {round_idx} summary:")
                print(f"       Round best:  Fitness = {round_best.fitness:.5f}")
                print(f"       Global best: Fitness = {self.best_global_fitness:.5f}")
                print(f"{'=' * 50}")

            return self.best_global_particle

        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            self.report.interrupted = True
            return self.best_global_particle

    def save_convergence_plot(self, output_path: str):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed; skipping convergence plot.")
            return

        if not self.best_history:
            print("No optimisation history to plot.")
            return

        iterations = [h[0] for h in self.best_history]
        best_so_far = [h[1] for h in self.best_history]
        round_best_all = [p.fitness for p in self.report.best_per_iteration]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, best_so_far, 'o-', linewidth=2, markersize=6, color='#1f77b4', label='Global best (best so far)')

        if len(round_best_all) == len(iterations):
            ax.plot(iterations, round_best_all, 's--', linewidth=1.2, markersize=5, color='#ff7f0e', alpha=0.7, label='Round best')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Fitness value', fontsize=12)

        title = 'NPSO convergence'
        if self.report.weights:
            w = self.report.weights
            title += f"\n(MAE={w['mae']}, vol={w['volatility']}, acc={w['acceleration']}, zc={w['zero_crossings']})"
        ax.set_title(title, fontsize=11)

        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        ax.set_xticks(iterations)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"📈 Saved convergence plot: {output_path}")

    def cleanup(self):
        print("\nCleaning up...")
        try: self.collector_node.destroy_node()
        except Exception: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass
        self._kill_stale_controllers()
        print("Done")
        time.sleep(1.0)

def _sign_no_zero(x: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(x, dtype=float))
    if s.size == 0: return s
    zeros = s == 0
    if not zeros.any(): return s
    idx = np.where(~zeros, np.arange(s.size), 0)
    np.maximum.accumulate(idx, out=idx)
    s = s[idx]
    s[s == 0] = 1.0
    return s
