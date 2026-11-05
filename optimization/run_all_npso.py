#!/usr/bin/env python3
# ==============================================================================
# FLAGS EXPLANATION:
#   --seed <int>         : (Required) Random seed for NPSO.
#   --experiments <ids>  : List of experiments to run (e.g., --experiments 2 5 6).
#   --particles <int>    : Number of particles (default: 20).
#   --iterations <int>   : Number of iterations (default: 10).
#   --secs <float>       : Evaluation time per particle in seconds (default: 50).
#   --default            : Resets to standard values (20 particles, 10 iters, 50 secs).
#   --force              : Overwrites existing results (re-runs finished experiments).
#   --yes                : Skips the confirmation prompt before starting.
# ==============================================================================
"""
run_all_npso.py
===============
Batch runner for the six NPSO fitness configurations of the manuscript
(Experiments 2-7, Table 3). Replaces run_experiment.sh + run_all_npso.py.

Starts distance_node (YOLO) in the background, then for each experiment runs
the State-Space PD optimiser (default: 20 particles, 1 initial + 10 update
rounds = 220 evaluations) with that experiment's fitness weights.
Reports are saved as opt_exp{N}_seed{S}.json under REPORT_DIR.
Experiments with a completed report are skipped unless --force is passed.

Usage (sources ~/setup_py311.sh by itself; any python3 can launch it):
    python3 run_all_npso.py --seed 1 [--experiments 2 5 6] [--particles N]
        [--iterations N] [--secs S] [--default] [--force] [--yes]
"""

import os, sys

# Re-launch itself inside a shell that has sourced the Python 3.11 / ROS environment.
if os.environ.get('NPSO_ENV_SOURCED') != '1':
    os.environ['NPSO_ENV_SOURCED'] = '1'
    os.execvp('bash', ['bash', '-c', 'source ~/setup_py311.sh && exec python3 "$0" "$@"',
                       os.path.abspath(__file__), *sys.argv[1:]])

import rclpy, random, time, subprocess, signal, json, argparse
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from simulation_interfaces.srv import SetSimulationState
from simulation_interfaces.msg import SimulationState
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configurations ────────────────────────────────────────────────────────
Ts = 0.001
EVAL_DURATION_S = 100.0
EVAL_MAX_SAMPLES = int(EVAL_DURATION_S / Ts)

SETTLE_BEFORE_S = 4.0
CLEANUP_AFTER_S = 3.0
PARTICLE_DURATION = SETTLE_BEFORE_S + EVAL_DURATION_S + CLEANUP_AFTER_S

CONTROLLER_EXECUTABLE = 'controller_node'
WORKSPACE_DIR = os.path.expanduser("~/ros_hawkeye/robotics_project/ws_new2/SSPID_ws2_py311")
REPORT_DIR = os.path.join(WORKSPACE_DIR, "optimization_results")

KP_RANGE = (7.0, 14.0)
KD_RANGE = (0.0, 0.70)
OMEGA_RANGE = (20.0, 1000.0)
V_MAX = np.array([0.7, 0.07, 100.0])
V_MIN = -V_MAX

EXPERIMENTS = {
    2: {'name': 'MAE',                      'w': (1.0, 0.0, 0.0, 0.0)},
    3: {'name': 'MAE + Volatility',         'w': (1.0, 1.0, 0.0, 0.0)},
    4: {'name': 'MAE + Zero-crossings',     'w': (1.0, 0.0, 0.0, 1.0)},
    5: {'name': 'MAE + Acceleration',       'w': (1.0, 0.0, 1.0, 0.0)},
    6: {'name': 'Combined (equal weights)', 'w': (1.0, 1.0, 1.0, 1.0)},
    7: {'name': 'Combined (reweighted)',    'w': (1.0, 0.5, 0.5, 1.0)},
}

NUM_PARTICLES = 20
NUM_ITERATIONS = 10

# ─── PSO Schedule ──────────────────────────────────────────────────────────
def _pso_coefficients(k: int, n_it: int):
    rho = k / max(n_it - 2, 1)
    w_max = 2.0
    w_min = 0.4
    w = w_max - rho * (w_max - w_min)
    c1 = 2.0
    c2 = 2.0
    return w, c1, c2

def _sign_no_zero(x: np.ndarray) -> np.ndarray:
    """Sign of x where exact zeros inherit the previous sign (no fake zero-crossings)."""
    s = np.sign(np.asarray(x, dtype=float))
    zeros = s == 0
    if s.size == 0 or not zeros.any():
        return s
    idx = np.where(~zeros, np.arange(s.size), 0)
    np.maximum.accumulate(idx, out=idx)
    s = s[idx]
    s[s == 0] = 1.0
    return s

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
    raw: Optional[dict] = None
    iteration: int = 0
    particle_id: int = 0

class OptimizationReport:
    def __init__(self, num_particles, num_iterations, seed=None, scenario=None):
        self.num_particles, self.num_iterations, self.seed, self.scenario = num_particles, num_iterations, seed, scenario
        self.start_time = datetime.now()
        self.weights = None
        self.normalizers = None
        self.interrupted = False
        self.all_particles: List[Particle] = []
        self.best_per_iteration: List[Particle] = []

    def _copy(self, p: Particle) -> Particle:
        return Particle(kp=p.kp, kd=p.kd, omega_o=p.omega_o, fitness=p.fitness,
                        fitness_components=p.fitness_components.copy() if p.fitness_components else None,
                        iteration=p.iteration, particle_id=p.particle_id)

    def add_particle(self, p: Particle): self.all_particles.append(self._copy(p))
    def add_iteration_best(self, p: Particle): self.best_per_iteration.append(self._copy(p))

    def to_dict(self):
        best = min(self.all_particles, key=lambda p: p.fitness) if self.all_particles else None
        return {
            'METADATA': {
                'num_particles': self.num_particles, 'num_iterations_total_rounds': self.num_iterations,
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'total_evaluations': len(self.all_particles), 'interrupted': self.interrupted,
                'weights': self.weights, 'normalizers': self.normalizers, 'seed': self.seed, 'scenario': self.scenario,
                'eval_duration_sec': EVAL_DURATION_S, 'sampling_period_sec': Ts,
            },
            'BEST_PARTICLE': {'fitness': best.fitness, 'iteration': best.iteration, 'particle_id': best.particle_id,
                              'kp': best.kp, 'kd': best.kd, 'omega_o': best.omega_o,
                              'fitness_components': best.fitness_components} if best else None,
        }

# ─── ROS 2 Node ────────────────────────────────────────────────────────────
class OptimizerNode(Node):
    def __init__(self):
        super().__init__('npso_optimizer')
        self.data = []
        self.collecting = False
        self.create_subscription(Float64MultiArray, '/pid_log', self.log_callback, 1000)

        self.sim_client = self.create_client(SetSimulationState, '/isaacsim/SetSimulationState')
        if not self.sim_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Isaac Sim sim control service not found. Ensure extension is enabled.')
            raise RuntimeError("Simulation control service unavailable.")

    def log_callback(self, msg):
        if self.collecting and len(self.data) < EVAL_MAX_SAMPLES:
            self.data.append([msg.data[0], msg.data[1]])

    def set_sim_state(self, state: int):
        req = SetSimulationState.Request()
        req.state.state = state
        future = self.sim_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

# ─── Optimiser Class ───────────────────────────────────────────────────────
class NPSO_SSPD_Optimizer:
    def __init__(self, node: OptimizerNode, num_particles: int, num_iterations: int, seed: int, scenario: str):
        self.node = node
        self.NUM_PARTICLES = int(num_particles)
        self.N_IT = int(num_iterations) + 1  # 1 initial round + update rounds

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.scenario = scenario
        self.best_global_particle: Optional[Particle] = None
        self.best_global_fitness = float('inf')
        self.best_history = []
        self.report = OptimizationReport(self.NUM_PARTICLES, self.N_IT, seed=seed, scenario=scenario)
        self.w_mae = self.w_vol = self.w_acc = self.w_zc = 1.0
        self.C = None  # normalisation constants, set after the initial round

    def _create_particle(self) -> Particle:
        return Particle(
            kp=random.uniform(*KP_RANGE), kd=random.uniform(*KD_RANGE), omega_o=random.uniform(*OMEGA_RANGE),
            v_kp=random.uniform(V_MIN[0], V_MAX[0]), v_kd=random.uniform(V_MIN[1], V_MAX[1]), v_omega_o=random.uniform(V_MIN[2], V_MAX[2])
        )

    def _launch_controller(self, p: Particle):
        cmd = ['ros2', 'run', 'sspid2', CONTROLLER_EXECUTABLE, '--ros-args', '-p', 'use_sim_time:=true',
               '-p', f'kp_yaw:={p.kp}', '-p', f'kd_yaw:={p.kd}', '-p', f'omega_o_yaw:={p.omega_o}',
               '-p', f'kp_pitch:={p.kp}', '-p', f'kd_pitch:={p.kd}', '-p', f'omega_o_pitch:={p.omega_o}']
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

    @staticmethod
    def _kill_stale_controllers():
        subprocess.run(['pkill', '-9', '-f', CONTROLLER_EXECUTABLE], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    @staticmethod
    def _raw_metrics(data: np.ndarray) -> Optional[dict]:
        if len(data) < 3:
            return None
        ex, ey = data[:, 0], data[:, 1]
        ep = np.sqrt(ex**2 + ey**2)
        sx, sy = _sign_no_zero(ex), _sign_no_zero(ey)
        return {'mae': np.mean(ep),
                'volatility': np.std(np.diff(ep)),
                'acceleration': np.mean(np.abs(ep[2:] - 2 * ep[1:-1] + ep[:-2])),
                'zero_crossings': np.sum(sx[:-1] != sx[1:]) + np.sum(sy[:-1] != sy[1:]),
                'samples': int(len(data))}

    def _compute_normalizers(self, particles):
        """C_m = 90th percentile of metric m over the initial-round evaluations (Eq. 29)."""
        valid = [p.raw for p in particles if p.raw is not None]
        self.C = {k: max(np.percentile([r[k] for r in valid], 90), 1e-12) if valid else 1.0
                  for k in ('mae', 'volatility', 'acceleration', 'zero_crossings')}
        self.report.normalizers = self.C

    def _fitness(self, p: Particle) -> float:
        if p.raw is None:
            p.fitness_components = {}
            return 10.0
        c = {k: p.raw[k] / (p.raw[k] + self.C[k]) for k in self.C}  # J / (J + C)
        p.fitness_components = {**c, 'samples': p.raw['samples']}
        return self.w_mae * c['mae'] + self.w_vol * c['volatility'] + self.w_acc * c['acceleration'] + self.w_zc * c['zero_crossings']

    def _score(self, particle: Particle, round_idx: int):
        particle.fitness = self._fitness(particle)
        print(f" Fitness = {particle.fitness:.5f}")
        self.report.add_particle(particle)

        if round_idx == 0 or particle.fitness < particle.b_fitness:
            particle.b_kp, particle.b_kd, particle.b_omega_o, particle.b_fitness = particle.kp, particle.kd, particle.omega_o, particle.fitness

        if particle.fitness < self.best_global_fitness:
            self.best_global_fitness = particle.fitness
            self.best_global_particle = Particle(kp=particle.kp, kd=particle.kd, omega_o=particle.omega_o, fitness=particle.fitness,
                                                 iteration=round_idx, particle_id=particle.particle_id)
            print(f"   NEW GLOBAL BEST! Fitness = {particle.fitness:.5f}")

    def _collect_real_data(self, particle, particle_num, total, round_idx, max_retries=3):
        info = (f"\n--------------------------------------------------\n          Particle {particle_num}/{total} - Round {round_idx}/{self.N_IT - 1}\n"
                f" Kp={particle.kp:.4f}, Kd={particle.kd:.4f}, omega_o={particle.omega_o:.2f}")

        for attempt in range(max_retries):
            if attempt > 0: time.sleep(5.0)
            print(f"   {info}")
            controller_process = None
            try:
                self.node.set_sim_state(SimulationState.STATE_STOPPED)
                time.sleep(1.0)

                controller_process = self._launch_controller(particle)
                time.sleep(SETTLE_BEFORE_S)

                self.node.set_sim_state(SimulationState.STATE_PLAYING)

                self.node.data = []
                self.node.collecting = True

                start_time = time.time()
                print("   -> Starting data collection...")
                while (time.time() - start_time < EVAL_DURATION_S) and \
                      (len(self.node.data) < EVAL_MAX_SAMPLES):
                    rclpy.spin_once(self.node, timeout_sec=0.001)

                self.node.collecting = False
                data = np.array(self.node.data)
                print(f"   -> Collection completed: {len(data)} samples")

                if len(data) > 0:
                    return data
            except Exception as e:
                print(f"   Error: {e}")
            finally:
                self.node.collecting = False
                self.node.set_sim_state(SimulationState.STATE_STOPPED)
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

    def _update_particle(self, p: Particle, w: float, c1: float, c2: float, gb: Particle):
        # NPSO: re-initialise at the previous global best with a random velocity, then apply the PSO update.
        p.kp, p.kd, p.omega_o = gb.kp, gb.kd, gb.omega_o
        p.v_kp, p.v_kd, p.v_omega_o = random.uniform(V_MIN[0], V_MAX[0]), random.uniform(V_MIN[1], V_MAX[1]), random.uniform(V_MIN[2], V_MAX[2])

        r1 = [random.random() for _ in range(3)]
        r2 = [random.random() for _ in range(3)]

        def _vel(v, pb, x, gbest, r1i, r2i): return w * v + c1 * r1i * (pb - x) + c2 * r2i * (gbest - x)
        def _clamp(x, lo, hi): return max(lo, min(hi, x))

        v_kp = _clamp(_vel(p.v_kp, p.b_kp, p.kp, gb.kp, r1[0], r2[0]), V_MIN[0], V_MAX[0])
        v_kd = _clamp(_vel(p.v_kd, p.b_kd, p.kd, gb.kd, r1[1], r2[1]), V_MIN[1], V_MAX[1])
        v_wo = _clamp(_vel(p.v_omega_o, p.b_omega_o, p.omega_o, gb.omega_o, r1[2], r2[2]), V_MIN[2], V_MAX[2])

        p.v_kp, p.v_kd, p.v_omega_o = v_kp, v_kd, v_wo
        p.kp = _clamp(gb.kp + v_kp, *KP_RANGE)
        p.kd = _clamp(gb.kd + v_kd, *KD_RANGE)
        p.omega_o = _clamp(gb.omega_o + v_wo, *OMEGA_RANGE)

    def optimize(self, w_mae=1.0, w_vol=1.0, w_acc=1.0, w_zc=1.0) -> Optional[Particle]:
        self.w_mae, self.w_vol, self.w_acc, self.w_zc = w_mae, w_vol, w_acc, w_zc
        self.report.weights = {'mae': w_mae, 'volatility': w_vol, 'acceleration': w_acc, 'zero_crossings': w_zc}
        particles = [self._create_particle() for _ in range(self.NUM_PARTICLES)]

        print(f"\n{'=' * 50}\nNPSO OPTIMIZATION - SS-PD GIMBAL TUNING\nParticles: {self.NUM_PARTICLES}, Total rounds: {self.N_IT}\n"
              f"Weights: wMAE={w_mae}, wVol={w_vol}, wAcc={w_acc}, wZC={w_zc}\nSeed: {self.seed}, Scenario: {self.scenario}\n{'=' * 50}")

        try:
            for round_idx in range(self.N_IT):
                if round_idx == 0:
                    print(f"\n{'#' * 50}\n   INITIAL ROUND (0/{self.N_IT - 1})\n{'#' * 50}")
                else:
                    w, c1, c2 = _pso_coefficients(round_idx - 1, self.N_IT)
                    print(f"\n{'#' * 50}\n   ROUND {round_idx}/{self.N_IT - 1}\n{'#' * 50}\n  PSO params: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
                    round_gbest = self.best_global_particle

                for i, particle in enumerate(particles):
                    particle.iteration = round_idx
                    particle.particle_id = i + 1
                    if round_idx > 0:
                        self._update_particle(particle, w, c1, c2, round_gbest)

                    particle.raw = self._raw_metrics(self._collect_real_data(particle, i + 1, self.NUM_PARTICLES, round_idx))
                    if round_idx > 0:
                        self._score(particle, round_idx)

                if round_idx == 0:  # initial round = preliminary evaluations that fix the constants
                    self._compute_normalizers(particles)
                    for particle in particles:
                        self._score(particle, round_idx)

                round_best = min(particles, key=lambda p: p.fitness)
                self.report.add_iteration_best(round_best)
                self.best_history.append((round_idx, self.best_global_fitness))
                print(f"\n{'=' * 50}\n   Round {round_idx} summary:\n       Round best:  Fitness = {round_best.fitness:.5f}\n"
                      f"       Global best: Fitness = {self.best_global_fitness:.5f}\n{'=' * 50}")
        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            self.report.interrupted = True
        return self.best_global_particle

    def save_convergence_plot(self, output_path: str):
        if not self.best_history: return
        iterations = [h[0] for h in self.best_history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, [h[1] for h in self.best_history], 'o-', linewidth=2, color='#1f77b4', label='Global best')
        if len(self.report.best_per_iteration) == len(iterations):
            ax.plot(iterations, [p.fitness for p in self.report.best_per_iteration], 's--', alpha=0.7, color='#ff7f0e', label='Round best')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Fitness value'); ax.grid(True); ax.legend()
        w = self.report.weights
        ax.set_title(f"NPSO convergence (MAE={w['mae']}, vol={w['volatility']}, acc={w['acceleration']}, zc={w['zero_crossings']})")
        fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
        print(f"📈 Saved convergence plot: {output_path}")

# ─── Batch Runner Logic ────────────────────────────────────────────────────
def report_path(exp_id: int, seed: int) -> str:
    return os.path.join(REPORT_DIR, f"opt_exp{exp_id}_seed{seed}.json")

def is_done(exp_id: int, seed: int) -> bool:
    path = report_path(exp_id, seed)
    if not os.path.exists(path): return False
    try:
        with open(path) as f:
            r = json.load(f)
        return r.get('BEST_PARTICLE') is not None and not r['METADATA'].get('interrupted', False)
    except Exception:
        return False

def run_experiment(exp_id: int, seed: int, node: OptimizerNode) -> bool:
    """Runs one experiment, saves its report and plot. Returns True if interrupted."""
    cfg = EXPERIMENTS[exp_id]
    print(f"\n{'=' * 70}\nEXPERIMENT {exp_id}: {cfg['name']}\n  weights = {cfg['w']}\n"
          f"  seed = {seed}, particles = {NUM_PARTICLES}, iterations = {NUM_ITERATIONS}\n{'=' * 70}\n")

    optimizer = NPSO_SSPD_Optimizer(node, NUM_PARTICLES, NUM_ITERATIONS, seed, f"scenario_1_in_sample_exp{exp_id}")
    try:
        optimizer.optimize(*cfg['w'])
    finally:
        report_dict = optimizer.report.to_dict()
        report_dict['METADATA'].update(experiment_id=exp_id, experiment_name=cfg['name'], weights_order=['mae', 'vol', 'acc', 'zc'])

        os.makedirs(REPORT_DIR, exist_ok=True)
        path = report_path(exp_id, seed)
        with open(path, 'w') as f: json.dump(report_dict, f, indent=2)
        print(f"\n📊 Saved report: {path}")
        optimizer.save_convergence_plot(os.path.splitext(path)[0] + '_convergence.png')

    return optimizer.report.interrupted

def start_distance_node() -> subprocess.Popen:
    print(f"{'=' * 54}\n🚀 Launching distance_node (YOLO) in the background...\n{'=' * 54}")
    log = open(os.path.join(WORKSPACE_DIR, 'distance_node.log'), 'w')
    proc = subprocess.Popen(['ros2', 'run', 'sspid2', 'distance_node', '--ros-args', '-p', 'use_sim_time:=true'],
                            stdout=log, stderr=subprocess.STDOUT)
    time.sleep(3)  # let YOLO initialise
    return proc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help="Random seed for NPSO")
    parser.add_argument('--experiments', type=int, nargs='+', default=sorted(EXPERIMENTS.keys()), help="Experiments to run (e.g. 2 5 6)")
    parser.add_argument('--force', action='store_true', help="Re-run experiments that already have a report")
    parser.add_argument('--yes', action='store_true', help="Skip the confirmation prompt")
    parser.add_argument('--particles', type=int, default=None, help="Number of particles")
    parser.add_argument('--iterations', type=int, default=None, help="Number of update iterations")
    parser.add_argument('--secs', type=float, default=None, help="Evaluation duration in seconds per particle")
    parser.add_argument('--default', action='store_true', help="Use 20 particles, 10 iterations, 50 secs")
    args = parser.parse_args()

    global NUM_PARTICLES, NUM_ITERATIONS, EVAL_DURATION_S, EVAL_MAX_SAMPLES, PARTICLE_DURATION
    if args.default:
        NUM_PARTICLES, NUM_ITERATIONS, EVAL_DURATION_S = 20, 10, 50.0
    else:
        if args.particles is not None: NUM_PARTICLES = args.particles
        if args.iterations is not None: NUM_ITERATIONS = args.iterations
        if args.secs is not None: EVAL_DURATION_S = args.secs

    EVAL_MAX_SAMPLES = int(EVAL_DURATION_S / Ts)  # must follow EVAL_DURATION_S
    PARTICLE_DURATION = SETTLE_BEFORE_S + EVAL_DURATION_S + CLEANUP_AFTER_S

    todo = [e for e in args.experiments if args.force or not is_done(e, args.seed)]
    if not todo:
        print("All requested experiments already have reports. Nothing to do.")
        return

    total_h = (NUM_PARTICLES * (1 + NUM_ITERATIONS) * PARTICLE_DURATION / 3600.0) * len(todo)
    print(f"\n{'=' * 70}\nBATCH NPSO — seed {args.seed}\nExperiments: {todo}\nEstimated time: {total_h:.2f} h\nReports: {REPORT_DIR}\n{'=' * 70}")

    if not args.yes and input("\nStart batch? [y/N]: ").strip().lower() != 'y':
        print("Operation cancelled."); return

    distance_proc = start_distance_node()
    try:
        rclpy.init()
        node = OptimizerNode()
        try:
            for i, exp_id in enumerate(todo, 1):
                print(f"\n>>> [{i}/{len(todo)}] Starting experiment {exp_id} <<<\n")
                if run_experiment(exp_id, args.seed, node):
                    break
        finally:
            node.destroy_node()
            rclpy.shutdown()
    finally:
        print("\n🛑 Stopping distance_node...")
        distance_proc.terminate()
        try: distance_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired: distance_proc.kill()

if __name__ == '__main__':
    main()
