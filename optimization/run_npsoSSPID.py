####################
# run_npsoSSPID.py #
####################

# Entry point for the offline NPSO tuning of the SS-PD gimbal controller.
# use <<ros2 topic echo /control --field data >> to see the controller logs

import json
from datetime import datetime
from pathlib import Path

from npsoSSPID import NPSO_SSPD_Optimizer, PARTICLE_DURATION

# Results live inside the repository, next to everything else the README
# documents (the previous version wrote to a hard-coded absolute path under
# /home/hawkeye/... which only existed on one machine).
REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# Fitness weights (Eq. 27) in the order (w_mae, w_vol, w_acc, w_zc).
# These are Table 3 of the paper.
EXPERIMENTS = {
    "2": ("MAE",                       (1.0, 0.0, 0.0, 0.0)),
    "3": ("MAE + Volatility",          (1.0, 1.0, 0.0, 0.0)),
    "4": ("MAE + Zero-crossings",      (1.0, 0.0, 0.0, 1.0)),
    "5": ("MAE + Acceleration",        (1.0, 0.0, 1.0, 0.0)),
    "6": ("Combined (equal weights)",  (1.0, 1.0, 1.0, 1.0)),
    "7": ("Combined (reweighted)",     (1.0, 0.5, 0.5, 1.0)),
}


def save_report(optimizer, interrupted=False):
    optimizer.report.interrupted = bool(interrupted)
    report_dict = optimizer.report.to_dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = REPORT_DIR / f"opt_{timestamp}.json"

    with filename.open("w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReport saved to: {filename}")
    return filename


def print_particle(p, prefix="  "):
    print(f"{prefix}kp      = {p.kp:.6f}")
    print(f"{prefix}kd      = {p.kd:.6f}")
    print(f"{prefix}omega_o = {p.omega_o:.4f}  rad/s")


def print_summary(optimizer):
    sorted_particles = optimizer.report.get_sorted_particles()

    print(f"\n{'='*50}")
    print("SS-PD OPTIMIZATION SUMMARY")
    print(f"{'='*50}")

    if sorted_particles:
        best = sorted_particles[0]
        print(f"\nBEST SOLUTION (Fitness: {best.fitness:.5f} | Round: {best.iteration})")
        print_particle(best)
    else:
        print("\nNo particle was evaluated.")
    print(f"{'='*50}\n")


def ask_int(prompt, default):
    try:
        return int(input(prompt) or str(default))
    except (ValueError, EOFError):
        return default


def main():
    print("\n" + "=" * 50)
    print("State-Space PD (SS-PD) NPSO Optimization")
    print("=" * 50)
    print("\nSwarm size options:")
    print("  1. Fast       (10 particles,  4 update rounds)")
    print("  2. Normal     (20 particles, 10 update rounds)  <- paper setup")
    print("  3. Extended   (30 particles, 15 update rounds)")
    print("  4. Exhaustive (50 particles, 20 update rounds)")
    print("  5. Custom")

    option = ask_int("\nSelect option [1-5]: ", 2)

    presets = {1: (10, 4), 2: (20, 10), 3: (30, 15), 4: (50, 20)}
    if option in presets:
        num_particles, num_iterations = presets[option]
    else:
        num_particles = ask_int("Number of particles: ", 20)
        num_iterations = ask_int("Number of update rounds: ", 10)

    print("\nFitness weight presets (w_mae, w_vol, w_acc, w_zc):")
    for key, (label, weights) in EXPERIMENTS.items():
        print(f"  {key}. Experiment {key} - {label:<38} {weights}")
    exp = (input("\nSelect experiment [2-7, default 6]: ") or "6").strip()
    if exp not in EXPERIMENTS:
        print(f"Unknown experiment '{exp}', falling back to 6.")
        exp = "6"
    exp_label, weights = EXPERIMENTS[exp]

    total_tests = num_particles * (1 + num_iterations)
    estimated_hours = (total_tests * PARTICLE_DURATION) / 3600

    print(f"\n{'='*50}")
    print("Configuration:")
    print(f"  Experiment:     {exp} ({exp_label})")
    print(f"  Weights:        {weights}")
    print(f"  Particles:      {num_particles}")
    print(f"  Update rounds:  {num_iterations}")
    print(f"  Total tests:    {total_tests}")
    print(f"  Estimated time: {estimated_hours:.2f} hours")
    print(f"  Results dir:    {REPORT_DIR}")
    print(f"{'='*50}")

    print("\nStart optimisation? [y/n]: ", end='')
    try:
        confirm = input().strip().lower()
    except EOFError:
        confirm = 'n'
    if confirm != 'y':
        print("Operation cancelled.")
        return

    optimizer = NPSO_SSPD_Optimizer(
        num_particles=num_particles,
        num_iterations=num_iterations,
    )

    interrupted = False
    try:
        w_mae, w_vol, w_acc, w_zc = weights
        best = optimizer.optimize(w_mae=w_mae, w_vol=w_vol, w_acc=w_acc, w_zc=w_zc)

        print_summary(optimizer)
        save_report(optimizer, interrupted=False)

        print("\n" + "=" * 50)
        print("OPTIMISATION COMPLETED!")
        print("=" * 50)
        if best is not None:
            print(f"\nBEST RESULT: {best.fitness:.5f} fitness")
            print_particle(best)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nOptimisation interrupted by user!")

        print_summary(optimizer)
        save_report(optimizer, interrupted=True)

        best = optimizer.best_global_particle
        if best is not None:
            print("\nBest parameters found so far:")
            print(f"  Combined Fitness: {best.fitness:.5f}")
            print_particle(best)

    except Exception as e:
        print(f"\nError during optimisation: {e}")
        import traceback
        traceback.print_exc()
        save_report(optimizer, interrupted=True)
    finally:
        optimizer.cleanup()
        if interrupted:
            print("Partial results were saved before exiting.")


if __name__ == '__main__':
    main()
