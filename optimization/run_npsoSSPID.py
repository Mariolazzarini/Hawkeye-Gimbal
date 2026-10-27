####################
# run_npsoSSPID.py #
####################

# use <<ros2 topic echo /control --field data >> to see these logs

from npsoSSPID import NPSO_SSPD_Optimizer, PARTICLE_DURATION
import json, os
from datetime import datetime

def save_report(optimizer, interrupted=False):
    report_dict = optimizer.report.to_dict()
    timestamp   = datetime.now().strftime("%H:%M_%d.%m")
    report_dir = "/home/hawkeye/ros_hawkeye/robotics_project/ws_new2/extra/npso_sspid2"
    os.makedirs(report_dir, exist_ok=True) # Create directory if it does not exist
    filename    = os.path.join(report_dir, f"opt_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n📊 Report saved to: {filename}")
    return filename

def print_summary(optimizer):
    sorted_particles = optimizer.report.get_sorted_particles()

    print(f"\n{'='*50}")
    print("📊 SS-PD OPTIMIZATION SUMMARY")
    print(f"{'='*50}")

    if sorted_particles:
        best = sorted_particles[0]
        print(f"\n🏆 BEST SOLUTION (Fitness: {best.fitness:.5f} | Iteration: {best.iteration})")
        print(f"  kp={best.kp:.6f}")
        print(f"  kd={best.kd:.6f}")
        print(f"  omega_o={best.omega_o:.4f}  rad/s  ")
    print(f"{'='*50}\n")

def main():
    print("\n" + "="*50)
    print("State-Space-PD Optimization ")
    print("="*50)
    print("\nOptimisation options:")
    print("  1. Fast       (10 particles,  4 iterations)")
    print("  2. Normal     (20 particles, 10 iterations)")
    print("  3. Extended   (30 particles, 15 iterations)")
    print("  4. Exhaustive (50 particles, 20 iterations)")
    print("  5. Custom")

    try:
        option = int(input("\nSelect option [1-5]: ") or "5")
    except:
        option = 5

    if option == 1:
        num_particles  = 10
        num_iterations = 4
    elif option == 2:
        num_particles  = 20
        num_iterations = 10
    elif option == 3:
        num_particles  = 30
        num_iterations = 15
    elif option == 4:
        num_particles  = 50
        num_iterations = 20
    else:
        try:
            num_particles  = int(input("Number of particles: ") or "20")
            num_iterations = int(input("Number of iterations: ") or "10")
        except:
            num_particles  = 20
            num_iterations = 10

    # Empty -> fall back to npsoSSPID.DEFAULT_SEED (unseeded if that is None).
    try:
        seed_str = input("\nRandom seed (int, empty = default): ").strip()
        seed = int(seed_str) if seed_str else None
    except ValueError:
        seed = None

    total_tests     = num_particles * (1 + num_iterations)
    estimated_hours = (total_tests * PARTICLE_DURATION) / 3600

    print(f"\n{'='*50}")
    print(f"Configuration:")
    print(f"  Particles:      {num_particles}")
    print(f"  Iterations:     {num_iterations}")
    print(f"  Total tests:    {total_tests}")
    print(f"  Seed:           {seed if seed is not None else 'default (may be unseeded)'}")
    print(f"  Estimated time: {estimated_hours:.2f} hours")
    print(f"{'='*50}")

    print("\n⚠️  Start optimisation? [y/n]: ", end='')
    if input().strip().lower() != 'y':
        print("Operation cancelled.")
        return

    optimizer = NPSO_SSPD_Optimizer(
        num_particles=num_particles,
        num_iterations=num_iterations,
        seed=seed,
    )

    try:
        best = optimizer.optimize()
        interrupted = optimizer.report.interrupted

        print_summary(optimizer)
        save_report(optimizer, interrupted=interrupted)

        print("\n" + "="*50)
        if interrupted:
            print("⚠️  OPTIMISATION INTERRUPTED BY USER")
        else:
            print("🎉 OPTIMISATION COMPLETED!")
        print("="*50)

        if best is not None:
            label = "Best parameters found so far" if interrupted else "BEST RESULT"
            print(f"\n🏆 {label}: {best.fitness:.5f} fitness")
            print(f"  kp      = {best.kp:.6f}")
            print(f"  kd      = {best.kd:.6f}")
            print(f"  omega_o = {best.omega_o:.4f}")
        else:
            print("\nNo evaluation was completed, so there is no result to report.")

    except Exception as e:
        print(f"\n❌ Error during optimisation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        optimizer.cleanup()

if __name__ == '__main__':
    main()
