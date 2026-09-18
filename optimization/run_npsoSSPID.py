####################
# run_npsoSSPID.py #
####################

# use <<ros2 topic echo /control --field data >> to see these logs

from npsoSSPID import NPSO_SSPIDOptimizer, PARTICLE_DURATION
import json, os
from datetime import datetime

def save_report(optimizer, interrupted=False):
    report_dict = optimizer.report.to_dict()
    timestamp   = datetime.now().strftime("%H:%M_%d.%m")
    report_dir = "/home/hawkeye/ros_hawkeye/robotics_project/ws_new2/extra/npso_sspid2"
    os.makedirs(report_dir, exist_ok=True) # Crear el directorio si no existe
    filename    = os.path.join(report_dir, f"opt_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n📊 Report saved to: {filename}")
    return filename

def print_summary(optimizer):
    sorted_particles = optimizer.report.get_sorted_particles()

    print(f"\n{'='*50}")
    print("📊 SS-PID OPTIMIZATION SUMMARY")
    print(f"{'='*50}")

    if sorted_particles:
        best = sorted_particles[0]
        print(f"\n🏆 BEST SOLUTION (Fitness: {best.fitness:.5f} | Iteration: {best.iteration})")
        print(f"  kp={best.kp:.6f}")
        print(f"  ki={best.ki:.9f}")
        print(f"  kd={best.kd:.6f}")
        print(f"  omega_o={best.ωo:.4f}  rad/s  ")
    print(f"{'='*50}\n")

def main():
    print("\n" + "="*50)
    print("State-Space-PID Optimization ")
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
            num_particles  = int(input("Number of particles: ") or "50")
            num_iterations = int(input("Number of iterations: ") or "20")
        except:
            num_particles  = 20
            num_iterations = 50

    total_tests     = num_particles * (1 + num_iterations)
    estimated_hours = (total_tests * PARTICLE_DURATION) / 3600

    print(f"\n{'='*50}")
    print(f"Configuration:")
    print(f"  Particles:      {num_particles}")
    print(f"  Iterations:     {num_iterations}")
    print(f"  Total tests:    {total_tests}")
    print(f"  Estimated time: {estimated_hours:.2f} hours")
    print(f"{'='*50}")

    print("\n⚠️  Start optimisation? [y/n]: ", end='')
    if input().strip().lower() != 'y':
        print("Operation cancelled.")
        return

    optimizer = NPSO_SSPIDOptimizer(
        num_particles=num_particles,
        num_iterations=num_iterations,
    )

    try:
        best = optimizer.optimize()

        print_summary(optimizer)
        save_report(optimizer, interrupted=False)

        print("\n" + "="*50)
        print("🎉 OPTIMISATION COMPLETED!")
        print("="*50)
        print(f"\n🏆 BEST RESULT: {best.fitness:.5f} fitness")
        print(f"  kp      = {best.kp:.6f}")
        print(f"  ki      = {best.ki:.9f}")
        print(f"  kd      = {best.kd:.6f}")
        print(f"  omega_o = {best.ωo:.4f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimisation interrupted by user!")

        print_summary(optimizer)
        save_report(optimizer, interrupted=True)

        best = optimizer.best_global_particle
        if best is not None:
            print("\n📊 Best parameters found so far:")
            print(f"  Combined Fitness: {best.fitness:.5f}")
            print(f"\nBEST PARAMETERS:")
            print(f"  kp      = {best.kp:.6f}")
            print(f"  ki      = {best.ki:.9f}")
            print(f"  kd      = {best.kd:.6f}")
            print(f"  omega_o = {best.ωo:.4f}")


    except Exception as e:
        print(f"\n❌ Error during optimisation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        optimizer.cleanup()

if __name__ == '__main__':
    main()