#!/usr/bin/env python3
"""
Hawkeye Gimbal evaluator.

Usage:
  python3 hawkeye_evaluator.py                # all metrics
  python3 hawkeye_evaluator.py --scenario:=1  # MAE, jerk, volatility, zero-crossings/sec
  python3 hawkeye_evaluator.py --scenario:=2  # ITAE, settling time, overshoot
  --experiment:=x  # Experiment ID

The /pid_log message must contain at least:
  data[0] = pixel error x [px]
  data[1] = pixel error y [px]
"""

import argparse, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float64MultiArray

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent / "data" / "results")
SINUSOIDAL_DURATION_SEC = 100.0
STEP_RESPONSE_DURATION_SEC = 5.0 # Initial error = (200, 150) px
SETTLING_TOLERANCE_PX = 5.0
SETTLING_HOLD_SEC = 2.0
PLOT_UPDATE_SEC = 0.001

SINUSOIDAL_METRICS = ["mae", "jerk", "volatility", "zero_crossings", "summary"]
STEP_METRICS = ["itae", "settling_time", "overshoot"]
ALL_METRICS = SINUSOIDAL_METRICS + STEP_METRICS

def normalize_ros_style_cli(argv):
    """Allow ROS-like user syntax: --scenario:=1, --experiment:=1, --show:=false."""
    fixed = []
    for arg in argv:
        if arg.startswith("--scenario:="):
            fixed.append("--scenario=" + arg.split(":=", 1)[1])
        elif arg.startswith("--experiment:="):
            fixed.append("--experiment=" + arg.split(":=", 1)[1])
        elif arg.startswith("--show:="):
            fixed.append("--show=" + arg.split(":=", 1)[1])
        else:
            fixed.append(arg)
    return fixed


def str2bool(value):
    """Parse true/false CLI values."""
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("true", "t", "1", "yes", "y"):
        return True

    if value in ("false", "f", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError("--show must be true or false")


def parse_args():
    parser = argparse.ArgumentParser(description="Hawkeye metric evaluator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sinusoidal", action="store_true", help="Only measure MAE, jerk, volatility, and zero-crossings")
    group.add_argument("--step-response", "--step_response", dest="step_response", action="store_true", help="Only measure ITAE, settling time, and overshoot")
    parser.add_argument("--duration", type=float, default=0.0, help="Override recording duration in seconds")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Folder for summary files and plot PNGs")
    parser.add_argument("--scenario", default="all", choices=["all", "1", "2"], help="Scenario ID: 1=sinusoidal, 2=step response, all=all metrics")
    parser.add_argument("--experiment", default="1", help="Experiment ID used in the output folder name")
    parser.add_argument("--show", type=str2bool, nargs="?", const=True, default=True, help="Show real-time plot windows: true or false. Default: true.")
    cli_args = normalize_ros_style_cli(remove_ros_args(args=sys.argv)[1:])
    return parser.parse_args(cli_args)


def zero_crossings(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.size < 2:
        return 0, np.zeros(0, dtype=int)
    events = (np.sign(signal[:-1]) != np.sign(signal[1:])).astype(int)
    return int(np.sum(events)), events


def cumulative_trapezoid(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def settling_time(radial, t, tolerance=SETTLING_TOLERANCE_PX, hold=SETTLING_HOLD_SEC):
    if radial.size < 2:
        return float("nan")
    inside = radial <= tolerance
    for start in np.where(inside)[0]:
        end = np.searchsorted(t, t[start] + hold, side="left")
        if end >= t.size:
            break
        if np.all(inside[start:end + 1]):
            return float(t[start])
    return float("nan")


def overshoot(error, t):
    if error.size < 2 or abs(error[0]) < 1e-12:
        return {"px": 0.0, "percent": 0.0, "peak_time_sec": float("nan")}
    initial_sign = np.sign(error[0])
    transitions = np.where(np.sign(error[:-1]) != np.sign(error[1:]))[0]
    if transitions.size == 0:
        return {"px": 0.0, "percent": 0.0, "peak_time_sec": float("nan")}
    start = int(transitions[0] + 1)
    opposite_side = -initial_sign * error[start:]
    peak_local = int(np.argmax(opposite_side))
    amplitude = max(0.0, float(opposite_side[peak_local]))
    peak_index = start + peak_local
    return {
        "px": amplitude,
        "percent": 100.0 * amplitude / abs(float(error[0])),
        "peak_time_sec": float(t[peak_index]),
    }


class HawkeyeEvaluator(Node):
    def __init__(self, args):
        super().__init__("hawkeye_evaluator")
        self.args = args
        self.experiment = args.experiment
        self.scenario = args.scenario
        self.metrics_to_use = self.choose_metrics(self.scenario)
        self.duration_sec = self.choose_duration(args.duration, self.scenario)
        self.output_dir = Path(DEFAULT_OUTPUT_ROOT).expanduser() / f"Exp_{args.experiment}_scenario_{self.scenario}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.t0 = None
        self.t, self.ex, self.ey = [], [], []
        self.finished = False
        self.figures = {}

        self.create_subscription(Float64MultiArray, "/pid_log", self.callback, 1000)
        self.create_timer(PLOT_UPDATE_SEC, self.update)
        self.setup_plots()
        self.get_logger().info(
            f"Started Hawkeye evaluator | metrics={self.metrics_to_use} | "
            f"duration={self.duration_sec:.1f}s | topic={"/pid_log"} | output={self.output_dir}"
        )

    @staticmethod
    def choose_metrics(scenario):
        if scenario == "1":
            return SINUSOIDAL_METRICS
        if scenario == "2":
            return STEP_METRICS
        return ALL_METRICS

    @staticmethod
    def choose_duration(duration, scenario): 
        if duration != 0.0:
            return float(duration)
        return STEP_RESPONSE_DURATION_SEC if scenario == "2" else SINUSOIDAL_DURATION_SEC

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def callback(self, msg):
        if len(msg.data) < 2 or self.finished:
            return
        if self.t0 is None:
            self.t0 = self.now_sec()
        self.t.append(self.now_sec() - self.t0)
        self.ex.append(float(msg.data[0]))
        self.ey.append(float(msg.data[1]))

    def arrays(self):
        t = np.asarray(self.t, dtype=float)
        ex = np.asarray(self.ex, dtype=float)
        ey = np.asarray(self.ey, dtype=float)
        return t, ex, ey, np.sqrt(ex ** 2 + ey ** 2)

    def compute_metrics(self):
        t, ex, ey, radial = self.arrays()
        diff = np.diff(radial)
        jerk_samples = np.abs(np.diff(diff))
        zc_x, zc_x_events = zero_crossings(ex)
        zc_y, zc_y_events = zero_crossings(ey)
        ox = overshoot(ex, t)
        oy = overshoot(ey, t)

        m = {
            "samples": int(t.size),
            "duration_sec": float(t[-1] - t[0]) if t.size > 1 else 0.0,
            "mae_px": float(np.mean(radial))*0.75 if radial.size else float("nan"),
            "mae_npso": min(1.0, float(np.mean(radial))*0.75 / 50.0) if radial.size else float("nan"),
            "volatility_std_delta_px_per_sample": float(np.std(diff)) if diff.size else float("nan"),
            "volatility_npso": min(1.0, float(np.std(diff)) / 20.0) if diff.size else float("nan"),
            "jerk_mean_abs_second_difference_px_per_sample2": float(np.mean(jerk_samples)) if jerk_samples.size else float("nan"),
            "jerk_npso": min(1.0, float(np.mean(jerk_samples)) / 5.0) if jerk_samples.size else float("nan"),
            "zero_crossings_x": zc_x,
            "zero_crossings_y": zc_y,
            "zero_crossings_total": zc_x + zc_y,
            "zero_crossings_npso": min(1.0, float(zc_x + zc_y) / (SINUSOIDAL_DURATION_SEC*2)),
            "itae_radial_px_s2": float(TRAPZ(t * radial, t)) if t.size > 1 else float("nan"),
            "settling_time_sec": settling_time(radial, t),
            "settling_tolerance_px": SETTLING_TOLERANCE_PX,
            "settling_hold_sec": SETTLING_HOLD_SEC,
            "overshoot_x_px": ox["px"],
            "overshoot_x_percent": ox["percent"],
            "overshoot_x_peak_time_sec": ox["peak_time_sec"],
            "overshoot_y_px": oy["px"],
            "overshoot_y_percent": oy["percent"],
            "overshoot_y_peak_time_sec": oy["peak_time_sec"],
            "overshoot_max_px": max(ox["px"], oy["px"]),
            "overshoot_max_percent": max(ox["percent"], oy["percent"]),
        }
        return {k: v for k, v in m.items() if self.keep_metric_key(k)}

    def keep_metric_key(self, key):
        always = key in {"samples", "duration_sec"}
        sinusoidal = (
            key.startswith("mae") or key.startswith("volatility") or
            key.startswith("jerk") or key.startswith("zero_crossings")
        )
        step = key.startswith("itae") or key.startswith("settling") or key.startswith("overshoot")
        return always or (sinusoidal and any(x in self.metrics_to_use for x in SINUSOIDAL_METRICS)) or (step and any(x in self.metrics_to_use for x in STEP_METRICS))

    def setup_plots(self):
        if self.args.show:
            plt.ion()
        else:
            plt.switch_backend("Agg")
            plt.ioff()

        for name in self.metrics_to_use:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            if self.args.show and fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(name)
            self.figures[name] = {"fig": fig, "ax": ax}

        if self.args.show:
            plt.show(block=False)

    def update_plot(self, name, t, ex, ey, radial, metrics):
        ax = self.figures[name]["ax"]
        ax.clear()
        ax.grid(True)

        if name == "mae":
            ax.plot(t, radial, label="Radial error")
            ax.axhline(metrics.get("mae_px", float("nan")), linestyle="--", linewidth=1, label="MAE")
            ax.set_title(f"MAE = {metrics.get("mae_px", float("nan")):.4f} px | NPSO = {metrics.get("mae_npso", float("nan")):.4f}")            
            ax.set_ylabel("Radial error [px]")
            ax.legend()
        elif name == "volatility":
            ax.plot(t[1:], np.diff(radial))
            ax.axhline(0.0, linewidth=1)
            ax.set_title(f"Volatility = {metrics.get('volatility_std_delta_px_per_sample', float('nan')):.4f} px/sample | NPSO = {metrics.get('volatility_npso', float('nan')):.4f}")
            ax.set_ylabel("Δ radial error [px/sample]")
        elif name == "jerk":
            diff = np.diff(radial)
            ax.plot(t[2:], np.abs(np.diff(diff)))
            ax.set_title(f"Jerk = {metrics.get('jerk_mean_abs_second_difference_px_per_sample2', float('nan')):.4f} px/sample² | NPSO = {metrics.get('jerk_npso', float('nan')):.4f}")
            ax.set_ylabel("|Δ² radial error| [px/sample²]")
        elif name == "zero_crossings":
            _, zcx = zero_crossings(ex)
            _, zcy = zero_crossings(ey)
            total = np.cumsum(zcx) + np.cumsum(zcy)
            ax.step(t[1:], total, where="post")
            ax.set_title(f"Zero-crossings = {metrics.get('zero_crossings_total', 0)} | NPSO = {metrics.get('zero_crossings_npso', float('nan')):.4f}")
            ax.set_ylabel("Cumulative count")
        elif name == "summary":
            labels = ["MAE", "Volatility", "Jerk", "Zero-crossings"]
            values = [
                metrics.get("mae_npso", float("nan")),
                metrics.get("volatility_npso", float("nan")),
                metrics.get("jerk_npso", float("nan")),
                metrics.get("zero_crossings_npso", float("nan")),
            ]
            bars = ax.bar(labels, values)
            ax.set_ylim(0.0, 1.1)
            ax.set_ylabel("Normalized NPSO value")
            ax.set_title(
                f"Scenario 1 normalized NPSO metrics | "
                f"Experiment {self.experiment}"
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, min(value + 0.03, 1.05), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        elif name == "itae":
            ax.plot(t, cumulative_trapezoid(t * radial, t))
            ax.set_title(f"ITAE = {metrics.get('itae_radial_px_s2', float('nan')):.4f} px·s²")
            ax.set_ylabel("Cumulative ITAE [px·s²]")
        elif name == "settling_time":
            ax.plot(t, radial)
            ax.axhline(SETTLING_TOLERANCE_PX, linestyle="--", linewidth=1)
            st = metrics.get("settling_time_sec", float("nan"))
            if np.isfinite(st):
                ax.axvline(st, linestyle="--", linewidth=1)
            ax.set_title(f"Settling time = {st:.4f}s" if np.isfinite(st) else "Settling time = not settled")
            ax.set_ylabel("Radial error [px]")
        elif name == "overshoot":
            ax.plot(t, ex, label="error x")
            ax.plot(t, ey, label="error y")
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.legend()
            ax.set_title(f"Overshoot max = {metrics.get('overshoot_max_px', float('nan')):.4f} px | {metrics.get('overshoot_max_percent', float('nan')):.2f}%")
            ax.set_ylabel("Signed error [px]")

        ax.set_xlabel("Time [s]")
        self.figures[name]["fig"].tight_layout()

    def update(self):
        if self.finished or self.t0 is None:
            return
        t, ex, ey, radial = self.arrays()
        if t.size < 2:
            return
        metrics = self.compute_metrics()
        if self.args.show:
            for name in self.metrics_to_use:
                self.update_plot(name, t, ex, ey, radial, metrics)
            plt.pause(0.001)
        if t[-1] >= self.duration_sec:
            self.finalize()
            rclpy.shutdown()

    @staticmethod
    def to_list(array):
        return np.asarray(array, dtype=float).tolist()

    def save_plot_report(self, metrics):
        """
        Save one compact JSON report with only the data needed
        to reproduce the generated plots directly.
        """
        t, ex, ey, radial = self.arrays()

        report = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "duration_sec": metrics.get("duration_sec"),
            "samples": metrics.get("samples"),
            "plots": {}
        }

        # -----------------------------
        # Scenario 1 plots
        # -----------------------------
        if "mae" in self.metrics_to_use:
            report["plots"]["mae"] = {
                "x_label": "Time [s]",
                "y_label": "Radial error [px]",
                "x": self.to_list(t),
                "y": self.to_list(radial),
                "mae_line_px": metrics.get("mae_px"),
                "final_value_px": metrics.get("mae_px"),
                "npso_value": metrics.get("mae_npso"),
            }

        if "volatility" in self.metrics_to_use:
            delta_radial = np.diff(radial)

            report["plots"]["volatility"] = {
                "x_label": "Time [s]",
                "y_label": "Delta radial error [px/sample]",
                "x": self.to_list(t[1:]),
                "y": self.to_list(delta_radial),
                "final_value_px_per_sample": metrics.get("volatility_std_delta_px_per_sample"),
                "npso_value": metrics.get("volatility_npso"),
            }

        if "jerk" in self.metrics_to_use:
            delta_radial = np.diff(radial)
            jerk_values = np.abs(np.diff(delta_radial))

            report["plots"]["jerk"] = {
                "x_label": "Time [s]",
                "y_label": "Absolute second difference [px/sample²]",
                "x": self.to_list(t[2:]),
                "y": self.to_list(jerk_values),
                "final_value_px_per_sample2": metrics.get("jerk_mean_abs_second_difference_px_per_sample2"),
                "npso_value": metrics.get("jerk_npso"),
            }

        if "zero_crossings" in self.metrics_to_use:
            _, zcx = zero_crossings(ex)
            _, zcy = zero_crossings(ey)
            cumulative_zc = np.cumsum(zcx) + np.cumsum(zcy)

            report["plots"]["zero_crossings"] = {
                "x_label": "Time [s]",
                "y_label": "Cumulative count",
                "x": self.to_list(t[1:]),
                "y": self.to_list(cumulative_zc),
                "zero_crossings_x": metrics.get("zero_crossings_x"),
                "zero_crossings_y": metrics.get("zero_crossings_y"),
                "zero_crossings_total": metrics.get("zero_crossings_total"),
                "npso_value": metrics.get("zero_crossings_npso"),
            }

        if "summary" in self.metrics_to_use:
            report["plots"]["summary"] = {
                "x_label": "Metric",
                "y_label": "Normalized NPSO value",
                "labels": ["MAE", "Volatility", "Jerk", "Zero-crossings"],
                "values": [
                    metrics.get("mae_npso"),
                    metrics.get("volatility_npso"),
                    metrics.get("jerk_npso"),
                    metrics.get("zero_crossings_npso"),
                ],
                "normalizers": {
                    "mae": 50.0,
                    "volatility": 20.0,
                    "jerk": 5.0,
                    "zero_crossings": 100.0,
                },
            }

        # -----------------------------
        # Scenario 2 plots
        # -----------------------------
        if "itae" in self.metrics_to_use:
            cumulative_itae = cumulative_trapezoid(t * radial, t)

            report["plots"]["itae"] = {
                "x_label": "Time [s]",
                "y_label": "Cumulative ITAE [px·s²]",
                "x": self.to_list(t),
                "y": self.to_list(cumulative_itae),
                "final_value_px_s2": metrics.get("itae_radial_px_s2"),
            }

        if "settling_time" in self.metrics_to_use:
            report["plots"]["settling_time"] = {
                "x_label": "Time [s]",
                "y_label": "Radial error [px]",
                "x": self.to_list(t),
                "y": self.to_list(radial),
                "settling_time_sec": metrics.get("settling_time_sec"),
                "settling_tolerance_px": SETTLING_TOLERANCE_PX,
                "settling_hold_sec": SETTLING_HOLD_SEC,
            }

        if "overshoot" in self.metrics_to_use:
            report["plots"]["overshoot"] = {
                "x_label": "Time [s]",
                "y_label": "Signed error [px]",
                "x": self.to_list(t),
                "error_x_px": self.to_list(ex),
                "error_y_px": self.to_list(ey),
                "overshoot_x_px": metrics.get("overshoot_x_px"),
                "overshoot_y_px": metrics.get("overshoot_y_px"),
                "overshoot_max_px": metrics.get("overshoot_max_px"),
                "overshoot_max_percent": metrics.get("overshoot_max_percent"),
            }

        with (self.output_dir / "plot_report.json").open("w") as f:
            json.dump(report, f, indent=2, allow_nan=True)

    def finalize(self):
        if self.finished:
            return

        self.finished = True
        metrics = self.compute_metrics()

        t, ex, ey, radial = self.arrays()

        if t.size >= 2:
            for name in self.metrics_to_use:
                self.update_plot(name, t, ex, ey, radial, metrics)

        self.save_plot_report(metrics)

        for name, item in self.figures.items():
            item["fig"].savefig(self.output_dir / f"{name}.png", dpi=180)

        self.get_logger().info("\n" + json.dumps(metrics, indent=2, allow_nan=True))
        self.get_logger().info(f"Saved results to: {self.output_dir}")


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = HawkeyeEvaluator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.finalize()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    plt.ioff()
    if args.show:
        plt.show(block=True)
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
