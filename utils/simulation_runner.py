"""
Parameter sweep runner for mFHN solver.
Generates config files, runs simulations, and cleans up.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()
EXECUTABLE = BASE_DIR / "build" / "mfhn_solver.exe"
CONFIG_TEMPLATE_PATH = BASE_DIR / "config_template.json"
RESULTS_BASE = BASE_DIR / "results"

# Initial conditions file (relative to BASE_DIR)
# INPUT_FILE = BASE_DIR / "utils" / "initial_conditions_1024_0.1.h5"

# Simulation dimension
DIMENSION = 1
STEPS = 1_000_000
SAVE_STEPS = 1000
DT, DX, NX = 0.001, 0.1, 1024
# Parameter ranges for sweep
EPS3_RANGE = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48]
A_RANGE = [
    1.50, 1.75, 
    2.00, 2.25, 2.50, 2.75, 
    3.00, 3.25, 3.50, 3.75, 
    4.00, 4.25, 4.50, 4.75, 
    5.00, 5.25, 5.50, 5.75, 
    6.00, 6.25, 6.50,
    ]

# ============================================================================
# Helper Functions
# ============================================================================


def load_config_template():
    """Load the base configuration template."""
    if not CONFIG_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Config template not found: {CONFIG_TEMPLATE_PATH}")
    with open(CONFIG_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_config(
    template,
    a_value,
    eps3_value,
    dim=DIMENSION,
    steps=STEPS,
    num_snapshots=SAVE_STEPS,
    dx=DX,
    dt=DT,
    Nx=NX,
):
    """Generate a config dict with specified parameters."""
    config = template.copy()
    config["simulation"]["N"] = Nx
    config["simulation"]["dim"] = dim
    config["simulation"]["dx"] = dx
    config["simulation"]["dt"] = dt
    config["simulation"]["steps"] = steps
    config["simulation"]["num_snapshots"] = num_snapshots
    config["model"]["a"] = a_value
    config["model"]["eps3"] = eps3_value

    # Set initial conditions file if it exists
    # if INPUT_FILE.exists():
    #     config["simulation"]["init_file"] = str(INPUT_FILE)

    return config


def save_config(config, filepath):
    """Save config to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def run_simulation(config_path):
    """
    Run the solver with the given config file.

    Returns:
        tuple: (success: bool, elapsed_time: float or None)
    """
    try:
        result = subprocess.run(
            [str(EXECUTABLE), str(config_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per simulation
        )

        if result.returncode != 0:
            print(f"   ERROR: Exit code {result.returncode}")
            if result.stderr:
                print(f"   stderr: {result.stderr[:200]}")
            return False, None

        # Parse elapsed time from stdout if available
        elapsed = None
        for line in result.stdout.split("\n"):
            if "Total time:" in line:
                try:
                    elapsed = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        return True, elapsed

    except subprocess.TimeoutExpired:
        print("   ERROR: Timeout (exceeded 1 hour)")
        return False, None
    except FileNotFoundError:
        print(f"   ERROR: Executable not found: {EXECUTABLE}")
        return False, None


def format_runtime(seconds):
    """Format seconds as human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.2f} h"


# ============================================================================
# Main Sweep Logic
# ============================================================================
def run_parameter_sweep():
    """Run parameter sweep over a and eps3."""

    # Validate executable
    if not EXECUTABLE.exists():
        print(f"Error: Executable not found: {EXECUTABLE}")
        print("Please build the project first:")
        print("  mkdir build && cd build")
        print("  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..")
        print("  ninja")
        return

    # Validate input file
    # if not INPUT_FILE.exists():
    #     print(f"Warning: Initial conditions file not found: {INPUT_FILE}")
    #     print("Solver will use default initial conditions - config.json.")

    # Load template
    print(f"Loading config template: {CONFIG_TEMPLATE_PATH}")
    template = load_config_template()

    # Statistics
    total_runs = len(A_RANGE) * len(EPS3_RANGE)
    successful_runs = 0
    failed_runs = 0
    start_time = datetime.now()

    print(f"\n{'=' * 60}")
    print(f"Parameter Sweep: {total_runs} simulations in {DIMENSION}D")
    print(f"  a: {A_RANGE[0]} to {A_RANGE[-1]}")
    print(f"  eps3: {EPS3_RANGE[0]} to {EPS3_RANGE[-1]}")
    print(f"{'=' * 60}\n")

    run_index = 0

    for a_value in A_RANGE:
        for eps3_value in EPS3_RANGE:
            run_index += 1

            # Generate config filename
            config_filename = f"config_a{a_value:.3f}_eps3{eps3_value:.3f}.json"

            # Generate and save config
            config = generate_config(template, a_value, eps3_value)
            save_config(config, config_filename)

            # Print progress
            print(f"[{run_index}/{total_runs}] a={a_value:.3f}, eps3={eps3_value:.3f}")

            # Run simulation
            success, elapsed = run_simulation(config_filename)

            if success:
                successful_runs += 1
                status = "OK"
                if elapsed:
                    print(f"   -> Completed in {format_runtime(elapsed)}")
            else:
                failed_runs += 1
                status = "FAILED"

            # Clean up config file
            try:
                Path(config_filename).unlink()
            except OSError:
                pass  # Ignore cleanup errors

    # Summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print("Sweep Complete")
    print(f"{'=' * 60}")
    print(f"  Total runs:     {total_runs}")
    print(f"  Successful:     {successful_runs} ({100 * successful_runs / total_runs:.1f}%)")
    print(f"  Failed:         {failed_runs} ({100 * failed_runs / total_runs:.1f}%)")
    print(f"  Total time:     {format_runtime(total_time)}")
    print(f"  Average per run:{format_runtime(total_time / successful_runs) if successful_runs > 0 else 'N/A'}" )
    print(f"{'=' * 60}\n")


# ============================================================================
if __name__ == "__main__":
    run_parameter_sweep()
