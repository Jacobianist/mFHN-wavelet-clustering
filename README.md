# CUDA Simulation of 1D/2D Three-Component Reaction-Diffusion System (Extended FitzHugh-Nagumo)

A high-performance CUDA implementation of an explicit solver for the modified FitzHugh-Nagumo (mFHN) equation system. The solver supports both 1D and 2D simulations with customizable parameters.

## Mathematical Model

A system of three coupled partial differential equations is simulated:

$$
\frac{\partial u}{\partial t} = D_1 \nabla^2 u + \phi (a u - \alpha u^3 - b v - c w)
$$

$$
\frac{\partial v}{\partial t} = D_2 \nabla^2 v + \phi \varepsilon_2 (u - v)
$$

$$
\frac{\partial w}{\partial t} = D_3 \nabla^2 w + \phi \varepsilon_3 (u - w)
$$

where:

- **u** — activator (primary variable, e.g., membrane potential)
- **v, w** — inhibitors (recovery variables)
- **D₁, D₂, D₃** — diffusion coefficients
- **φ, a, b, c, α, ε₂, ε₃** — kinetic parameters of the system

## Numerical Method

The solver uses an explicit time-stepping scheme:

- **Reaction terms**: 4th-order Runge-Kutta (RK4)
- **Diffusion terms**: Central finite differences (2nd order)
  - 1D: 3-point stencil
  - 2D: 5-point stencil
- **Boundary conditions**: Neumann (zero-flux) on all boundaries
- **Stability**: CFL condition checked with 10% safety margin

## Project Structure

```
mFHN-wavelet-clustering/
├── CMakeLists.txt              # CMake build configuration
├── config_template.json        # Runtime configuration parameters
├── README.md                   # This documentation file
├── .gitignore                  # Git ignore rules
├── src/                        # Source code directory
│   ├── main.cu                 # Main entry point and simulation loop
│   ├── solver_explicit.cu      # CUDA kernel implementations (1D/2D)
│   ├── solver_explicit.cuh     # CUDA kernel declarations
│   ├── params.h                # Parameter structures (ModelParams, SimParams)
│   ├── config.h / config.cpp   # JSON configuration loader
│   ├── h5_exporter.h / h5_exporter.cpp  # HDF5 output writer
│   ├── cuda_utils.h            # CUDA error checking and RAII wrappers
│   └── logger.h                # File/console logging utility
└── utils/                      # Python utilities
    ├── initial_conditions_*.h5 # Example initial conditions (HDF5 format)
    ├── megaplot.py             # Visualization script for 252 snapshots
    ├── simulation_runner.py    # Automation script for computational experiments
    ├── initial_field_generator.py # tool for generating custom 2D initial conditions to HDF5 format
    ├── clustering_pipeline.py  # Comprehensive clustering framework and validation
    └── wavelets-on-interval/   # C++ source code (Wavelet Analysis)
        └─ fwt01.cpp / fwt01.h  # CDV wavelet core (1D DWT on finite interval)
 ```

### Source Files Description

| File | Description |
| :--- | :--- |
| `main.cu` | Program entry point; handles initialization, time-stepping loop, and I/O |
| `solver_explicit.cu` | CUDA kernels for 1D and 2D explicit integration |
| `solver_explicit.cuh` | Header with CUDA kernel declarations | 
| `params.h` | Data structures for model and simulation parameters |
| `config.h` / `config.cpp` | JSON configuration loading and validation |
| `h5_exporter.h` / `h5_exporter.cpp` | HDF5 file writer for simulation snapshots |
| `cuda_utils.h` | CUDA error checking macros and RAII memory wrappers |
| `logger.h` | Simple logging utility for console and file output |
| `utils/megaplot.py` | Advanced visualization tool for generating publication-ready supplementary figures (1D/2D 252 snapshots). |
| `utils/simulation_runner.py` | Automation script for computational experiments; handles parameter sweeping and batch execution of 1D/2D simulations. |
| `utils/initial_field_generator.py` | Script for generating, previewing, and exporting custom 2D initial conditions to HDF5 format. |
| `utils/clustering_pipeline.py` | Comprehensive clustering framework featuring bootstrap stability analysis and statistical validation |

## Configuration

Edit `config_template.json` to set simulation parameters:

```json
{
  "simulation": {
    "N": 1024, // Grid size (N for 1D, N×N for 2D)
    "dim": 1, // Spatial dimension (1 or 2)
    "dx": 0.1, // Spatial step size
    "dt": 0.001, // Time step size
    "steps": 100000, // Total number of time steps
    "num_snapshots": 100, // Number of output snapshots
    "init_file": "./path/to/initial.h5" // Initial conditions file
  },
  "model": {
    "a": 3.5, // Reaction rate coefficient
    "b": 3.0, // Coupling coefficient for v
    "c": 3.5, // Coupling coefficient for w
    "alpha": 1.5, // Cubic nonlinearity coefficient
    "phi": 0.5, // Kinetics scaling factor
    "eps2": 1.0, // Time scale for inhibitor v
    "eps3": 0.5, // Time scale for inhibitor w
    "D1": 0.0, // Diffusion coefficient for u
    "D2": 0.0, // Diffusion coefficient for v
    "D3": 0.5 // Diffusion coefficient for w
  }
}
```

## Building

### Prerequisites

- CUDA Toolkit (10.0+)
- CMake (3.18+)
- HDF5 library with C++ bindings
- nlohmann/json library
- ZLIB

```bash
# Build
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```


## Running

```bash
./mfhn_solver [config_file]
```

**Examples:**

```bash
./mfhn_solver                    # Uses config.json (default)
./mfhn_solver my_config.json     # Uses specified config file
```

The solver will:

1. Load configuration from `config.json` or specified file
2. Initialize or load initial conditions
3. Run the simulation on GPU
4. Save results to `results/YYYY-MM-DD/HHMMSS_dimN_N.../`

### Output Files

- `result.h5` — HDF5 file containing time-series data for u, v, w
- `simulation.log` — Log file with progress and timing information
- `config.json` — Copy of the configuration used

### HDF5 Output Structure

```
result.h5
├── /u              # Dataset: (num_snapshots, N) for 1D or (num_snapshots, N, N) for 2D
├── /v              # Dataset: same shape as u
├── /w              # Dataset: same shape as u
└── @attributes     # Simulation metadata (parameters, timestamp, etc.)
```

### Configuring Output Components

By default, only the **u** component is saved to reduce output file size. To save additional components (**v**, **w**):

1. Edit `src/h5_exporter.h` and modify the configuration macros:

   ```cpp
   #define SAVE_U_COMPONENT 1  // Always 1 (required)
   #define SAVE_V_COMPONENT 1  // Set to 1 to save v component
   #define SAVE_W_COMPONENT 1  // Set to 1 to save w component
   ```

2. Recompile the project:

   ```bash
   cd build && ninja
   ```

3. Run the simulation — the HDF5 output will include only the enabled components

This allows you to selectively save components based on your analysis needs without modifying the solver logic.

### HDF5 Metadata Attributes

The output file includes simulation metadata as attributes for reproducibility:

- `creation_timestamp` — Date and time of simulation
- `dimension` — Spatial dimension (1 or 2)
- `grid_size_N` — Grid size per dimension
- `dx`, `dt` — Spatial and temporal step sizes
- `total_steps` — Total number of time steps
- `num_snapshots` — Number of output snapshots
- Model parameters: `model_a`, `model_b`, `model_c`, `model_alpha`, `model_phi`, `model_eps2`, `model_eps3`
- Diffusion coefficients: `model_D1`, `model_D2`, `model_D3`
- Derived values: `r_u`, `r_v`, `r_w`, `is_stable`

### HDF5 Compression

Output data uses chunked storage with deflate compression (level 6) for efficient file sizes:

- **1D chunks**: `[1, N]` — one snapshot per chunk
- **2D chunks**: `[1, N, N]` — one snapshot per chunk

## Initial Conditions

### Default Generation

If no initial condition file is provided or loading fails, the solver generates default conditions:

- **1D**: Localized pulse in the center (u≈1.0, v≈0.0, w≈0.0)
- **2D**: Circular spot in the center (u≈1.0, v≈0.0, w≈0.0)

Small random noise is added to mimic perturbations.

### Custom Initial Conditions

Provide an HDF5 file with datasets named `u`, `v`, `w`. The file should contain:

- **1D simulation**: 2D dataset (will extract middle row) or 1D dataset
- **2D simulation**: 2D dataset of shape (N, N)

## Stability Considerations

The explicit scheme requires the CFL condition to be satisfied:

- **1D**: dt ≤ dx² / (2 × D_max)
- **2D**: dt ≤ dx² / (4 × D_max)

where D_max = max(D₁, D₂, D₃).

The solver automatically checks this condition and warns if the time step may be unstable.

## Features

- **CUDA error checking**: Automatic error detection for all CUDA operations
- **RAII memory management**: Automatic cleanup of GPU resources
- **HDF5 compression**: Chunked storage with deflate compression (level 6)
- **Simulation metadata**: All parameters stored in output file for reproducibility
- **Performance timing**: Per-step timing logged to simulation.log
- **Stability validation**: CFL condition checked before simulation starts

## Wavelet Stage: Orthogonal DWT on a Finite Interval

For a bounded spatial domain, classical DWT with periodic or symmetric extension injects
spurious energy at the boundaries, corrupting the wavelet energy spectrum near edges.
We therefore use the **CDV construction** (Cohen, Daubechies, Vial, 1993), which builds
an orthonormal wavelet basis adapted to the interval with prescribed boundary conditions.

### Implementation (`wavelets-on-interval/`)
The wavelet core is located in wavelets-on-interval/. It can be compiled as a standalone executable or integrated into the main CMake.

| File | Role |
|---|---|
| `fwt01.h` / `fwt01.cpp` | 1D CDV forward/inverse DWT kernels on a finite interval |
| `read_h5.cpp` | HDF5 reader: loads a snapshot of `u` (or `v`, `w`) |

**1D:** direct application of `fwt01` along the spatial axis.

**2D (separable):** the 1D transform is applied **row-wise, then column-wise**, producing
the four standard sub-bands `{LL, LH, HL, HH}` per level. This preserves orthogonality
because the CDV basis is orthonormal in each direction independently.

---

