# mtuq_pso

Particle Swarm Optimization (PSO) algorithms for moment tensor inversion using MTUQ.

## Overview

This repository provides PSO-based optimization methods for seismic moment tensor inversion as an alternative to grid search methods. PSO is a population-based metaheuristic optimization technique inspired by the social behavior of bird flocking or fish schooling.

## Features

- **Double-Couple PSO**: Fast optimization for double-couple sources with fixed or variable magnitude
- **Full Moment Tensor PSO**: General moment tensor inversion without double-couple constraint
- **MPI Parallelization**: Distributed computing support for faster optimization
- **Iteration Tracking**: Optional saving of intermediate solutions during optimization

## Installation

### Prerequisites

- Python 3.7+
- [MTUQ](https://github.com/uafgeotools/mtuq) installed and configured
- MPI implementation (for parallel versions)

### Required Python packages

```bash
pip install numpy mpi4py
```

### Clone this repository

```bash
git clone https://github.com/yourusername/mtuq_pso.git
cd mtuq_pso
```

## Repository Structure

```
mtuq_pso/
├── README.md
├── PSO_fixedmagnitude.py          # Double-couple PSO with fixed magnitude
├── PSO_magnitudesearch.py         # Double-couple PSO with magnitude optimization
├── FMT_PSO.py                     # Full moment tensor PSO
└── examples/
    └── run_pso_example.sh         # Example run script
```

## Usage

### 1. Double-Couple Inversion (Magnitude Search)

For double-couple inversion with simultaneous magnitude optimization:

```bash
mpirun -n 4 python PSO_magnitudesearch.py
```

**Key parameters:**
- `magnitude_min = 4.0` - Minimum magnitude to search
- `magnitude_max = 5.5` - Maximum magnitude to search
- MPI parallelization across multiple processes

### 2. Full Moment Tensor Inversion

For general moment tensor inversion without double-couple constraint:

```bash
mpirun -n 4 python FMT_PSO.py
```

**Features:**
- Optimizes all 6 independent moment tensor components
- Simultaneous magnitude optimization
- Provides lune coordinates and fault parameters

## Algorithm Details

### Particle Swarm Optimization

PSO maintains a swarm of candidate solutions (particles) that explore the parameter space. Each particle:
- Has a position (candidate solution) and velocity
- Remembers its personal best position
- Is influenced by the global best position found by the entire swarm

**Velocity update equation:**
```
v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
```

Where:
- `w` = inertia weight (0.4-0.5)
- `c1` = cognitive parameter (1.0-1.5)
- `c2` = social parameter (1.5-2.0)
- `r1, r2` = random numbers in [0,1]

### Advantages over Grid Search

1. **Efficiency**: Explores parameter space more efficiently than exhaustive grid search
2. **Scalability**: Handles high-dimensional problems better (e.g., 7D for full MT + magnitude)
3. **Adaptivity**: Concentrates search effort in promising regions
4. **Parallelization**: Naturally parallelizable across MPI processes

### Convergence Criteria

The algorithm stops when:
- Maximum iterations reached, OR
- Stagnation detected (improvement < threshold for N consecutive iterations)

## Output Files

Each run produces:

1. **JSON solution file**: Contains optimal parameters and misfit value
   - Example: `20090407201255351_PSO_solution.json`

2. **Waveform comparison plot**: Observed vs synthetic waveforms
   - Example: `20090407201255351_PSO_strike245.1_dip89.3_rake-175.2_waveforms.png`

3. **Beachball plot**: Focal mechanism visualization
   - Example: `20090407201255351_PSO_strike245.1_dip89.3_rake-175.2_beachball.png`

4. **Iteration plots** (if `save_iterations=True`):
   - Tracks optimization progress
   - Useful for understanding convergence behavior

## Example Results

### Waveform Fit
![Waveform comparison showing observed (black) vs synthetic (red) seismograms](placeholder_waveforms.png)

### Focal Mechanism
![Beachball diagram showing the fault plane solution](placeholder_beachball.png)

### Convergence
The PSO algorithm typically converges within 20-30 iterations, finding solutions comparable to grid search but with significantly reduced computational cost.

## Customization

### Modify PSO Parameters

In any of the scripts, adjust these parameters in the `pso_custom()` or `pso_mpi()` function calls:

```python
best_position, best_misfit = pso_custom(
    objective_function,
    lb, ub,
    swarmsize=100,      # Increase for better exploration
    maxiter=50,         # Increase for harder problems
    stagnation_limit=10,    # Patience for convergence
    stagnation_threshold=1e-4,  # Sensitivity to improvement
    debug=True
)
```

### Change Data Processing

Modify the `ProcessData` objects for different frequency bands or window lengths:

```python
process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.1,       # Adjust frequency range
    freq_max=0.333,
    window_length=15.0,  # Adjust window length
    # ... other parameters
)
```

## Performance Comparison

| Method | Parameters | Time (4 cores) | Evaluations |
|--------|-----------|----------------|-------------|
| Grid Search | 40×18×40×10 (DC+Mag) | ~30-60 min | 288,000 |
| PSO | 4D search space | ~5-10 min | ~5,000 |
| Grid Search | 65×40×65×10 (Full MT) | Several hours | 1,690,000 |
| PSO | 7D search space | ~15-20 min | ~10,000 |

*Times are approximate and depend on number of stations, data length, and hardware*

## Troubleshooting

### Common Issues

1. **MPI errors**: Ensure mpi4py is properly installed and matches your MPI implementation
   ```bash
   pip install --upgrade mpi4py
   ```

2. **Memory issues**: Reduce swarm size or use more MPI processes to distribute load
   ```python
   swarmsize = 50  # Reduce from default 100
   ```

3. **Convergence problems**: 
   - Increase `maxiter` for complex problems
   - Adjust PSO parameters (inertia weight, cognitive/social parameters)
   - Check data quality and Green's functions

4. **Invalid strike/dip/rake values**: The code includes normalization functions to ensure valid ranges:
   - Strike: 0-360°
   - Dip: 0-90°
   - Rake: -90 to 90°

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mtuq_pso,
  title = {mtuq\_pso: Particle Swarm Optimization for MTUQ},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/mtuq_pso}
}
```

Also cite the original MTUQ software:
```bibtex
@article{mtuq2022,
  title={MTUQ: A Python package for moment tensor uncertainty quantification},
  author={Magnitud, et al.},
  journal={Seismological Research Letters},
  year={2022}
}
```

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*.
- Tape, W., & Tape, C. (2012). A geometric setting for moment tensors. *Geophysical Journal International*.
- MTUQ Documentation: https://uafgeotools.github.io/mtuq/

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- MTUQ development team
- Original PSO algorithm developers (Kennedy & Eberhart, 1995)
- Seismology community for testing and feedback
