# Quickstart

This example constructs a two-mode Gaussian state, defines a dissipative system, evolves the state in time, and evaluates simple observables from the resulting trajectory.

## Imports

```python
import numpy as np

from gaussian_systems.initial_state import GaussianCVState
from gaussian_systems.systems import GaussianCVSystem
```

## 1. Construct a Gaussian state

Create a two-mode vacuum state and apply single-mode squeezing to both modes.

```python
state = GaussianCVState.vacuum(2)

state.single_mode_squeeze((1.0, 0.0), 1)
state.single_mode_squeeze((1.0, 0.0), 2)
```

The library uses **1-indexed mode labels**, so the two modes are labeled `1` and `2`.

State transformations act **in place** and return `self`, which allows chaining if desired.

```python
state = GaussianCVState.vacuum(2)
state.single_mode_squeeze((1.0, 0.0), 1).single_mode_squeeze((1.0, 0.0), 2)
```

## 2. Construct a Gaussian system

Create a two-mode system with free evolution and add a collective annihilation dissipator.

```python
system = GaussianCVSystem.free_evolution(2, np.array([0.0, 0.0]))
system.multi_annihilation_dissipator((1, 2), 1.0)
```

More structured models can be built by adding Hamiltonian couplings and different dissipative channels.

## 3. Evolve the state

Choose a time grid and evolve the state.

```python
t_eval = np.linspace(0.0, 20.0, 500)
solution = system.evolve_state(state, t_eval)
```

The returned `GaussianSolution` object stores:

- the time grid
- the evolved mean vectors
- the evolved covariance matrices

## 4. Evaluate observables

Compute a few trajectory-level quantities.

```python
entanglement = solution.entanglement_time_trace((1, 2))
purity = solution.purity_time_trace()
entropy = solution.entropy_time_trace()
```

These are evaluated directly from the evolved covariance matrices.

## 5. Compare to a reference state

You can also compute fidelity to a fixed Gaussian state.

```python
target = GaussianCVState.vacuum(2)
fidelity = solution.fidelity_time_trace_fixed(target)
```

## Notes

- Phase-space ordering is always

$$
(x_1, \dots, x_n, p_1, \dots, p_n)
$$

- Subsystems are specified using **1-indexed** mode labels.
- Evolution checks covariance matrices for quantum physicality at each time step.
- The library is restricted to **Gaussian states** and **time-independent** generators.

## Next Steps

For the mathematical conventions used throughout the package, see the **Core Concepts** page.

For more complete demonstrations, including structured reservoirs and finite-memory embeddings, see the **Examples** page.