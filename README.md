# Gaussian Open Systems

A Python framework for modeling Gaussian continuous-variable (CV) open quantum systems using explicit drift–diffusion dynamics in phase space.

The library is designed for problems where the structure of the Gaussian channel is central: it evolves mean vectors and covariance matrices exactly under quadratic Hamiltonians and Lindblad dissipation, and extends naturally to finite-memory environments via pseudomode embeddings.

### Scope and Positioning

This project is not a general-purpose quantum simulation framework. Instead, it targets a narrower class of problems:

- Gaussian states only (first and second moments)
- time-independent quadratic generators
- structured dissipation and reservoir modeling

Compared to tools such as QuTiP or Strawberry Fields, the emphasis here is on transparency of the underlying Gaussian channel (drift–diffusion form) and direct access to its dynamical structure.

---

## Overview

This library provides a modular framework for constructing, evolving, and analyzing multimode Gaussian quantum systems under quadratic Hamiltonians, Markovian dissipative channels, and non-Markovian dissipative channels via pseudomode embedding.

The implementation operates entirely in phase space, evolving mean vectors and covariance matrices exactly using linear dynamics. It is designed to bridge theoretical modeling and computational simulation, enabling both reproduction of standard results and exploration of structured open-system dynamics.

Key capabilities:

- Exact time evolution of Gaussian states under Lindblad dynamics  
- Construction of multimode systems with collective or independent dissipation  
- Subsystem-resolved metrics (entanglement, purity, entropy, fidelity)  
- Finite-memory reservoir modeling via exact pseudomode embedding  
- Strict validation of numerical and physical consistency  

---

# Quickstart Example

```python
import numpy as np
from gaussian_systems.initial_state import GaussianCVState # API layer that contains state constructors
from gaussian_systems.systems import GaussianCVSystem # API layer that contains system dynamic constructors

# State
state = GaussianCVState.vacuum(2) # Initialize the state as 2 vacuum modes
state.single_mode_squeeze((1.0,0.0), 1) # Squeeze mode #1 along it's position quadrature
state.single_mode_squeeze((1.0,np.pi), 2) # Squeeze mode #2 along it's momentum quadrature

# System
system = GaussianCVSystem.free_evolution(2, np.array([0.0, 0.0])) # Initialize the system without any dynamics
system.multi_annihilation_dissipator((1, 2), 1.0) # Update a single dissipative environment that acts on both modes (1,2). Lindblad of the form 1.0*(a_1 + a_2) for a_i is the annihilation operator of mode i

# Evolution
t_eval = np.linspace(0, 20, 500) # time grid for simulation
solution = system.evolve_state(state, t_eval) # .evolve_state applies the system to state over the time grid and returns a GaussianSolution object containing the time-trace

# Metrics
ent = solution.entanglement_time_trace((1, 2)) # computes the logarithmic negativity between modes (1,2) over the time grid defined by the solution object. Returns a a list of Reals representing the time-trace of the entanglement.
pur = solution.purity_time_trace() # computes the purity of the entire system over the time grid. Returns a list of Reals representing the time-traced purity.
```

---

## Documentation

[Full documentation is public for rapid prototyping. Click the link.](https://acouvertier.github.io/gaussian-open-systems/index.html)

---

## Core Concepts

### Gaussian Representation

An  $n$-mode Gaussian state is represented by:
- Mean vector $\mathbf{r} \in \mathbb{R}^{2n}$
- Covariance matrix $V \in \mathbb{R}^{2n \times 2n}$

using the phase-space ordering:

$$
(x_1, \dots, x_n, p_1, \dots, p_n)
$$

All transformations, evolution, and metrics are defined consistently in this convention.

---

## Core Objects

### `GaussianCVState`

Represents a Gaussian state and provides:

- Vacuum and thermal state construction  
- Gaussian unitary transformations (displacement, squeezing, mixing)  
- In-place state preparation workflows  

---

### `GaussianCVSystem`

Defines system dynamics through:

- Quadratic Hamiltonians  
- Dissipative channels (Lindblad operators encoded as Gram matrices)  

Generates drift and diffusion matrices and evolves states exactly.

---

### `GaussianSolution`

Represents the time-evolved trajectory of a system.

Provides direct access to:

- Entanglement (logarithmic negativity)  
- Purity and Rényi-2 entropy  
- Gaussian fidelity to reference states  

---

## Indexing Convention

All mode indices are **1-indexed**:

- Modes are labeled `(1, 2, ..., n)`  
- Index `0` is invalid by design  
- Subsystems are specified using physical mode labels  

Internally, indices are mapped to zero-based array positions.

---

## State Construction

```python
from gaussian_systems.initial_state import GaussianCVState

state = GaussianCVState.vacuum(2)

state.single_mode_squeeze((1.0, 0), 1) # squeezing is a tuple (magnitude,phase)
state.single_mode_squeeze((1.0, 0), 2)

state.single_mode_displacement(1.0, 1) # displacement can be any complex scalar
state.single_mode_displacement(1.0 + 2.0j, 2)
```

---

## Mutation Model

All transformation methods:

- modify the state **in place**
- return the same object

Use `.copy_state()` to create independent states.

---

## System Construction

```python
from gaussian_systems.systems import GaussianCVSystem
import numpy as np

system = GaussianCVSystem.free_evolution(2, np.array([0.0, 0.0]))
system.multi_annihilation_dissipator((1, 2), 1.0)
```

Systems are constructed by combining:
- Hamiltonian terms (quadratic couplings)
- Dissipative channels

---

## Evolution

```python
t_eval = np.linspace(0, 20, 500)
solution = system.evolve_state(state, t_eval)
```

Evolution is computed exactly using matrix exponentials of the drift and diffusion generators.
Covariance matrices are validated for quantum physicality at each step

---

## Analysis and Metrics

Metrics are evaluated directly from the solution:
```python
ent = solution.entanglement_time_trace((1,2))
pur = solution.purity_time_trace()
entropy = solution.entropy_time_trace
```

Available metrics:
- **Logarithmic negativity** (two-mode entanglement)
- **Purity**
- **Rény-2 entropy**
- **n-mode Gaussian fidelity (Uhlmann)**

All metrics support subsystem-level evaluation.

---

## Subsystems

Subsystems are specified using mode labels:
```python
pur_sub = solution.purity_time_trace((1,))
```

The above example gives the purity of subsystem/mode 1 only. Extraction follows the same phase-space ordering:

$$
(x_1, \dots, x_k, p_1, \dots, p_k)
$$

---

## Validation and Physicality

The library enforces strict validation throughout: 
- finite, real-valued numerical inputs
- consistent phase-space dimensions
- positive semidefinite covariance matrices
- valid subsystem indices **(1-indexed)**

Two levels of validity are distinguished:
- **Classical**: covariance is positive semidefinite
- **Quantum**: covariance satisfies the uncertainty relation

Quantum physicality is enforced during evolution

---

# Finite-Memory Environments

Finite correlation-time environments can be modeled using a single-pole Ornstein-Uhlenbeck kernel. This is implemented using an exact pseudomode embedding process. This can be extended to multi-pole expansions via repeated embedding. The embedding process is implemented such that:

- system and state are augmented from $n$-dimenions to ($n+1$)-dimenions, with the auxiliary mode take the ($n+1$)-index.
- non-Markovian dynamics become embedded in an enlarged Markovian framework.

---

## Limitations (Globally)

- Gaussian states only (no non-Gaussian states)
- Time-independent quadratic hamiltonians only (no modulation, control, or linear drive)
- Time-independent and linear dissipators only
- Computational scaling non-optimized
- Visualization tool `.plot_state()` generates single-mode marginals

---

## Installation

Clone the repository and install dependencies in your preferred environment.

pip install numpy scipy matplotlib

(Optional) for development and documentation:

pip install sphinx myst-parser numpydoc

---

## Dependencies

Core dependencies:

- numpy — array operations and linear algebra  
- scipy — matrix exponentials and numerical routines  
- matplotlib — visualization  

The library is designed to remain lightweight and relies only on standard scientific Python tools.

---

## Project Structure

The codebase is organized as a modular package:

gaussian_systems/  
├── initial_state.py   # Gaussian state construction and transformations  
├── systems.py         # Hamiltonian, dissipation, and evolution engine  
├── solution.py        # Time-resolved solution container  
├── metrics.py         # Entanglement, purity, entropy, fidelity  
├── conventions.py     # Phase-space ordering and subsystem utilities  
├── _validation.py     # Internal validation and safety checks  

This structure separates:

- state representation  
- system dynamics  
- numerical validation  
- observable extraction  

to maintain clarity and extensibility.

---

## Notes

This project is designed to:
- provide a reusable framework for Gaussian open-system simulations
- support rapid prototyping of physically structured models
- enable direct comparison between Markovian and finite-memory dynamics

It emphasizes explicit physical structure, consistent conventions, and numerically stable implementations.
