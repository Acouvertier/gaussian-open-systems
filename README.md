# Gaussian Open Systems

A structured simulation framework for Gaussian continuous-variable (CV) open quantum systems, supporting exact moment evolution, subsystem-resolved analysis, and finite-memory reservoir modeling via pseudomode embeddings.

---

## Overview

This library provides a modular framework for constructing, evolving, and analyzing multimode Gaussian quantum systems under quadratic Hamiltonians and dissipative channels.

The implementation operates entirely in phase space, evolving mean vectors and covariance matrices exactly using linear dynamics. It is designed to bridge theoretical modeling and computational simulation, enabling both reproduction of standard results and exploration of structured open-system dynamics.

Key capabilities:

- Exact time evolution of Gaussian states under Lindblad dynamics  
- Construction of multimode systems with collective or independent dissipation  
- Subsystem-resolved metrics (entanglement, purity, entropy, fidelity)  
- Finite-memory reservoir modeling via exact pseudomode embedding  
- Strict validation of numerical and physical consistency  

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

# Quickstart Example

```python
import numpy as np
from gaussian_systems.initial_state import GaussianCVState
from gaussian_systems.systems import GaussianCVSystem

# State
state = GaussianCVState.vacuum(2)
state.single_mode_squeeze((1.0,0.0), 1)
state.single_mode_squeeze((1.0,0.0), 2)

# System
system = GaussianCVSystem.free_evolution(2, np.array([0.0, 0.0]))
system.multi_annihilation_dissipator((1, 2), 1.0)

# Evolution
t_eval = np.linspace(0, 20, 500)
solution = system.evolve_state(state, t_eval)

# Metrics
ent = solution.entanglement_time_trace((1, 2))
pur = solution.purity_time_trace()
```

---

## Notes

This project is designed to:
- provide a reusable framework for Gaussian open-system simulations
- support rapid prototyping of physically structured models
- enable direct comparison between Markovian and finite-memory dynamics

It emphasizes explicit physical structure, consistent conventions, and numerically stable implementations.
