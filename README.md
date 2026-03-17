### Phase-Space Representation
- Quadrature ordering:
  (x₁, x₂, …, xₙ, p₁, p₂, …, pₙ)
- Mean vector and covariance matrix representation
- Symplectic form construction and subsystem extraction utilities

### Gaussian State Construction
- Thermal and vacuum states
- Coherent displacements
- Single-mode squeezing
- Two-mode squeezing (direct symplectic construction)

### Gaussian Transformations
- Single-mode Gaussian unitaries
- Two-mode mixing (beam splitter–type transformations)
- Two-mode squeezing operations
- Subsystem embedding into arbitrary n-mode systems

### Metrics and Diagnostics
- Logarithmic negativity (Gaussian entanglement)
- State purity
- Rényi-2 entropy
- Gaussian state fidelity (1-mode, 2-mode, and general n)

---

## Project Structure

```text
project_root/
├── src/
│   └── gaussian_systems/
│       ├── conventions.py      # indexing, symplectic form, subsystem utilities
│       ├── initial_state.py    # state construction and Gaussian transformations
|       ├── systems.py          # Hamiltonian construction, drift and diffusion automation, multiple independent environments
│       └── metrics.py          # entanglement, purity, fidelity
|
├── scripts/
|   └── experiments/
|   └── validation/
|       ├── conventions_validation.ipynb          
│       └── initial_states_validation.ipynb          
|
├── notebooks/
|
├── results/
|
├── docs/
├── tests/                      # (planned) pytest-based validation suite
├── pyproject.toml
└── README.md
```

---

## Design Principles

### 1. Quadrature-Ordered Formalism
All objects are defined in the global ordering:

(x₁, …, xₙ, p₁, …, pₙ)

All subsystem operations and embeddings are consistent with this convention.

---

### 2. Local-to-Global Embedding

All Gaussian unitaries are constructed locally (2×2 or 4×4) and embedded into the full system using projection operators:

S = I + Pᵀ (S_local − I) P

This ensures transformations act only on the intended modes.

---

### 3. Direct Symplectic Construction

Two-mode squeezing is implemented directly in quadrature space rather than via beam-splitter decompositions, thereby avoiding basis inconsistencies.

---

## Example Usage

### Initialize Vacuum State

```python
import numpy as np
from gaussian_systems.initial_state import (
  thermal_vacuum_mean,
  thermal_vacuum_covariance
)

mean = thermal_vacuum_mean(n=2)
cov = thermal_vacuum_covariance(n=2, nbars=np.array([0.0, 0.0]))
```
```python
from gaussian_systems.initial_state import apply_2_mode_squeeze_unitary

r, phi = 0.5, 0.0

mean, cov = apply_2_mode_squeeze_unitary(
    (mean, cov),
    (r, phi),
    (1, 2)
)
```
```python
from gaussian_systems.metrics import compute_logarithmic_negativity
E = compute_logarithmic_negativity(cov)
print(E) #1.0
```
## Current Status

This project is under active development.

- Core Gaussian transformations are implemented  
- Metric functions are available  
- Validation notebooks confirm expected qualitative behavior (e.g., entanglement generation)  

However:

- Formal test suite (pytest) is in progress  
- Numerical robustness and edge-case validation are ongoing  
- API is subject to refinement  

---

## Limitations

- No full automated test coverage yet  
- No guarantees of numerical stability outside tested regimes  
- Assumes correct usage of quadrature ordering  
- Limited input validation (invalid states may not raise errors)  

---

## Planned Improvements

- Full pytest-based validation suite  
- Explicit physicality checks (uncertainty condition enforcement)  
- Improved error handling and input validation  
- Documentation of conventions and formulas  
- Performance optimization for larger mode numbers  

---

## Intended Use

This toolkit is designed for:

- Research prototyping in Gaussian quantum systems  
- Simulation of CV quantum optics models  
- Development of reusable Gaussian-state infrastructure  

It is not yet intended as a production-grade library.

---

## Dependencies

- numpy  
- scipy  
- matplotlib (for visualization)  

---

## Notes

This codebase prioritizes:

- transparency of mathematical structure  
- explicit symplectic construction  
- modular composition of Gaussian operations  

over abstraction-heavy design.
