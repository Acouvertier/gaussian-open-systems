# Examples

This project includes three representative notebooks that demonstrate correctness, physical consistency, and advanced modeling capabilities.

Each example is designed to highlight a distinct aspect of Gaussian open-system dynamics.

---

## 1. Validation: Two-Mode Squeezing under Thermal Loss

**Notebook:**  
`notebooks/01_validation_two_mode_squeezing_under_loss.ipynb`

### Purpose

Establishes correctness of Gaussian state evolution under independent thermal dissipation.

### Setup

- Two-mode squeezed vacuum state  
- Independent Markovian environments  
- Variable thermal occupation $\bar{n}$

### Key Results

- Entanglement decays monotonically to zero  
- Higher $\bar{n}$ accelerates disentanglement  
- Purity behavior distinguishes:
  - $\bar{n} = 0$: transient mixing → recovery to pure vacuum  
  - $\bar{n} > 0$: monotonic decay → thermal steady state  

### What this demonstrates

- Correct drift–diffusion implementation  
- Correct steady-state structure  
- Quantitative agreement with known Gaussian results  

---

## 2. Collective Dissipation: Phase-Dependent Entanglement Generation

**Notebook:**  
`notebooks/02_markov_common_bath_baseline.ipynb`

### Purpose

Demonstrates environment-induced entanglement and its dependence on squeezing phase.

### Setup

- Two separable single-mode squeezed states  
- Collective annihilation dissipator  
- Relative squeezing phase $\Delta\phi$

### Key Results

- Aligned squeezing ($\Delta\phi = 0$) → maximal steady entanglement  
- Partial misalignment → reduced entanglement  
- Orthogonal squeezing ($\Delta\phi = \pi$) → suppressed entanglement  

### What this demonstrates

- Dissipation can generate entanglement from separable inputs  
- Squeezing phase controls coupling to dissipative structure  
- Existence of configurations with no entanglement generation  

---

## 3. Finite-Memory Environments: OU Pseudomode Embedding

**Notebook:**  
`notebooks/03_ou_pseudomode_embedding_showcase.ipynb`

### Purpose

Shows how finite environmental memory modifies Gaussian dynamics using an exact pseudomode embedding.

---

### A. Resonant aligned squeezing

**Setup**

- Resonant modes  
- Aligned squeezing  
- Comparison: Markov vs OU memory  

**Key Results**

- OU dynamics exhibit overshoot and damped oscillations in entanglement  
- Markov dynamics rapidly reach steady state  
- OU purity remains lower over long times due to persistent system–memory correlations  

**Interpretation**

Finite memory reshapes the transient pathway to entanglement without eliminating the underlying steady-state mechanism.

---

### B. Detuned orthogonal squeezing

**Setup**

- Detuned modes  
- Orthogonal squeezing  
- Comparison: Markov vs OU memory  

**Key Results**

- Markov case remains effectively disentangled  
- OU dynamics generate repeated, decaying entanglement bursts  
- Purity indicates sustained mixing during memory-mediated exchange  

**Interpretation**

Finite memory enables transient entanglement in configurations that are inactive in the Markovian limit.

---

## Summary

These examples demonstrate:

- Correct Gaussian open-system evolution  
- Phase-sensitive dissipative structure  
- Finite-memory effects beyond Markovian dynamics  

Together, they provide a progression from validation to physically structured modeling.