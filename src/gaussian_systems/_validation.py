from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric, eigvalsh

Array = npt.NDArray[np.generic]

_ALLOWED_COUPLING_TYPES = ("annihilation", "position", "momentum")

_ALLOWED_DECAY_TYPES = ("a", "x", "p", "ad")

def _require_type(value, expected_type, name:str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type}, got {type(value)}.")

def _require_finite(value:Real|Array, name:str) -> None:
    if isinstance(value, Real):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar. Got {value}")
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite array. Got {value}")
    else:
        raise TypeError(f"{name} must be a real scalar or numpy array. Got {valye}")

def _require_real_scalar(value: Real, name: str) -> None:
    _require_type(value, Real, name)
    _require_finite(value, name)

def _require_integral_scalar(value: Integral, name: str) -> None:
    _require_type(value, Integral, name)

def _require_nonnegative_real_scalar(value:Real, name: str) -> None:
    _require_real_scalar(value, name)
    sign_criteria = value >= 0
    if not sign_criteria:
        raise ValueError(f"{name} must be non-negative real scalar. Got {value}.")

def _require_positive_real_scalar(value:Real, name: str) -> None:
    _require_real_scalar(value, name)
    sign_criteria = value > 0
    if not sign_criteria:
        raise ValueError(f"{name} must be positive real scalar. Got {value}.")

def _require_nonnegative_integral_scalar(value:Integral, name: str) -> None:
    _require_integral_scalar(value, name)
    sign_criteria = value >= 0
    if not sign_criteria:
        raise ValueError(f"{name} must be non-negative integer scalar. Got {value}.")

def _require_positive_integral_scalar(value:Integral, name: str) -> None:
    _require_integral_scalar(value, name)
    sign_criteria = value > 0
    if not sign_criteria:
        raise ValueError(f"{name} must be positive integer scalar. Got {value}.")

def _require_tuple_length(value: tuple, length: Integral, name: str) -> None:
    _require_type(value, tuple, name)
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {value}")

def _require_ndarray(arr: Array, name: str) -> None:
    _require_type(arr, np.ndarray, name)

def _require_ndim(arr: Array, ndim: Integral, name: str) -> None:
    _require_ndarray(arr,  name)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.ndim}D.")

def _require_real_array(arr: Array, name: str) -> None:
    _require_ndarray(arr, name)
    _require_finite(arr, name)
    if not np.isrealobj(arr):
        raise ValueError(f"{name} must be real-valued.")
    
def _require_real_vector(arr:Array, name:str) -> None:
    _require_ndim(arr, 1, name)
    _require_real_array(arr, name)

def _require_nonnegative_real_vector(arr: Array, name:str) -> None:
    _require_real_vector(arr, name)
    sign_criteria = np.all(arr >= 0)
    if not sign_criteria:
        raise ValueError(f"{name} must contain only non-negative real values. Got {arr}.")

def _require_positive_real_vector(arr: Array, name:str) -> None:
    _require_real_vector(arr, name)
    sign_criteria = np.all(arr > 0)
    if not sign_criteria:
        raise ValueError(f"{name} must contain only positive real values. Got {arr}.")

def _require_square_matrix(matrix: Array, name: str) -> None:
    _require_ndim(matrix, 2, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")

def _require_even_vector_length(arr: Array, name: str) -> None:
    _require_ndim(arr, 1, name)
    if arr.shape[0] % 2 != 0:
        raise ValueError(f"{name} must have even dimension, got shape {arr.shape}")

def _require_even_matrix_dimension(arr: Array, name: str) -> None:
    _require_ndim(arr, 2, name)
    for dim in arr.shape:
        if dim % 2 != 0:
            raise ValueError(f"{name} must have even dimension, got shape {arr.shape}")

def _require_symmetric(matrix: Array, name: str, *,atol:float = 1e-8, rtol:float = 1e-8) -> None:
    _require_square_matrix(matrix, name)
    if not issymmetric(matrix, atol=atol, rtol=rtol):
        raise ValueError(f"{name} must be approximately symmetric. Got {matrix}")

def _valid_mode_number(n: Integral) -> None:
    _require_positive_integral_scalar(n, 'number of modes')


def _valid_indices(n:Integral, indices:tuple[Integral ,...]) -> None:
    name = 'subsystem indices'
    _valid_mode_number(n)
    _require_type(indices, tuple, name)
    if not all(isinstance(i, Integral) for i in indices):
        raise TypeError(f"{name} must be integer valued. Got {indices}")
    if not all(1 <= i <= n for i in indices):
        raise ValueError(f"{name} must be between 1 and {n} inclusive. Got {indices}")

def _valid_mean_vector(mean_vector:Array) -> None:
    name = 'mean vector'
    _require_real_vector(mean_vector, name)
    _require_even_vector_length(mean_vector, name)

def _valid_covariance_matrix(covariance_matrix:Array) -> None:
    name = 'covariance matrix'
    _require_real_array(covariance_matrix, name)
    _require_even_matrix_dimension(covariance_matrix, name)
    _require_symmetric(covariance_matrix, name)

def _valid_mean_covariance(mean_vector:Array,covariance_matrix:Array) -> None:
    _valid_mean_vector(mean_vector)
    _valid_covariance_matrix(covariance_matrix)
    if mean_vector.shape[0] != covariance_matrix.shape[0]:
        raise ValueError(f"mean vector and covariance matrix dimensions must match, got {mean_vector.shape[0]} and {covariance_matrix.shape}.")

def _valid_nbars_array(nbars:Array) -> None:
    name = 'thermal occupations'
    _require_nonnegative_real_vector(nbars,name)

def _valid_mean_covariance_tuple(state: tuple[Array, Array]) -> None:
    _require_tuple_length(state, 2, 'given state')
    mean_vector, covariance_matrix = state
    _valid_mean_covariance(mean_vector, covariance_matrix)

def _valid_fidelity_input(state1:tuple[Array, Array], state2:tuple[Array, Array]) -> None:
    _valid_mean_covariance_tuple(state1,'state1')
    _valid_mean_covariance_tuple(state2, 'state2')
    
    mean1, cov1 = state1
    mean2, cov2 = state2
   
    if mean1.shape[0] != mean2.shape[0]:
        raise ValueError(f"states must have matching dimensions, got {mean1.shape[0]} and {mean2.shape[0]}")

def _valid_t_eval(t_eval:Array) -> None:
    name = "time grid"
    _require_nonnegative_real_vector(t_eval, name)
    if t_eval.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError(f"{name} must be strictly increasing. Got {t_eval}.")

def _valid_parameter_tuple(parameter_tuple: tuple[Real,Real], name1: str, name2: str) -> None:
    _require_tuple_length(parameter_tuple, 2, 'state/system parameters')
    
    param1, param2 = parameter_tuple
    _require_real_scalar(param1, name1)
    _require_real_scalar(param2, name2)

def _valid_frequency_array(frequency_array: Array) -> None:
    _require_real_vector(frequency_array, 'mode frequencies')

def _valid_hamiltonian_parameter(coefficient: Real) -> None:
    _require_real_scalar(coefficient, 'coupling coefficient')

def _valid_subsystem(n:Integral, mode_id: Integral|tuple[Integral,...]) -> None:
    _valid_mode_number(n)
    if not isinstance(mode_id, tuple) and not isinstance(mode_id, Integral):
        raise TypeError(f"subsystem must be either a single integer mode index or a tuple of integer mode indices, got {type(mode_id)}.")
    if isinstance(mode_id, tuple):
        _valid_indices(n, mode_id)
    else:
        _valid_indices(n,(mode_id))

def _valid_term_inputs(n:Integral, mode_id: tuple[Integral,Integral], coefficient: Real):
    _valid_hamiltonian_parameter(coefficient)
    _require_tuple_length(mode_id, 2, 'coupling indices')
    _valid_subsystem(n, mode_id)
    
    
def _valid_hamiltonian_matrix(hamiltonian: Array):
    name = 'hamiltonian'
    _require_real_array(hamiltonian, name)
    _require_even_matrix_dimension(hamiltonian, name)
    _require_symmetric(hamiltonian, name)

def _valid_lindblad_gram_matrix(M: Array, atol=1e-8):
    name = 'jump operator'
    _require_finite(M, name)
    _require_even_matrix_dimension(M, name)
    _require_square_matrix(M, name)

    M_herm = (M + M.conj().T)/2
    eigs = eigvalsh(M_herm)

    if np.any(eigs < -atol):
        raise ValueError(f"Hermitian part of {name} must be positive semi-definite up to tolerance. Got eigenspectrum {eigs}.")

def _valid_system(ham: Array, M: Array):
    _valid_hamiltonian_matrix(ham)
    _valid_lindblad_gram_matrix(M)

def _valid_single_pole_input(n:Integral, subsystem:Integral|tuple[Integral,...], coupling_types:tuple[str,...], memory_rate:Real, env_freq:Real, thermal_occupation:Real) -> None:
    _require_nonnegative_real_scalar(thermal_occupation, 'thermal occupation')
    _require_real_scalar(env_freq, 'environmental frequency')
    _require_positive_real_scalar(memory_rate, 'memory rate')
    _valid_subsystem(n, subsystem)
    if isinstance(subsystem,tuple):
        coupled_mode_count = len(subsystem)
    else:
        coupled_mode_count = 1
    _require_tuple_length(coupling_types, coupled_mode_count, 'coupling types')
    
    if not all([elt in _ALLOWED_COUPLING_TYPES for elt in coupling_types]):
        raise ValueError(f"pseudo-coupling types must contain strings that are either 'annihilation', 'position', 'momentum'. Got {coupling_types}")

def _valid_decay_element(decay_element:tuple[str,Integral,Real]) -> None:
    _require_tuple_length(decay_element, 3, 'decay code')
    decay_str, decay_int, decay_rate = decay_element
    _require_type(decay_str, str, 'decay type')
    _require_positive_integral_scalar(decay_int, 'decaying mode id')
    _require_nonnegative_real_scalar(decay_rate, 'decay rate')
    if decay_str not in _ALLOWED_DECAY_TYPES:
        raise ValueError(f"decay element must contain a string code for the decay type. Got {decay_str}")