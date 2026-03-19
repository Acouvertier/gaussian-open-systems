import math
import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric, eigvalsh

def _valid_mode_number(n:Integral) -> None:
    if not isinstance(n, Integral):
        raise TypeError(f"number of modes must be integer-valued, got {type(n)}.")
    if n < 1:
        raise ValueError(f"number of modes must be positive, got {n}.")

def _valid_indices(n:Integral, indices:tuple[Integral ,...]) -> None:
    _valid_mode_number(n)
    if not isinstance(indices, tuple):
        raise TypeError(f"indices must be a tuple, got {type(indices)}.")
    if not all(isinstance(i, Integral) for i in indices):
        raise TypeError(f"indices must be integer valued (type int).")
    if not all(1 <= i <= n for i in indices):
        raise ValueError(f"indicies must be between 1 and {n} inclusive.")

def _valid_square_matrix(matrix:npt.NDArray[np.number]) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"expected an NDArray, got {type(matrix)}")
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D numpy array, got {matrix.ndim}D.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix (n x n), got {matrix.shape}.")

def _valid_mean_vector(mean_vector:npt.NDArray[np.number]) -> None:
    if not isinstance(mean_vector, np.ndarray):
        raise TypeError(f"expected an NDArray, got {type(mean_vector)}")
    if mean_vector.ndim != 1:
        raise ValueError(f"expected a 1D numpy array, got {mean_vector.ndim}D.")
    if mean_vector.shape[0] %  2 != 0:
        raise ValueError(f"mean vector must have even dimension, got shape {mean_vector.shape}.")
    if not np.isrealobj(mean_vector):
        raise ValueError("mean vector must be real-valued")

def _valid_covariance_matrix(covariance_matrix:npt.NDArray[np.number]) -> None:
    _valid_square_matrix(covariance_matrix)
    if covariance_matrix.shape[0] %  2 != 0:
        raise ValueError(f"covariance matrix must have even dimension, got shape {covariance_matrix.shape}.")
    if not np.isrealobj(covariance_matrix):
        raise ValueError("covariance matrix must be real-valued.")
    if not issymmetric(covariance_matrix, atol=1e-8, rtol=1e-8):
        raise ValueError("covariance matrix must be approximately symmetric.")

def _valid_mean_covariance(mean_vector:npt.NDArray[np.number],covariance_matrix:npt.NDArray[np.number]) -> None:
    _valid_mean_vector(mean_vector)
    _valid_covariance_matrix(covariance_matrix)
    if mean_vector.shape[0] != covariance_matrix.shape[0]:
        raise ValueError(f"mean vector and covariance matrix dimensions must match, got {mean_vector.shape[0]} and {covariance_matrix.shape}.")

def _valid_nbars_array(nbars:npt.NDArray[np.float64]) -> None:
    if not isinstance(nbars, np.ndarray):
        raise TypeError(f"thermal occupations must be a numpy array. Got {type(nbars)}")
    if nbars.ndim != 1:
        raise ValueError(f"thermal occupation must be a 1D numpy array. Got array with dimensions {nbars.ndim}")
    if not np.isrealobj(nbars):
        raise ValueError(f"thermal occupations must be real-valued.")
    if not np.all(nbars >= 0):
        raise ValueError(f"thermal occupations must be non-negative. Got {nbars}.")

def _valid_tuple_pair(tuple_pair:tuple) -> None:
    if not isinstance(tuple_pair, tuple):
        raise TypeError(f"Expected tuple to unpack. Got {type(tuple_pair)}")
    if len(tuple_pair) != 2:
        raise ValueError(f"Expected a tuple of length 2. Got {tuple_pair}.")

def _valid_fidelity_input(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]):
    _valid_tuple_pair(mean_covariance_tuple1)
    _valid_tuple_pair(mean_covariance_tuple1)
    
    mean1, cov1 = mean_covariance_tuple1
    mean2, cov2 = mean_covariance_tuple2
   
    _valid_mean_covariance(mean1,cov1)
    _valid_mean_covariance(mean2,cov2)
    if len(mean1) != len(mean2):
        raise ValueError(f"Providd states tuple (pair of states) have mismatched dimensions. State 1 is {len(mean1)}D, while State 2 is {len(mean2)}D. Got {mean_covariance_tuple1} and {nean_covariance_tuple2}")

def _valid_t_eval(t_eval:npt.NDArray[np.float64]) -> None:
    if not isinstance(t_eval, np.ndarray):
        raise TypeError(f"expected an NDArray, got {type(t_eval)}")
    if t_eval.ndim != 1:
        raise ValueError(f"expected a 1D numpy array, got {t_eval.ndim}D.")
    if not np.isrealobj(t_eval):
        raise ValueError("evaluation times must be real-valued")
    if np.min(t_eval) < 0:
        raise ValueError(f"evaluation times cannot go below 0. Got min evaluation for t = {np.min(t_eval)}")

def _valid_mean_vector_covariance_matrix_tuple(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> None:
    _valid_tuple_pair(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    _valid_mean_covariance(mean_vector, covariance_matrix)

def _valid_parameter_pair(magnitude:Real, angle:Real) -> None:
    if not isinstance(magnitude, Real):
        raise TypeError(f"parameter magnitude must be real-valued. Got {type(magnitude)}.")
    if magnitude < 0:
        raise ValueError(f"parameter magnitude must be non-negative. Got {magnitude}")
    if not isinstance(angle, Real):
        raise TypeError(f"parameter angle must be real-valued. Got {angle}.")

def _valid_parameter_tuple(parameter_magnitude_angle_tuple:tuple) -> None:
    _valid_tuple_pair(parameter_magnitude_angle_tuple)
    magnitude, angle = parameter_magnitude_angle_tuple
    _valid_parameter_pair(magnitude, angle)

def _valid_frequency_array(frequency_array:npt.NDArray[np.float64]) -> None:
    if not isinstance(frequency_array, np.ndarray):
        raise TypeError(f"mode frequencies must be a numpy array. Got {type(frequency_array)}")
    if frequency_array.ndim != 1:
        raise ValueError(f"mode frequencies must be a 1D numpy array. Got array with dimensions {frequency_array.ndim}")
    if not np.isrealobj(frequency_array):
        raise ValueError(f"mode frequencies must be real-valued.")

def _valid_hamiltonian_parameter(coefficient:Real) -> None:
    if not isinstance(coefficient,Real):
        raise TypeError(f"hamiltonian construction requires real-valued coefficients. Got {type(coefficient)}")

def _valid_term_inputs(n:Integral, mode_id:Integral|tuple[Integral,Integral], coefficient:Real):
    _valid_hamiltonian_parameter(coefficient)
    _valid_mode_number(n)
    if not isinstance(mode_id, tuple):
        if not isinstance(mode_id, Integral):
            raise ValueError(f"mode designation must be a single index of a tuple with exactly 2 indices. Got {type(mode_id)}.")
        else: 
            _valid_indices(n,(mode_id,))
    elif len(mode_id) != 2:
        raise ValueError(f"mode designoation must a tuple with exactly 2 indices. Got tuple of length {len(mode_id)}")
    _valid_indices(n, mode_id)

def _valid_hamiltonian_matrix(ham:npt.NDArray[Real]):
    _valid_covariance_matrix(ham)

def _valid_lindblad_gram_matrix(M: npt.NDArray[complex], atol=1e-8):
    _valid_square_matrix(M)

    if M.shape[0] % 2 != 0:
        raise ValueError("must be even-dimensional")

    M_herm = (M + M.conj().T)/2
    eigs = eigvalsh(M_herm)

    if np.any(eigs < -atol):
        raise ValueError("Hermitian part must be positive semidefinite")

def _valid_system(ham, M):
    _valid_hamiltonian_matrix(ham)
    _valid_lindblad_gram_matrix(M)

def _valid_single_pole_input(n:Integral, subsystem:tuple[Integral,...], coupling_types:tuple[str,...], memory_rate:Real, env_freq:Real, thermal_occupation:Real) -> None:
    if not isinstance(thermal_occupation,Real):
        raise TypeError(f"thermal occupation must be real. Got {type(thermal_occupation)}")
    if thermal_occupation < 0:
        raise ValueError(f"thermal occupation must be non-negative. Got {thermal_occupation}")
    if not isinstance(memory_rate,Real):
        raise TypeError(f"memory rate must be real. Got {type(memory_rate)}")
    if memory_rate < 0:
        raise ValueError(f"memory rate must be non-negative. Got {memory_rate}")
    if not isinstance(env_freq,Real):
        raise TypeError(f"environmental frequency must be real. Got {type(env_freq)}")
    if not isinstance(n,Integral):
        raise TypeError(f"mode count must be integer. Got {type(n)}")
    if n <= 0:
        raise ValueError(f"mode count must be positive. Got {n}")
    _valid_indices(n, subsystem)
    
    if not isinstance(coupling_types, tuple):
        raise TypeError(f"Pseudo-coupling structure must be a tuple of string. Got {type(coupling_types)}")
    if not all([elt in ["annihilation", "position", "momentum"] for elt in coupling_types]):
        raise TypeError(f"Psuedo-coupling structure must contain strings that are either 'annihilation', 'position', 'momentum'. Got {coupling_types}")
    if len(coupling_types) != len(subsystem):
        raise ValueError(f"Pseudo-coupling structure must be same length as subsystems provided. Got {coupling_types}")

def _valid_decay_element(decay_element:tuple[str,Integral,Real]) -> None:
    if not isinstance(decay_element, tuple):
        raise TypeError(f"Dissipation generation requires jump operator information contained in a tuple. Got {type(decay_element)}")
    if len(decay_element) != 3:
        raise ValueError(f"Jump operator is defined by three parts. Got {decay_element}")
    decay_str, decay_int, decay_complex = decay_element
    if (not isinstance(decay_str, str)) or (decay_str not in ["a","x","p","ad"]):
        raise ValueError(f"First part of jump definition must be string representing type. Got {decay_str}")
    if (not isinstance(decay_int, Integral)) or (decay_int <= 0):
        raise ValueError(f"Second part of jump definition must be natural number representing affected mode. Got {decay_int}")
    if (not isinstance(decay_complex, Real)):
        raise TypeError(f"Tird part of jump definition must be the real-valued decay rate. Got {decay_complex}")