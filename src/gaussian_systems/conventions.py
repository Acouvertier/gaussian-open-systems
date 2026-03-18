import math
import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric

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
        

def _x_subsystem(n:Integral , indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1).astype(int)
        
def _p_subsystem(n:Integral, indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1 + n).astype(int)

def _index_list(n:Integral, indices: tuple[Integral, ...]) -> npt.NDArray[int]:
    x_idx = _x_subsystem(n,indices)
    p_idx = _p_subsystem(n,indices)
    final_idx = np.append(x_idx,p_idx).astype(int)
    return final_idx

def _compress_mean_covariance(mean_vector:npt.NDArray[np.number], covariance_matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _valid_mean_covariance(mean_vector,covariance_matrix)
    return np.append(mean_vector, covariance_matrix.flatten())

def _extract_mean_covariance(mean_covariance_vector:npt.NDArray[np.number]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    if not isinstance(mean_covariance_vector,np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(mean_covariance_vector)}.")
    if mean_covariance_vector.ndim != 1:
        raise ValueError(f"expected 1D array, got {mean_covariance_vector.ndim}D array.")
    element_count = len(mean_covariance_vector)
    n_float = 0.25*(-1 + np.sqrt(1+4*element_count))
    n = int(round(n_float))
    if 4*n*n + 2*n != element_count:
        raise ValueError(f"compressed array length {element_count} is incompatible with expected 2n + 4n^2 form for an n-mode system.")
    _valid_mode_number(n)
    
    mean_vector = mean_covariance_vector[0:2*n]
    covariance_matrix = symmetrize_matrix((mean_covariance_vector[2*n:]).reshape( (2*n,2*n) ))
    
    _valid_mean_covariance(mean_vector,covariance_matrix)

    return ( mean_vector, covariance_matrix )

"""Public"""

def symmetrize_matrix(matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _valid_square_matrix(matrix)
    return (matrix + matrix.conj().T)/2

def rotation_matrix(theta: Real) -> npt.NDArray[np.float64]:
    if not isinstance(theta,Real):
        raise TypeError(f"rotation angle must be real-valued, got {type(theta)}.")
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta),   np.cos(theta)]
    ])

def symplectic_matrix(n:int) -> npt.NDArray[np.float64]:
    _valid_mode_number(n)
    identity_matrix = np.identity(n)
    w = np.array([
        [0.0 ,1.0],
        [-1.0,0.0]
    ])
    return np.kron(w,identity_matrix)

def mean_subsystem(mean_vector:npt.NDArray[np.number], indices: tuple[Integral, ...]) -> npt.NDArray[np.number]:
    _valid_mean_vector(mean_vector)
    n = len(mean_vector)//2
    final_idx = index_list(n, indices)
    return mean_vector[final_idx]

def covariance_subsystem(covariance_matrix:npt.NDArray[np.number], indices: tuple[Integral, ...]) -> npt.NDArray[np.number]:
    _valid_covariance_matrix(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
    final_idx = index_list(n, indices)
    return covariance_matrix[final_idx,:][:,final_idx]






