import math
import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric, eigvals

from ._validation import _valid_indices, _valid_mean_covariance, _valid_mode_number, _require_square_matrix, _valid_mean_vector, _valid_covariance_matrix, _require_real_scalar, _require_real_vector, _require_positive_real_scalar

def _x_subsystem(n:Integral , indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1).astype(int)
        
def _p_subsystem(n:Integral, indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1 + n).astype(int)

"""Public"""
    
def index_list(n:Integral, indices: tuple[Integral, ...]) -> npt.NDArray[int]:
    x_idx = _x_subsystem(n,indices)
    p_idx = _p_subsystem(n,indices)
    final_idx = np.append(x_idx,p_idx).astype(int)
    return final_idx

def symmetrize_matrix(matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _require_square_matrix(matrix, "provided matrix")
    return (matrix + matrix.conj().T)/2

def rotation_matrix(theta: Real) -> npt.NDArray[np.float64]:
    _require_real_scalar(theta, "rotation angle")
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

def compress_mean_covariance(mean_vector:npt.NDArray[np.number], covariance_matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _valid_mean_covariance(mean_vector,covariance_matrix)
    return np.append(mean_vector, covariance_matrix.flatten())

def extract_mean_covariance(mean_covariance_vector:npt.NDArray[np.number]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    _require_real_vector(mean_covariance_vector)
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

def physical_covariance_matrix(covariance_matrix:npt.NDArray[np.float64], tol:np.number = 1e-8) -> bool:
    _valid_covariance_matrix(covariance_matrix)
    _require_positive_real_scalar(tol, "positive spectrum tolerance")
        
    n = (covariance_matrix.shape[0])//2
        
    cov_sym = symmetrize_matrix(covariance_matrix)
    
    eigs = np.real(eigvals(cov_sym + 0.5j*symplectic_matrix(n)))
    return np.all(eigs >= -1*tol)

def require_physical_covariance(covariance_matrix:npt.NDArray[np.float64]) -> None:
    _valid_covariance_matrix(covariance_matrix)
    if not physical_covariance_matrix(covariance_matrix):
        raise ValueError(f"Provided covariance matrix failed the heisenberg uncertainty relation. Got {covariance_matrix}")






