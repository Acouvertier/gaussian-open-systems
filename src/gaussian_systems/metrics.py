import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvals, det, cholesky, solve, eigh, issymmetric
from numbers import Real

from .conventions import mean_subsystem, covariance_subsystem, symplectic_matrix, symmetrize_matrix, physical_covariance_matrix
from ._validation import _valid_fidelity_input, _valid_covariance_matrix, _valid_square_matrix, _valid_indices, _valid_tuple_pair

_ppt_matrix = np.diag([1,1,-1,1])

def _valid_physical_covariance(covariance_matrix:npt.NDArray[np.float64]) -> None:
    _valid_covariance_matrix(covariance_matrix)
    if not physical_covariance_matrix(covariance_matrix):
        raise ValueError(f"Provided covariance matrix failed the heisenberg uncertainty relation. Got {covariance_matrix}")

def _lambda_matrix(covariance_matrix:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    _valid_physical_covariance(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    return symmetrize_matrix(-omega_matrix @ covariance_matrix @ omega_matrix @ covariance_matrix)

def _gamma_matrix(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    _valid_physical_covariance(covariance_matrix1)
    _valid_physical_covariance(covariance_matrix2)
    if (covariance_matrix1.shape)[0] != (covariance_matrix2.shape)[0]:
        raise ValueError(f"Cannot combine covariance matrics with different dimensions. Got system1: {covariance_matrix1} and system2: {covariance_matrix2}")
    n = (covariance_matrix1.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    return omega_matrix @ covariance_matrix1 @ omega_matrix @ covariance_matrix2 - 0.25*np.identity(2*n)

def _sigma_matrix(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    _valid_physical_covariance(covariance_matrix1)
    _valid_physical_covariance(covariance_matrix2)
    if (covariance_matrix1.shape)[0] != (covariance_matrix2.shape)[0]:
        raise ValueError(f"Cannot combine covariance matrics with different dimensions. Got system1: {covariance_matrix1} and system2: {covariance_matrix2}")
    return symmetrize_matrix(covariance_matrix1 + covariance_matrix2)

def _logdet_spd(matrix:npt.NDArray[np.float64]) -> np.float64:
    _valid_square_matrix(matrix)
    if not issymmetric(matrix, atol=1e-8, rtol=1e-8):
        raise ValueError(f"cholesky expected a matrix that is approximately symmetric. Got {matrix}")
    if not np.allclose(eigvals(matrix), np.zeros(matrix.shape[0])):
        raise ValueError(f"cholesky expected a matrix that is approximately positive. Got {matrix}")
    logdet_val = 2*np.sum(np.log(np.diagonal(cholesky(matrix))))
    return logdet_val

def _fidelity_fixed_parts(mean_covariance_tuple1:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], mean_covariance_tuple2:tuple[npt.NDArray[np.number],npt.NDArray[np.number]]) -> tuple[np.number,np.number]:
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    mean1, covariance1 = mean_covariance_tuple1
    mean2, covariance2 = mean_covariance_tuple2
    du = mean2 - mean1
    sigma = _sigma_matrix(covariance1,covariance2)
    return (_logdet_spd(sigma),np.real(np.exp(-0.25*du@(solve(sigma,du)))))

def _calculate_lambda(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> np.number:
    _valid_physical_covariance(covariance_matrix1)
    _valid_physical_covariance(covariance_matrix2)
    if (covariance_matrix1.shape)[0] != (covariance_matrix2.shape)[0]:
        raise ValueError(f"Cannot combine covariance matrics with different dimensions. Got system1: {covariance_matrix1} and system2: {covariance_matrix2}")
    lambdas = [_lambda_matrix(cov) for cov in [covariance_matrix1, covariance_matrix2]]
    eigens = [np.sqrt(np.abs(eigvals(lamb))) for lamb in lambdas]

    n = (covariance_matrix1.shape)[0]//2
    vals = [np.prod(np.real((spectrum-0.5)*(spectrum+0.5))) for spectrum in eigens]
    return np.max([0,(4**n)*vals[0]*vals[1]])

def _calculate_gamma(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> np.number:
    _valid_physical_covariance(covariance_matrix1)
    _valid_physical_covariance(covariance_matrix2)
    if (covariance_matrix1.shape)[0] != (covariance_matrix2.shape)[0]:
        raise ValueError(f"Cannot combine covariance matrics with different dimensions. Got system1: {covariance_matrix1} and system2: {covariance_matrix2}")
    n = (covariance_matrix1.shape)[0]//2
    gamma = _gamma_matrix(covariance_matrix1, covariance_matrix2)
    gammaSign, gammaVal = np.linalg.slogdet(gamma)
    return np.max([0,(4**n)*gammaSign*np.exp(gammaVal)])

"""PUBLIC"""

def compute_logarithmic_negativity(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,int] = (1,2)) -> Real:
    _valid_physical_covariance(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
   
    _valid_tuple_pair(subsystem)
    _valid_indices(n, subsystem)
    sub_covariance = symmetrize_matrix(covariance_subsystem(covariance_matrix,subsystem))
    omega_matrix = symplectic_matrix(2)
    symplectic_form = 1j*omega_matrix@_ppt_matrix@sub_covariance@_ppt_matrix
    nu_min = np.min(np.abs(eigvals(symplectic_form)))
    log_neg = np.max([0,-np.log(2*nu_min)])
    return log_neg
    
def state_purity(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,...]|None = None) -> Real:
    _valid_physical_covariance(covariance_matrix)
    n =  (covariance_matrix.shape)[0]//2
    if subsystem is None:
        subsystem = tuple(range(1,n+1))
    _valid_indices(n, subsystem)
    return 1/np.sqrt(det(2*symmetrize_matrix(covariance_subsystem(covariance_matrix,subsystem))))

def renyi_two_entropy(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,...]|None = None)  -> Real:
    return -np.log(state_purity(covariance_matrix,subsystem))

def one_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    delta, exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    lambda_value = _calculate_lambda(covariance1,covariance2)
    return np.sqrt((np.sqrt(delta+lambda_value)+np.sqrt(lambda_value))/delta)*exponential_part
    
def two_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    delta, exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    lambda_value = _calculate_lambda(covariance1,covariance2)
    gamma_value = _calculate_gamma(covariance1,covariance2)
    return np.sqrt((np.sqrt(gamma_value)+np.sqrt(lambda_value)+np.sqrt(np.sqrt(gamma_value + lambda_value)**2 - delta))/delta)*exponential_part

def n_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    delta, exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    n = (covariance1.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    
    w1, w2 = [-2j*cov@omega_matrix for cov in [covariance1, covariance2]]
    w1_w2_inv = solve(w1 + w2,np.identity(2*n))
    w_auxiliary = -  w1_w2_inv @ (np.identity(2*n) + w2@w1) 
    w_eigs = np.real(eigvals(w_auxiliary))
    f_tot = np.prod(np.array([np.sqrt(wi + np.sqrt((wi+1)*(wi-1))) if wi > 1 else 1 for wi in w_eigs]))
    return (f_tot/(delta**0.25))*exponential_part

def compute_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    n = (covariance1.shape)[0]//2
    if n == 1:
        return one_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    elif n == 2:
        return two_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    else:
        return n_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
        