import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvals, det, cholesky, slogdet, solve
from .conventions import mean_subsystem, covariance_subsystem, symplectic_matrix, symmetrize_matrix

_ppt_matrix = np.diag([1,1,-1,1])

def compute_logarithmic_negativity(covariance_matrix:npt.NDArray[np.number], subsystems:tuple[int,int] = (1,2)) -> np.number:
    sub_covariance = symmetrize_matrix(covariance_subsystem(covariance_matrix,subsystems))
    omega_matrix = symplectic_matrix(2)
    symplectic_form = 1j*omega_matrix@_ppt_matrix@sub_covariance@_ppt_matrix
    nu_min = np.min(np.abs(eigvals(symplectic_form)))
    log_neg = np.max([0,-np.log(2*nu_min)])
    return log_neg
    
def state_purity(covariance_matrix:npt.NDArray[np.number]) -> np.number:
    n = int((covariance_matrix.shape)[0]/2)
    return 1/np.sqrt(det(2*symmetrize_matrix(covariance_matrix)))

def renyi_two_entropy(covariance_matrix:npt.NDArray[np.number])  -> np.number:
    return -np.log(state_purity(covariance_matrix))

def lambda_matrix(covariance_matrix:npt.NDArray[np.number])  -> npt.NDArray[npt.number]:
    n = int((covariance_matrix.shape)[0]/2)
    omega_matrix = symplectic_matrix(n)
    return symmetrize_matrix(-omega_matrix @ covariance_matrix @ omega_matrix @ covariance_matrix)

def gamma_matrix(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> npt.NDArray[npt.number]:
    n = int((covariance_matrix1.shape)[0]/2)
    omega_matrix = symplectic_matrix(n)
    return omega_matrix @ covariance_matrix1 @ omega_matrix @ covariance_matrix2 - 0.25*np.identity(2*n)

def sigma_matrix(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> npt.NDArray[npt.number]:
    return symmetrize_matrix(covariance_matrix1 + covariance_matrix2)

def _logdet_spd(matrix:npt.NDArray[np.number]) -> np.number:
    logdet_val = np.sum(np.log(np.diagonal(cholesky(matrix))))
    return logdet_val

def fidelity_fixed_parts(mean_covariance_tuple1:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], mean_covariance_tuple2:tuple[npt.NDArray[np.number],npt.NDArray[np.number]]) -> tuple[np.number,np.number]:
    mean1, covariance1 = mean_covariance_tuple1
    mean2, covariance2 = mean_covariance_tuple2
    du = mean2 - mean1
    sigma = sigma_matrix(covariance1,covariance2)
    return (_logdet_spd(sigma),np.real(np.exp(-0.25*du@(solve(sigma,du)))))

def calculate_lambda(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> npt.number:
    lambdas = [lambda_matrix(cov) for cov in [covariance_matrix1, covariance_matrix2]]
    eigens = [np.sqrt(np.abs(eigvals(lamb))) for lamb in lambdas]

    n = int((covariance_matrix1.shape)[0]/2)
    vals = [np.prod(np.real((spectrum-0.5)*(spectrum+0.5))) for spectrum in eigens]
    return np.max([0,(4**n)*vals[0]*vals[1]])

def calculate_gamma(covariance_matrix1:npt.NDArray[np.number],covariance_matrix2:npt.NDArray[np.number])  -> npt.number:
    n = int((covariance_matrix1.shape)[0]/2)
    gamma = gamma_matrix(covariance_matrix1, covariance_matrix2)
    gammaSign, gammaVal = slogdet(gamma)
    return np.max([0,(4**n)*gammaSign*np.exp(gammaVal)])
    

def one_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], mean_covariance_tuple2:tuple[npt.NDArray[np.number],npt.NDArray[np.number]]) -> np.number:
    delta, exponential_part = fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    lambda_value = calculate_lambda(covariance1,covariance2)
    return np.sqrt((np.sqrt(delta+lambda_value)+np.sqrt(lambda_value))/delta)*exponential_part
    
def two_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], mean_covariance_tuple2:tuple[npt.NDArray[np.number],npt.NDArray[np.number]]) -> np.number:
    delta, exponential_part = fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    lambda_value = calculate_lambda(covariance1,covariance2)
    gamma_value = calculate_gamma(covariance1,covariance2)
    return np.sqrt((np.sqrt(gamma_value)+np.sqrt(lambda_value)+np.sqrt(np.sqrt(gamma_value + lambda_value)**2 - delta))/delta)*exponential_part

def compute_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], mean_covariance_tuple2:tuple[npt.NDArray[np.number],npt.NDArray[np.number]]) -> np.number:
    delta, exponential_part = fidelity_exponential(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    n = int((covariance1.shape)[0]/2)
    if n == 1:
        return one_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    elif n == 2:
        return two_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    else:
        omega_matrix = symplectic_matrix(n)
        
        w1, w2 = [-2*1j*cov@omega_matrix for cov in [covariance1, covariance2]]
        w1_w2_inv = solve(w1 + w2,np.identity(2*n))
        w_auxiliary = -  w1_w2_inv @ (np.identity(2*n) + w2@w1) 
        w_eigs = np.real(eigvals(w_auxiliary))
        f_tot = np.prod(np.array([np.sqrt(wi + np.sqrt((wi+1)*(wi-1))) if wi > 1 else 1 for wi in w_eigs]))
        return (f_tot/(delta**0.25))*exponential_part
        
        
    