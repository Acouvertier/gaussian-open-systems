import numpy as np
import numpy.typing as npt
from .conventions import rotation_matrix, index_list

def thermal_vacuum_covariance(n:int, nbars:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    if n > len(nbars):
        nbars_full = np.append(nbars,np.zeros(int(n-len(nbars))))
    elif n < len(nbars):
        nbars_full = nbars[:n+1]
    else: nbars_full = nbars
    diagonal_elements = nbars_full + 0.5
    return np.diag(np.append(diagonal_elements, diagonal_elements))

def thermal_vacuum_mean(n:int) -> npt.NDArray[np.number]:
    return np.zeros(2*n)

def apply_1_mode_displacement(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]],
                              displacement: complex, mode_id:int) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    n = int(len(mean_vector)/2)
    x_disp = np.sqrt(2) * np.real(displacement)
    p_disp = np.sqrt(2) * np.imag(displacement)
    idx, idp = (mode_id - 1, mode_id - 1 + n) 
    x_shift_vector = np.zeros(2*n)
    p_shift_vector = np.zeros(2*n)

    x_shift_vector[idx] = x_disp
    p_shift_vector[idp] = p_disp
        
    return (mean_vector + x_shift_vector + p_shift_vector, covariance_matrix)

def single_mode_squeeze_matrix(squeeze_magnitude:np.number, squeeze_angle:np.number) -> npt.NDArray[np.number]:
    rot_matrix = rotation_matrix(squeeze_angle)
    return (rot_matrix @ np.diag([np.exp(-squeeze_magnitude),np.exp(squeeze_magnitude)])) @ rot_matrix.T

def apply_1_mode_squeeze_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]],
                                 squeeze_magnitude_angle_tuple:tuple[np.number,np.number], mode_id:int) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    n = int(len(mean_vector)/2)
    unit_vector = (np.identity(n))[mode_id-1]
    single_squeeze_matrix = single_mode_squeeze_matrix(squeeze_magnitude, squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) + np.kron(single_squeeze_matrix - np.identity(2), np.outer(unit_vector, unit_vector))
    return (n_mode_squeeze_unitary @ mean_vector, (n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T)


def two_mode_mixing_matrix(coupling_magnitude:np.number,coupling_angle:np.number) -> npt.NDArray[np.number]:
    unitary_exponential_matrix = np.array([
        [np.cos(coupling_magnitude), np.exp(1j*coupling_angle)*np.sin(coupling_magnitude)],
        [-np.exp(-1j*coupling_angle)*np.sin(coupling_magnitude), np.cos(coupling_magnitude)]
    ])
    return np.kron(np.identity(2),np.real(unitary_exponential_matrix)) + np.kron(np.array([[0,-1],[1,0]]),np.imag(unitary_exponential_matrix))

def apply_2_mode_mixing_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]],
                                 coupling_magnitude_angle_tuple:tuple[np.number,np.number], mode_ids:tuple[int,int]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    mean_vector, covariance_matrix = mean_vector_variance_matrix_tuple
    coupling_magnitude, coupling_angle = coupling_magnitude_angle_tuple
    n = int(len(mean_vector)/2)
    transformed_idx = index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_mix_matrix = two_mode_mixing_matrix(coupling_magnitude,coupling_angle)
    n_mode_mix_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_mix_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_mix_unitary @ mean_vector, (n_mode_mix_unitary @ covariance_matrix) @ n_mode_mix_unitary.T)

def two_mode_squeezing_matrix(squeeze_magnitude:np.number,squeeze_angle:np.number) -> npt.NDArray[np.number]:
    mu = np.cosh(squeeze_magnitude)
    nu_r = np.sinh(squeeze_magnitude)*np.cos(squeeze_angle)
    nu_i = np.sinh(squeeze_magnitude)*np.sin(squeeze_angle)
    return np.array([
        [mu,nu_r,0,nu_i],
        [nu_r,mu,nu_i,0],
        [0,nu_i,mu,-nu_r],
        [nu_i,0,-nu_r,mu]
    ])

def apply_2_mode_squeezing_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]],
                                 squeeze_magnitude_angle_tuple:tuple[np.number,np.number], mode_ids:tuple[int,int]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    mean, covariance = mean_vector_covariance_matrix_tuple
    n = len(mean)/2
    transformed_idx = index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_squeeze_matrix = two_mode_squeezing_matrix(squeeze_magnitude,squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_squeeze_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_msqueeze_unitary @ mean_vector, (n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T)
    
    
    
    
    