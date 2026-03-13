import numpy as np
import numpy.typing as npt
from .conventions import _rotation_matrix

def thermal_vacuum_covariance(n:int, nbars:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    diagonal_elements = nbars + 0.5
    return np.diag(np.append(diagonal_elements, diagonal_elements))

def thermal_vacuum_mean(n:int) -> npt.NDArray[np.number]:
    return np.zeros(2*n)

def apply_1_mode_displacement(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], displacement: complex, mode_id:int) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    mean_vector, covariance_matrix = mean_vector_variance_matrix_tuple
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
    rot_matrix = _rotation_matrix(squeeze_angle)
    return (rot_matrix @ np.diag([np.exp(-squeeze_magnitude),np.exp(squeeze_magnitude)])) @ rot_matrix.T

def apply_1_mode_squeeze_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.number],npt.NDArray[np.number]], squeeze_magnitude_angle_tuple:tuple[np.number,np.number], mode_id:int) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    mean_vector, covariance_matrix = mean_vector_variance_matrix_tuple
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    n = int(len(mean_vector)/2)
    unit_vector = (np.indentity(n))[mode_id-1]
    single_squeeze_matrix = single_mode_squeeze_matrix(squeeze_magnitude, squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) + np.kron(single_squeeze_matrix - np.identity(2), np.outer(unit_vector, unit_vector))
    return (n_mode_squeeze_unitary @ mean_vector, (n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T)
    