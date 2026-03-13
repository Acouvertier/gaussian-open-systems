import numpy as np
import numpy.typing as npt

def _x_subsystem(n:int, indices: tuple[int, ...]) -> npt.NDArray[int]:
    valid_indicies = np.array(indices) - 1
    return valid_indicies.astype(int)

def _p_subsystem(n:int, indices: tuple[int, ...]) -> npt.NDArray[int]:
    valid_indicies = np.array(indices) - 1 + n
    return valid_indicies.astype(int)

def _index_list(n:int, indices: tuple[int, ...]) -> npt.NDArray[int]:
    x_idx = _x_subsystem(n,indices)
    p_idx = _p_subsystem(n,indices)
    final_idx = np.append(x_idx,p_idx).astype(int)
    return final_idx

def _selection_matrix(n:int, indices: tuple[int, ...]) -> npt.NDArray[np.number]:
    full_identity = np.identity(2*n)
    final_idx = _index_list(n, indices)
    return full_identity[:,final_idx]

def _rotation_matrix(theta:np.number) -> npt.NDArray[np.number]:
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta),   np.cos(theta)]
    ])

def symplectic_matrix(n:int) -> npt.NDArray[np.number]:
    identity_matrix = np.identity(n)
    w = np.array([
        [0.0 ,1.0],
        [-1.0,0.0]
    ])
    return np.kron(w,identity_matrix)

def mean_subsystem(mean_vector:npt.NDArray[np.number], n:int, indices: tuple[int, ...]) -> npt.NDArray[np.number]:
    final_idx = _index_list(n, indices)
    return mean_vector[final_idx]

def covariance_subsystem(covariance_matrix:npt.NDArray[np.number], n:int, indices: tuple[int, ...]) -> npt.NDArray[np.number]:
    selection_mat = _selection_matrix(n,indices)
    return selection_mat.T @ covariance_matrix @ selection_mat

def compress_mean_covariance(mean_vector:npt.NDArray[np.number], covariance_matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    return np.append(mean_vector, covariance_matrix.flatten())

def extract_mean_covariance(mean_covariance_vector:npt.NDArray[np.number]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    element_count = len(mean_covariance_vector)
    n = int(0.25*(-1 + np.sqrt(1+4*element_count)))
    mean_vector = mean_covariance_vector[0:2*n]
    covariance_matrix = mean_covariance_vector[2*n:]
    return (mean_vector, covariance_matrix.reshape( (2*n,2*n) ) )
