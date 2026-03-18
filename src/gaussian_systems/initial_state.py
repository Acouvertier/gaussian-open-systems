import numpy as np
import numpy.typing as npt
from .conventions import symmetrize_matrix, rotation_matrix, _index_list, _compress_mean_covariance, _valid_mode_number, _valid_mean_covariance, _valid_indices
from numbers import Real, Integral, Complex
import warnings

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

"""Public"""

def thermal_vacuum_covariance(n: Integral, nbars:npt.NDArray[np.float64]=None) -> npt.NDArray[np.float64]:
    _valid_mode_number(n)
    
    if nbars is None:
        nbars_full = np.zeros(n)
    else: 
        _valid_nbars_array(nbars)
        if n >= len(nbars):
            missing_modes_count = int(n-len(nbars))
            warnings.warn(f"Warning: {n}-mode system but only {len(nbars)} thermal occupations provided. Assuming 0 temperature for modes {len(nbars)+1}-{n}.")
            nbars_full = np.append(nbars,np.zeros(missing_modes_count))
        else: 
            raise ValueError(f"thermal occupation contains more values than modes. Got {len(nbars)} values, expected {n} values.")
    
    diagonal_elements = nbars_full + 0.5
    return np.diag(np.append(diagonal_elements, diagonal_elements))

def thermal_vacuum_mean(n:int) -> npt.NDArray[np.float64]:
    _valid_mode_number(n)
    return np.zeros(2*n)

def apply_1_mode_displacement(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                              displacement:Complex, mode_id:Integral) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    
    _valid_mean_vector_covariance_matrix_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,(mode_id,))

    if not isinstance(displacement,Complex):
        raise TypeError(f"single mode displacement must be complex-valued. Got {type(displacement)}")
    
    x_disp = np.sqrt(2) * np.real(displacement)
    p_disp = np.sqrt(2) * np.imag(displacement)

    idx, idp = _index_list(n,(mode_id,))
    
    x_shift_vector = np.zeros(2*n)
    p_shift_vector = np.zeros(2*n)
    
    x_shift_vector[idx] = x_disp
    p_shift_vector[idp] = p_disp
            
    return (mean_vector + x_shift_vector + p_shift_vector, covariance_matrix)

def single_mode_squeeze_matrix(squeeze_magnitude:Real, squeeze_angle:Real) -> npt.NDArray[np.float64]:
    _valid_parameter_pair(squeeze_magnitude, squeeze_angle)
    
    rot_matrix = rotation_matrix(squeeze_angle/2)
    return (rot_matrix @ np.diag([np.exp(-squeeze_magnitude),np.exp(squeeze_magnitude)])) @ rot_matrix.T

def apply_1_mode_squeeze_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                                 squeeze_magnitude_angle_tuple:tuple[Real,Real], mode_id:Integral) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    
    _valid_mean_vector_covariance_matrix_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple

    _valid_parameter_tuple(squeeze_magnitude_angle_tuple)
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,(mode_id,))
    
    transformed_idx = _index_list(n,(mode_id,))
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    
    single_squeeze_matrix = single_mode_squeeze_matrix(squeeze_magnitude, squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) +  (selection_matrix.T) @ (single_squeeze_matrix - np.identity(2)) @ selection_matrix
    
    return (n_mode_squeeze_unitary @ mean_vector, symmetrize_matrix((n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T))


def two_mode_mixing_matrix(coupling_magnitude:Real,coupling_angle:Real) -> npt.NDArray[np.float64]:
    _valid_parameter_pair(coupling_magnitude, coupling_angle)
    unitary_exponential_matrix = np.array([
        [np.cos(coupling_magnitude), np.exp(1j*coupling_angle)*np.sin(coupling_magnitude)],
        [-np.exp(-1j*coupling_angle)*np.sin(coupling_magnitude), np.cos(coupling_magnitude)]
    ])
    return np.kron(np.identity(2),np.real(unitary_exponential_matrix)) + np.kron(np.array([[0,-1],[1,0]]),np.imag(unitary_exponential_matrix))

def apply_2_mode_mix_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                                 coupling_magnitude_angle_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

    _valid_mean_vector_covariance_matrix_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple

    _valid_parameter_tuple(coupling_magnitude_angle_tuple)
    coupling_magnitude, coupling_angle = coupling_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,mode_ids)
    
    transformed_idx = _index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_mix_matrix = two_mode_mixing_matrix(coupling_magnitude,coupling_angle)
    n_mode_mix_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_mix_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_mix_unitary @ mean_vector, symmetrize_matrix((n_mode_mix_unitary @ covariance_matrix) @ n_mode_mix_unitary.T))

def two_mode_squeezing_matrix(squeeze_magnitude:Real,squeeze_angle:Real) -> npt.NDArray[np.float64]:
    _valid_parameter_pair(squeeze_magnitude, squeeze_angle)
    mu = np.cosh(squeeze_magnitude)
    nu_r = np.sinh(squeeze_magnitude)*np.cos(squeeze_angle)
    nu_i = np.sinh(squeeze_magnitude)*np.sin(squeeze_angle)
    return np.array([
        [mu,nu_r,0,nu_i],
        [nu_r,mu,nu_i,0],
        [0,nu_i,mu,-nu_r],
        [nu_i,0,-nu_r,mu]
    ])

def apply_2_mode_squeeze_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                                 squeeze_magnitude_angle_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

    _valid_mean_vector_covariance_matrix_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    
    _valid_parameter_tuple(squeeze_magnitude_angle_tuple)
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,mode_ids)
    
    transformed_idx = _index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_squeeze_matrix = two_mode_squeezing_matrix(squeeze_magnitude,squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_squeeze_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_squeeze_unitary @ mean_vector, symmetrize_matrix((n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T))

class GaussianCVState:
    def __init__(self, mean_vector: npt.NDArray[np.float64], covariance_matrix: npt.NDArray[np.float64]):
        _valid_mean_covariance(mean_vector, covariance_matrix)
        self.n = len(mean_vector) // 2
        self._mean_vector = mean_vector.copy()
        self._covariance_matrix = covariance_matrix.copy()

    @classmethod
    def vacuum(cls, n: Integral):
        return cls(thermal_vacuum_mean(n), thermal_vacuum_covariance(n))

    @classmethod
    def thermal(cls, n: Integral, nbars: npt.NDArray[np.float64] | None = None):
        return cls(thermal_vacuum_mean(n), thermal_vacuum_covariance(n, nbars))

    @property
    def mean_vector(self) -> npt.NDArray[np.float64]:
        return self._mean_vector.copy()

    @property
    def covariance_matrix(self) -> npt.NDArray[np.float64]:
        return self._covariance_matrix.copy()

    def single_mode_displacement(self, displacement:Complex, mode_id:Integral):
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_1_mode_displacement((m0,c0), displacement, mode_id)
        return self
    
    def single_mode_squeeze(self, squeeze_tuple:tuple[Real,Real], mode_id:Integral):
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_1_mode_squeeze_unitary((m0,c0), squeeze_tuple, mode_id)
        return self
    
    def two_mode_mix(self, coupling_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]):
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_2_mode_mix_unitary((m0,c0), coupling_tuple, mode_ids)
        return self

    def two_mode_squeeze(self, squeeze_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]):
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_2_mode_squeeze_unitary((m0,c0), squeeze_tuple, mode_ids)
        return self

    def single_mode_thermal_occupation(self, nbar:Real, mode_id:Integral):
        if not isinstance(mode_id,Integral):
            raise TypeError(f"provided mode index must be integer valued. Got {type(mode_id)}")
        _valid_indices(self.n,(mode_id,))
        idx, idp = _index_list(self.n,(mode_id,))

        self._covariance_matrix[idx, idx] = nbar + 0.5
        self._covariance_matrix[idp, idp] = nbar + 0.5
        return self

    def state_to_vector(self) -> npt.NDArray[np.float64]:
        m0, c0 = self._mean_vector, self._covariance_matrix
        return _compress_mean_covariance(m0, c0)

    def copy_state(self):
        return GaussianCVState(self._mean_vector.copy(), self._covariance_matrix.copy())

    
    
    
    