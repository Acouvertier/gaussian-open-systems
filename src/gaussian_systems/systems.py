import numpy as np
import numpy.typing as npt
from gaussian_systems.conventions import symmetrize_matrix, symplectic_matrix, _index_list, _compress_mean_covariance, _extract_mean_covariance, _x_subsystem, _p_subsystem, _valid_covariance_matrix, _valid_mode_number, _valid_indices, _valid_square_matrix
from scipy.linalg import eigvals, issymmetric
from numbers import Real, Integral, Complex
from typing import Callable

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

def _mutate_frequency_array(n:Integral, frequency_array:npt.NDArray[Real,...] | None = None) -> npt.NDArray[np.float64]:
    _valid_mode_number(n)
    if frequency_array is None:
        return np.zeros(n)
    else:
        _valid_frequency_array(frequency_array)
        if n >= len(frequency_array):
            missing_modes_count = int(n-len(frequency_array))
            warnings.warn(f"Warning: {n}-mode system but only {len(frequency_array)} thermal occupations provided. Assuming 0 frequencies for modes {len(frequency_array)+1}-{n}.")
            return np.append(frequency_array,np.zeros(missing_modes_count))
        else:
            raise ValueError(f"mode frequencies contains more values than modes. Got {len(frequency_array)} values, expected {n} values.")

def _self_energies(n:Integral, frequency_array:npt.NDArray[Real,...] = None) -> npt.NDArray[np.float64]:

    _valid_mode_number(n)
    try:
        frequency_array_final = _mutate_frequency_array(n,frequency_array)
        return np.diag(np.append(frequency_array_final, frequency_array_final))
    except Exception as e:
        print(e)
        raise ValueError(f"unable to mutate frequency array to match system dimensions. Got dimenions {n} and frequencies {frequency_array}") 
        

def _xixj_term(n:Integral, subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:

    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2 = _x_subsystem(n,subsystem)
    blank_matrix[idx1,idx2] += coupling
    blank_matrix[idx2,idx1] += coupling
    return blank_matrix

def _pipj_term(n:Integral,subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:
   
    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idp1, idp2 = _p_subsystem(n,subsystem)
    blank_matrix[idp1,idp2] += coupling
    blank_matrix[idp2,idp1] += coupling
    return blank_matrix

def _xipj_term(n:Integral,subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:

    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2, idp1, idp2 = index_list(n,subsystem)
    blank_matrix[idx1,idp2] += coupling
    blank_matrix[idp2,idx1] += coupling
    return blank_matrix

"""
interaction_array element (interaction_key:str,subsystem:tuple[int,int],coupling:float)
interation_key: xx, pp, xp
"""

def _valid_hamiltonian_matrix(ham:npt.NDArray[Real]):
    _valid_covariance_matrix(ham)

def _valid_lindblad_gram_matrix(M: npt.NDArray[complex], atol=1e-8):
    _valid_square_matrix(M)

    if M.shape[0] % 2 != 0:
        raise ValueError("must be even-dimensional")

    M_herm = symmetrize_matrix(M)
    eigs = np.linalg.eigvalsh(M_herm)

    if np.any(eigs < -atol):
        raise ValueError("Hermitian part must be positive semidefinite")

def _valid_system(ham, M):
    _valid_hamiltonian_matrix(ham)
    _valid_lindblad_gram_matrix(M)
    

class GaussianCVSystem:
    def __init__(self, hamiltonian_matrix:npt.NDArray[np.float64],lindblad_matrix:npt.NDArray[np.float64]):
        _valid_system(hamiltonian_matrix, lindblad_matrix)
        self._n = (hamiltonian_matrix.shape[0]) // 2
        self._hamiltonian_matrix = hamiltonian_matrix
        self._lindblad_matrix = lindblad_matrix

    @classmethod
    def free_evolution(cls,n:Integral, frequency_array:npt.NDArray[np.float64]):
        ham = _self_energies(n,frequency_array)
        lind = np.zeros((2*n,2*n), dtype=complex)
        return cls(ham,lind)

    @property
    def hamiltonian_matrix(self) -> npt.NDArray[np.float64]:
        return self._hamiltonian_matrix.copy()

    @property
    def lindblad_matrix(self) -> npt.NDArray[np.float64]:
        return self._lindblad_matrix.copy()

    def position_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix += _xixj_term(self._n, subsystem, coupling)
        return self

    def beamsplitter_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix += _xixj_term(self._n, subsystem, coupling)
        self._hamiltonian_matrix += _pipj_term(self._n, subsystem, coupling)
        return self

    def squeezer_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix += _xixj_term(self._n, subsystem, coupling)
        self._hamiltonian_matrix -= _pipj_term(self._n, subsystem, coupling)
        return self

    def position_difference_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix -= 2*_xixj_term(self._n, subsystem, coupling)
        self._hamiltonian_matrix += _xixj_term(self._n, (subsystem[0],subsystem[0]), coupling)
        self._hamiltonian_matrix += _xixj_term(self._n, (subsystem[1],subsystem[1]), coupling)
        return self

    def multi_position_dissipator(self, subsystem:tuple[Integral,...], decay:Complex):
        _valid_indices(self._n, subsystem)
        decay_array = [("x", idx, decay) for idx in subsystem]
        self._lindblad_matrix += compile_single_lindblad_matrix(self._n, decay_array)
        return self

    def multi_annihilation_dissipator(self, subsystem:tuple[Integral,...], decay:Complex):
        _valid_indices(self._n, subsystem)
        decay_array = [("a", idx, decay) for idx in subsystem]
        self._lindblad_matrix += compile_single_lindblad_matrix(self._n, decay_array)
        return self

    def multi_thermal_dissipator(self, subsystem:tuple[Integral,...], decay:Complex, thermal_occupation:Real):
        if not isinstance(thermal_occupation,Real):
            raise TypeError(f"thermal occupation must be real. Got {type(thermal_occupation)}")
        if thermal_occupation < 0:
            raise ValueError(f"thermal occupation must be non-negative. Got {thermal_occupation}")
        _valid_indices(self._n, subsystem)
        decay_array = [("a", idx, decay*(thermal_occupation+1)) for idx in subsystem]
        thermalizing_array = [("ad", idx, decay*(thermal_occupation)) for idx in subsystem]
        self._lindblad_matrix += compile_single_lindblad_matrix(self._n, decay_array + thermalizing_array)
        return self

"""
decay_array element (x_p_a:str,mode_id:int,rate:float)
x_p_a: x, p, a, ad
"""

def _valid_decay_element(decay_element:tuple[str,Integral,Real]) -> None:
    if not isinstance(decay_element, tuple):
        raise TypeError(f"Dissipation generation requires jump operator information contained in a tuple. Got {type(decay_element)}")
    if len(decay_element) != 3:
        raise ValueError(f"Jump operator is defined by three parts. Got {decay_element}")
    decay_str, decay_int, decay_complex = decay_element
    if (not isinstance(decay_str, str)) or (decay_str not in ["a","x","p"]):
        raise ValueError(f"First part of jump definition must be string representing type. Got {decay_str}")
    if (not isinstance(decay_int, Integral)) or (decay_int <= 0):
        raise ValueError(f"Second part of jump definition must be natural number representing affected mode. Got {decay_int}")
    if (not isinstance(decay_complex, Real)):
        raise TypeError(f"Tird part of jump definition must be the real-valued decay rate. Got {decay_complex}")

def compile_single_lindblad_matrix(n:int, decay_array:list[tuple,...]=None) -> npt.NDArray[np.number]:
    decay_coefficient_array = np.zeros(2*n, dtype=complex)

    if decay_array is None:
        decay_array = []

    for decay in decay_array:
        _valid_decay_element(decay)
        x_p_a, mode_id, rate = decay
        root_rate = np.sqrt(rate)
        if x_p_a.lower() == "x":
            idx = _x_subsystem(n,(mode_id,))[0]
            decay_coefficient_array[idx] += root_rate
        elif x_p_a.lower() == "p":
            idp = _p_subsystem(n,(mode_id,))[0]
            decay_coefficient_array[idp] += root_rate
        elif x_p_a.lower() == "a":
            idx, idp = index_list(n,(mode_id,))
            decay_coefficient_array[idx] += root_rate/np.sqrt(2)
            decay_coefficient_array[idp] += 1j*root_rate/np.sqrt(2)
        elif x_p_a.lower() == "ad":
            idx, idp = index_list(n,(mode_id,))
            decay_coefficient_array[idx] += root_rate/np.sqrt(2)
            decay_coefficient_array[idp] -= 1j*root_rate/np.sqrt(2)

    return np.outer(np.conjugate(decay_coefficient_array),decay_coefficient_array)

def generate_drift_diffusion(n:int, frequency_tuple:tuple[float,...]=(),interaction_array:list[tuple,...]=None, decays_tuple:tuple[list[tuple,...],...]=None) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    
    hamiltonian_matrix = compile_hamiltonian(n,frequency_tuple,interaction_array)
    lindblad_matrix = np.zeros((2*n,2*n),dtype='complex128')
    
    if decays_tuple is None:
        decays_tuple = ()
    
    for decay_array in decays_tuple:
        lindblad_array = compile_lindblad(n, decay_array)
        lindblad_matrix += np.outer(np.conjugate(lindblad_array),lindblad_array)
    
    omega_matrix = symplectic_matrix(n)

    drift = omega_matrix @ (hamiltonian_matrix + np.imag(lindblad_matrix))
    diffusion = omega_matrix @ np.real(lindblad_matrix) @ (omega_matrix.T)

    return (drift, diffusion)

def heisenberg_eom(n:int, frequency_tuple:tuple[float,...]=(),interaction_array:list[tuple,...]=None,decays_tuple:tuple[list[tuple,...],...]=None) -> Callable[[float,npt.NDArray[np.number]],npt.NDArray[np.number]]:
    drift, diffusion = generate_drift_diffusion(n, frequency_tuple,interaction_array, decays_tuple)
    def dxdt(t:float,mean_covariance:npt.NDArray):
        x0, c0 = extract_mean_covariance(mean_covariance)
        c0_sym = symmetrize_matrix(c0)
        dx = drift @ x0
        dc = drift @ c0_sym + c0_sym @ drift.T + diffusion
        return compress_mean_covariance(dx,dc)

    return dxdt


def physical_covariance_matrix(covariance_matrix:npt.NDArray[np.float64], tol:np.number = 1e-8) -> bool:
    _valid_covariance_matrix(covariance_matrix)
    if not isinstance(tol, Real):
        raise TypeError(f"tol must be real-valued, got {type(tol)}.")
    if tol < 0:
        raise ValueError(f"tol must be nonnegative, got {tol}")
        
    n = (covariance_matrix.shape[0])//2
        
    cov_sym = symmetrize_matrix(covariance_matrix)
    
    eigs = np.real(eigvals(cov_sym + 0.5j*symplectic_matrix(n)))
    return np.all(eigs >= -1*tol)


    