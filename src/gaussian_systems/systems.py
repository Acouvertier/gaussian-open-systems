from .conventions import symmetrize_matrix, symplectic_matrix, index_list, compress_mean_covariance, extract_mean_covariance, mean_subsystem, covariance_subsystem, physical_covariance_matrix
from ._validation import  _valid_mode_number, _valid_frequency_array, _valid_term_inputs, _valid_decay_element, _valid_system, _valid_indices, _valid_single_pole_input, _valid_covariance_matrix, _valid_t_eval
from .initial_state import GaussianCVState
from .solution import GaussianSolution

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.integrate import quad_vec
from numbers import Real, Integral, Complex
import warnings

def _mutate_frequency_array(n:Integral, frequency_array:npt.NDArray[Real] | None = None) -> npt.NDArray[np.float64]:
    _valid_mode_number(n)
    if frequency_array is None:
        return np.zeros(n)
    else:
        _valid_frequency_array(frequency_array)
        if n > len(frequency_array):
            missing_modes_count = int(n-len(frequency_array))
            warnings.warn(f"Warning: {n}-mode system but only {len(frequency_array)} frequencies provided. Assuming 0 frequency for modes {len(frequency_array)+1}-{n}.")
            return np.append(frequency_array,np.zeros(missing_modes_count))
        elif n == len(frequency_array):
            return frequency_array
        else:
            raise ValueError(f"mode frequencies contains more values than modes. Got {len(frequency_array)} values, expected {n} values.")

def _self_energies(n:Integral, frequency_array:npt.NDArray[Real] | None = None) -> npt.NDArray[np.float64]:

    _valid_mode_number(n)
    try:
        frequency_array_final = _mutate_frequency_array(n,frequency_array)
        return np.diag(np.append(frequency_array_final, frequency_array_final))
    except Exception as e:
        print(e)
        raise ValueError(f"unable to mutate frequency array to match system dimensions. Got dimensions {n} and frequencies {frequency_array}") 
        

def _xixj_term(n:Integral, subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:

    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2 = index_list(n,subsystem)[:2]
    blank_matrix[idx1,idx2] += coupling
    blank_matrix[idx2,idx1] += coupling
    return blank_matrix

def _pipj_term(n:Integral,subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:
   
    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idp1, idp2 = index_list(n,subsystem)[2:]
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

def _embedding_matrix(n:Integral) ->npt.NDArray[np.float64]:
    return np.kron(np.identity(2),np.eye(n+1, n))

"""
decay_array element (x_p_a:str,mode_id:int,rate:float)
x_p_a: x, p, a, ad
"""

def _compile_single_lindblad_matrix(n:int, decay_array:list[tuple,...]=None) -> npt.NDArray[np.number]:
    decay_coefficient_array = np.zeros(2*n, dtype=complex)

    if decay_array is None:
        decay_array = []

    for decay in decay_array:
        _valid_decay_element(decay)
        x_p_a, mode_id, rate = decay
        root_rate = np.sqrt(rate)
        if x_p_a.lower() == "x":
            idx = index_list(n,(mode_id,))[0]
            decay_coefficient_array[idx] += root_rate
        elif x_p_a.lower() == "p":
            idp = index_list(n,(mode_id,))[1]
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

class GaussianCVSystem:
    def __init__(self, hamiltonian_matrix:npt.NDArray[np.float64],lindblad_matrix:npt.NDArray[np.float64]):
        _valid_system(hamiltonian_matrix, lindblad_matrix)
        self._n = (hamiltonian_matrix.shape[0]) // 2
        self._hamiltonian_matrix = hamiltonian_matrix
        self._lindblad_matrix = lindblad_matrix

    @classmethod
    def free_evolution(cls,n:Integral, frequency_array:npt.NDArray[np.float64]| None = None):
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

    def momentum_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix += _pipj_term(self._n, subsystem, coupling)
        return self

    def position_i_momentum_j_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        self._hamiltonian_matrix += _xipj_term(self._n, subsystem, coupling)
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

    def multi_position_dissipator(self, subsystem:tuple[Integral,...], decay:Real):
        _valid_indices(self._n, subsystem)
        decay_array = [("x", idx, decay) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self._n, decay_array)
        return self

    def multi_annihilation_dissipator(self, subsystem:tuple[Integral,...], decay:Real):
        _valid_indices(self._n, subsystem)
        decay_array = [("a", idx, decay) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self._n, decay_array)
        return self

    def multi_thermal_dissipator(self, subsystem:tuple[Integral,...], decay:Real, thermal_occupation:Real):
        if not isinstance(thermal_occupation,Real):
            raise TypeError(f"thermal occupation must be real. Got {type(thermal_occupation)}")
        if thermal_occupation < 0:
            raise ValueError(f"thermal occupation must be non-negative. Got {thermal_occupation}")
        _valid_indices(self._n, subsystem)
        decay_array = [("a", idx, decay*(thermal_occupation+1)) for idx in subsystem]
        thermalizing_array = [("ad", idx, decay*(thermal_occupation)) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self._n, decay_array + thermalizing_array)
        return self

    def generate_drift_and_diffusion(self):
        omega_matrix = symplectic_matrix(self._n)

        drift = np.real(omega_matrix @ (self.hamiltonian_matrix + np.imag(self.lindblad_matrix)))
        diffusion = np.real(omega_matrix @ np.real(self.lindblad_matrix) @ (omega_matrix.T))

        return (drift, diffusion)

    def copy_system(self):
        return GaussianCVSystem(self._hamiltonian_matrix.copy(), self._lindblad_matrix.copy())

    def gaussian_channel(self):
        m = int(2*self._n)
        m2 = int(4*self._n**2)
        I = np.identity(m)
        A, D = self.generate_drift_and_diffusion()
        K, D_vec = np.kron(A,I) + np.kron(I, A), D.flatten() 
        
        A_covariance = np.append(
            np.append(
                K.T,
                np.reshape(D_vec,(1,m2)), 
                axis=0).T,
            np.zeros((1,int(m2 + 1))),
            axis=0)
        return (A, A_covariance)
        

    def evolve_state(self, state:GaussianCVState, t_eval:npt.NDArray[np.float64]):
        _valid_state_system_pair(state, self)
        _valid_t_eval(t_eval)
        x0, c0 = state.mean_vector, state.covariance_matrix
        c0_vec = np.append(c0.flatten(),np.array([1.0]))
        A, A_covariance = self.gaussian_channel()
        
        means = []
        covariances = []

        for t in t_eval:
            mean, covariance = _apply_gaussian_channel(A, A_covariance, x0, c0_vec, t)
            if not physical_covariance_matrix(covariance):
                raise ValueError(f"Nonphysical covariance at t={t}. Got {covariance}")
            means.append(mean)
            covariances.append(covariance)
        
        return GaussianSolution(t_eval,means,covariances)

def _apply_gaussian_channel(mean_drift:npt.NDArray[np.float64], covariance_drift:npt.NDArray[np.float64], mean_vector:npt.NDArray[np.float64], covariance_vector:npt.NDArray[np.float64], t:Real) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    n = len(mean_vector)//2
    St = expm(t*mean_drift)
    Mt = expm(t*covariance_drift)
    
    evolved_mean = St @ mean_vector
    evolved_covariance_vector = Mt @ covariance_vector
    evolved_covariance = (evolved_covariance_vector[:-1]).reshape((2*n,2*n))
    return (evolved_mean, symmetrize_matrix(evolved_covariance))

def _valid_state_system_pair(state:GaussianCVState, system:GaussianCVSystem) -> None:
    if not isinstance(state, GaussianCVState):
        raise TypeError(f"embedding requires a valid state object. Got {type(state)}")
    if not isinstance(system, GaussianCVSystem):
        raise TypeError(f"embedding requires a valid system object. Got {type(system)}")
    if system._n != state.n:
        raise ValueError(f"system and state objects must represent same mode number. System is {system._n}-modes, while State is {state.n}")

def single_pole_ou_embedding(state:GaussianCVState, system:GaussianCVSystem, subsystem:tuple[Integral,...], coupling_types:tuple[str,...], memory_rate:Real, env_freq:Real, decay_rate:Real, thermal_occupation:Real):
    _valid_state_system_pair(state, system)
    n = state.n
    _valid_single_pole_input(n, subsystem, coupling_types, memory_rate, env_freq, thermal_occupation)
    
    embedding_matrix = _embedding_matrix(n)
    mean_vector, covariance_matrix = state.mean_vector, state.covariance_matrix
    hamiltonian_matrix, lindblad_matrix = system.hamiltonian_matrix, system.lindblad_matrix

    embedded_mean = embedding_matrix @ mean_vector
    embedded_covariance = symmetrize_matrix((embedding_matrix @ covariance_matrix) @ embedding_matrix.T)

    embedded_hamiltonian = symmetrize_matrix((embedding_matrix @ hamiltonian_matrix) @ embedding_matrix.T)
    embedded_lindblad = symmetrize_matrix((embedding_matrix @ lindblad_matrix) @ embedding_matrix.T)

    embedded_state = GaussianCVState(embedded_mean, embedded_covariance)
    embedded_state.single_mode_thermal_occupation(thermal_occupation,n+1)

    embedded_system = GaussianCVSystem(embedded_hamiltonian, embedded_lindblad)
    pseudo_coupling = np.sqrt(decay_rate*memory_rate/2)
    pseudo_decay = 2*memory_rate
    embedded_system.multi_thermal_dissipator((n+1,), pseudo_decay, thermal_occupation)
    
    for idx in range(len(coupling_types)):
        subsystem_id = subsystem[idx]
        coupling_type = coupling_types[idx]
        
        if coupling_type == "position":
            embedded_system.position_coupling((subsystem_id,n+1), pseudo_coupling)
        elif coupling_type == "momentum":
            embedded_system.position_i_momentum_j_coupling((n+1,subsystem_id), pseudo_coupling)
        elif coupling_type == "annihilation":
            embedded_system.beamsplitter_coupling((subsystem_id,n+1), pseudo_coupling)
        else:
            raise ValueError(f"individual pseudo-system couplings must be 'position', 'momentum', or 'annihilation' got {coupling_type}")
        
    pseudo_array = np.zeros(n+1)
    pseudo_array[n] = env_freq
    embedded_system._hamiltonian_matrix += _self_energies(n+1, pseudo_array)
        
    return embedded_state, embedded_system


    