import numpy as np
import numpy.typing as npt
from gaussian_systems.conventions import symmetrize_matrix, index_list, symplectic_matrix, compress_mean_covariance, extract_mean_covariance, _x_subsystem, _p_subsystem
from gaussian_systems.metrics import valid_covariance
from typing import Callable

def _self_energy(n:int, mode_id:int, frequency:float = 0.0) -> npt.NDArray[np.number]:
    blank_matrix = np.zeros((2*n,2*n))
    idx, idp = index_list(n, (mode_id,))
    blank_matrix[idx, idx] = frequency
    blank_matrix[idp, idp] = frequency
    return blank_matrix

def _xixj_term(n:int, subsystem:tuple[int,int], coupling:float) -> npt.NDArray[np.number]:
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2 = _x_subsystem(n,subsystem)
    blank_matrix[idx1,idx2] = coupling
    blank_matrix[idx2,idx1] = coupling
    return blank_matrix

def _pipj_term(n:int,subsystem:tuple[int,int], coupling:float) -> npt.NDArray[np.number]:
    blank_matrix = np.zeros((2*n,2*n))
    idp1, idp2 = _p_subsystem(n,subsystem)
    blank_matrix[idp1,idp2] = coupling
    blank_matrix[idp2,idp1] = coupling
    return blank_matrix

def _xipj_term(n:int,subsystem:tuple[int,int], coupling:float) -> npt.NDArray[np.number]:
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2, idp1, idp2 = index_list(n,subsystem)
    blank_matrix[idx1,idp2] = coupling
    blank_matrix[idp2,idx1] = coupling
    return blank_matrix

"""
interaction_array element (interaction_key:str,subsystem:tuple[int,int],coupling:float)
interation_key: xx, pp, xp
"""
def compile_hamiltonian(n:int,frequency_tuple:tuple[float,...]=(),interaction_array:list[tuple,...]=None) -> npt.NDArray[np.number]:
    
    if frequency_tuple == ():
        unique_frequencies = np.zeros(n)
    elif len(frequency_tuple) < n:
        unique_frequencies = np.append(np.array(frequency_tuple), np.zeros(int(n-len(frequency_tuple)))) 
    else:
        unique_frequencies = np.array(frequency_tuple)[:n]
   
    blank_matrix = np.zeros((2*n,2*n))
    
    for i in range(n):
        blank_matrix += _self_energy(n, int(i+1), unique_frequencies[i])
    
    if interaction_array is None:
        interaction_array = []
    
    for interaction in interaction_array:
        interaction_key, subsystem, coupling = interaction
        if interaction_key.lower() == "xx":
            interaction_matrix = _xixj_term(n, subsystem, coupling)
        elif interaction_key.lower() == "pp":
            interaction_matrix = _pipj_term(n, subsystem, coupling)
        elif interaction_key.lower() == "xp":
            interaction_matrix = _xipj_term(n, subsystem, coupling)
        else:
            print(f"compile_hamiltonian: invalid interaction key provided: {interaction_key}. Skipped.")
            interaction_matrix = 0
        blank_matrix += interaction_matrix

    return blank_matrix

"""
decay_array element (x_p_a:str,mode_id:int,rate:float)
x_p_a: x, p, a, ad
"""
def compile_lindblad(n:int, decay_array:list[tuple,...]=None) -> npt.NDArray[np.number]:
    decay_coefficient_array = np.zeros(2*n, dtype=complex)

    if decay_array is None:
        decay_array = []

    for decay in decay_array:
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
        else:
            print(f"compile_lindblad: invalid decay operator choice: {x_p_a}. Valid choices are x, p, or a. Skipped.")

    return decay_coefficient_array

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


    