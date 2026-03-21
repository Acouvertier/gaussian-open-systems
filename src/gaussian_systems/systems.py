from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.integrate import quad_vec
from numbers import Real, Integral, Complex
import warnings

from .conventions import symmetrize_matrix, symplectic_matrix, index_list, compress_mean_covariance, extract_mean_covariance, mean_subsystem, covariance_subsystem, is_physical_covariance_matrix, require_physical_covariance

from ._validation import  _valid_mode_number, _valid_frequency_array, _valid_term_inputs, _valid_decay_element, _valid_system, _valid_indices, _valid_single_pole_input, _valid_covariance_matrix, _valid_t_eval, _require_nonnegative_real_scalar

from .initial_state import GaussianCVState
from .solution import GaussianSolution

def _normalize_frequency_array(n:Integral, frequency_array:npt.NDArray[Real] | None = None) -> npt.NDArray[np.float64]:
    """
    Validate and normalize a mode-frequency array to length n.

    This function ensures that the frequency specification for an n-mode
    system is represented as a length-n array. If no frequency array is
    provided, all mode frequencies are set to zero. If fewer than n
    frequencies are provided, the remaining entries are padded with zeros.
    If exactly n frequencies are provided, the array is returned as a
    float64 NumPy array.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    frequency_array : numpy.ndarray or None, optional
        A one-dimensional array of mode frequencies. Entries must be real
        and finite. If ``None``, all frequencies are set to zero.

    Returns
    -------
    numpy.ndarray of shape (n,)
        A float64 array of mode frequencies.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or if ``frequency_array`` is not a valid
        NumPy array when provided.
    ValueError
        If ``n`` is not strictly positive, if ``frequency_array`` contains
        invalid values, or if more than ``n`` frequencies are supplied.

    Warns
    -----
    UserWarning
        If fewer than ``n`` frequencies are provided. Missing entries are
        set to zero.

    Notes
    -----
    No sign constraint is imposed on the frequencies. Zero-padded entries
    are interpreted as resonant modes in a rotated-frame convention.
    """
    _valid_mode_number(n)
    if frequency_array is None:
        return np.zeros(n)
    else:
        _valid_frequency_array(frequency_array)
        if n > len(frequency_array):
            missing_modes_count = int(n-len(frequency_array))
            warnings.warn(f"Warning: {n}-mode system but only {len(frequency_array)} frequencies provided. Assuming a rotated frame such that modes {len(frequency_array)+1}-{n} are resonant.")
            return np.concatenate((frequency_array,np.zeros(missing_modes_count)),axis=0)
        elif n == len(frequency_array):
            return np.array(frequency_array,dtype=np.float64)
        else:
            raise ValueError(f"mode frequencies contains more values than modes. Got {len(frequency_array)} values, expected {n} values.")

def _self_energies(n:Integral, frequency_array:npt.NDArray[Real] | None = None) -> npt.NDArray[np.float64]:
    """
    Construct the diagonal self-energy matrix for an n-mode system.

    This function returns the diagonal quadratic contribution associated
    with the mode frequencies in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If the normalized mode frequencies are
        (ω_1, ..., ω_n),
    the resulting matrix is

        diag(ω_1, ..., ω_n, ω_1, ..., ω_n).

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    frequency_array : numpy.ndarray or None, optional
        A one-dimensional array of mode frequencies. If fewer than ``n``
        entries are provided, missing values are filled with zeros according
        to the rotated-frame convention used by
        ``_normalize_frequency_array``. If ``None``, all frequencies are
        taken to be zero.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The diagonal self-energy matrix.

    Raises
    ------
    TypeError
        If ``n`` or ``frequency_array`` has invalid type.
    ValueError
        If the frequency specification is invalid or incompatible with the
        requested number of modes.

    Notes
    -----
    The same frequency appears in both the x and p sectors of each mode.
    """
    _valid_mode_number(n)
    frequency_array_final = _normalize_frequency_array(n,frequency_array)
    return np.diag(np.concatenate((frequency_array_final, frequency_array_final),axis=0)) 
        

def _xixj_term(n:Integral, subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:
    """
    Construct an x_i x_j coupling term for a quadratic Hamiltonian.

    This function returns a 2n × 2n matrix representing a quadratic
    interaction between the position quadratures of two modes in the
    x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    For subsystem (i, j), the resulting matrix has nonzero entries

        H[x_i, x_j] = H[x_j, x_i] = coupling,

    with all other entries equal to zero.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    subsystem : tuple of (Integral, Integral)
        The pair of mode indices (1-based) defining the interaction.
    coupling : Real
        The coupling strength.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The matrix representing the x_i x_j coupling term.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If the subsystem indices are invalid or if the coupling is not finite.

    Notes
    -----
    This function constructs only the matrix contribution corresponding to
    the quadratic term g x_i x_j. It does not assemble a full Hamiltonian.
    """
    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2 = index_list(n,subsystem)[:2]
    blank_matrix[idx1,idx2] += coupling
    blank_matrix[idx2,idx1] += coupling
    return blank_matrix

def _pipj_term(n:Integral,subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:
    """
    Construct a p_i p_j coupling term for a quadratic Hamiltonian.

    This function returns a 2n × 2n matrix representing a quadratic
    interaction between the momentum quadratures of two modes in the
    x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    For subsystem (i, j), the resulting matrix has nonzero entries

        H[p_i, p_j] = H[p_j, p_i] = coupling,

    with all other entries equal to zero.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    subsystem : tuple of (Integral, Integral)
        The pair of mode indices (1-based) defining the interaction.
    coupling : Real
        The coupling strength.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The matrix representing the p_i p_j coupling term.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If the subsystem indices are invalid or if the coupling is not finite.

    Notes
    -----
    This function constructs only the matrix contribution corresponding to
    the quadratic term g p_i p_j. It does not assemble a full Hamiltonian.
    """
    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idp1, idp2 = index_list(n,subsystem)[2:]
    blank_matrix[idp1,idp2] += coupling
    blank_matrix[idp2,idp1] += coupling
    return blank_matrix

def _xipj_term(n:Integral,subsystem:tuple[Integral,Integral], coupling:Real) -> npt.NDArray[np.float64]:
    """
    Construct an x_i p_j coupling term for a quadratic Hamiltonian.

    This function returns a 2n × 2n matrix representing a quadratic
    interaction between the position quadrature of mode i and the momentum
    quadrature of mode j in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    For subsystem (i, j), the resulting matrix has nonzero entries

        H[x_i, p_j] = H[p_j, x_i] = coupling,

    with all other entries equal to zero.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    subsystem : tuple of (Integral, Integral)
        The ordered pair of mode indices (1-based) defining the interaction.
    coupling : Real
        The coupling strength.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The matrix representing the x_i p_j coupling term.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If the subsystem indices are invalid or if the coupling is not finite.

    Notes
    -----
    This function constructs only the matrix contribution corresponding to
    the quadratic term g x_i p_j. It does not assemble a full Hamiltonian.
    The returned matrix is symmetric to ensure a valid quadratic form.
    """
    _valid_term_inputs(n,subsystem,coupling)
    
    blank_matrix = np.zeros((2*n,2*n))
    idx1, idx2, idp1, idp2 = index_list(n,subsystem)
    blank_matrix[idx1,idp2] += coupling
    blank_matrix[idp2,idx1] += coupling
    return blank_matrix

def _embedding_matrix(n:Integral) ->npt.NDArray[np.float64]:
    """
    Construct the phase-space embedding matrix for appending one mode.

    This function returns the linear embedding matrix that maps an n-mode
    phase-space vector into an (n+1)-mode phase space by appending a new
    mode at the end.

    In the x-then-p ordering, an n-mode vector is ordered as

        (x_1, ..., x_n, p_1, ..., p_n),

    and the enlarged (n+1)-mode space is ordered as

        (x_1, ..., x_n, x_{n+1}, p_1, ..., p_n, p_{n+1}),

    where the appended mode ``n+1`` is interpreted as the pseudomode.
    The embedding maps the original vector to

        (x_1, ..., x_n, 0, p_1, ..., p_n, 0).

    Parameters
    ----------
    n : Integral
        The number of original system modes. Must be strictly positive.

    Returns
    -------
    numpy.ndarray of shape (2n + 2, 2n)
        The embedding matrix from the n-mode phase space into the
        (n+1)-mode phase space.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n`` is not strictly positive.

    Notes
    -----
    The appended mode is always placed as mode ``n+1``. This convention is
    used for pseudomode embeddings throughout the module.
    """
    _valid_mode_number(n)
    return np.kron(np.identity(2),np.eye(n+1, n))

"""
decay_array element (x_p_a:str,mode_id:int,rate:float)
x_p_a: x, p, a, ad
"""

def _compile_single_lindblad_matrix(n: Integral, decay_array:list[tuple[str, Integral, Real],...]|None=None) -> npt.NDArray[np.complex128]:
    """
    Compile the Lindblad Gram matrix for a single environmental channel.

    This function constructs the complex coefficient vector associated with
    one environmental degree of freedom acting on an n-mode system in the
    x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n),

    and returns the corresponding Lindblad Gram matrix as the outer product

        M = outer(conj(c), c),

    where ``c`` is the complex coefficient vector of the channel.

    Each entry in ``decay_array`` specifies one contribution to the same
    environmental channel and has the form

        (decay_type, mode_id, rate),

    where ``decay_type`` is one of ``"x"``, ``"p"``, ``"a"``, or ``"ad"``.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    decay_array : list of tuple or None, optional
        A list of decay specifications contributing to the same environmental
        channel. If ``None``, an empty list is assumed and the zero matrix
        is returned.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The complex Lindblad Gram matrix associated with the compiled channel.

    Raises
    ------
    TypeError
        If ``n`` or any decay element has invalid type.
    ValueError
        If ``n`` is invalid or if any decay element is malformed.

    Notes
    -----
    The returned matrix is Hermitian positive semidefinite by construction.
    Multiple entries in ``decay_array`` are summed coherently into a single
    channel before forming the outer product.

    The supported decay types are interpreted as follows for mode i:
    ``"x"`` -> x_i,
    ``"p"`` -> p_i,
    ``"a"`` -> (x_i + i p_i) / sqrt(2),
    ``"ad"`` -> (x_i - i p_i) / sqrt(2).
    """
    _valid_mode_number(n)
    decay_coefficient_array = np.zeros(2*n, dtype=np.complex128)

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

def _apply_gaussian_channel(mean_drift:npt.NDArray[np.float64], covariance_drift:npt.NDArray[np.complex128], mean_vector:npt.NDArray[np.float64], covariance_vector:npt.NDArray[np.float64], t:Real) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Apply a precompiled Gaussian channel at time t.

    This private helper evolves a Gaussian state's mean vector and covariance
    matrix using matrix exponentials of precompiled drift generators.

    The mean evolves as

        mean(t) = exp(t * mean_drift) @ mean(0),

    while the covariance is assumed to evolve through a vectorized linear
    system,

        cov_vec(t) = exp(t * covariance_drift) @ cov_vec(0).

    The evolved covariance matrix is reconstructed by discarding the final
    auxiliary element of the evolved covariance vector, reshaping the
    remaining entries into a (2n, 2n) matrix, and symmetrizing the result.

    Parameters
    ----------
    mean_drift : numpy.ndarray
        The drift matrix governing mean evolution.
    covariance_drift : numpy.ndarray
        The drift matrix governing evolution of the vectorized covariance
        sector.
    mean_vector : numpy.ndarray
        The initial mean vector of shape (2n,).
    covariance_vector : numpy.ndarray
        The initial vectorized covariance input expected by
        ``covariance_drift``. This is not the covariance matrix itself.
    t : Real
        The evolution time.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The evolved mean vector and evolved covariance matrix.

    Notes
    -----
    This function assumes a specific internal vectorization convention for
    covariance evolution in which the final element of the covariance vector
    is auxiliary and is discarded before reshaping back into a (2n, 2n)
    covariance matrix.

    The returned covariance matrix is symmetrized numerically before being
    returned.
    """
    n = len(mean_vector)//2
    St = expm(t*mean_drift)
    Mt = expm(t*covariance_drift)
    
    evolved_mean = St @ mean_vector
    evolved_covariance_vector = Mt @ covariance_vector
    evolved_covariance = (evolved_covariance_vector[:-1]).reshape((2*n,2*n))
    return (evolved_mean, symmetrize_matrix(evolved_covariance))

def _valid_state_system_pair(state:GaussianCVState, system:"GaussianCVSystem") -> None:
    """
    Validate compatibility between a Gaussian state and system.

    This function checks that the provided objects are valid instances of
    ``GaussianCVState`` and ``GaussianCVSystem`` and that they represent
    the same number of modes.

    Parameters
    ----------
    state : GaussianCVState
        The Gaussian state to be evolved.
    system : GaussianCVSystem
        The system defining the dynamics.

    Raises
    ------
    TypeError
        If ``state`` or ``system`` is not of the expected type.
    ValueError
        If the state and system have mismatched mode numbers.
    """
    if not isinstance(state, GaussianCVState):
        raise TypeError(f"operation requires a valid state object. Got {type(state)}")
    if not isinstance(system, GaussianCVSystem):
        raise TypeError(f"operation requires a valid system object. Got {type(system)}")
    if system.n != state.n:
        raise ValueError(f"system and state objects must represent same mode number. System is {system.n}-mode, while State is {state.n}-mode")
    
"""PUBLIC"""

class GaussianCVSystem:
    """
    Container for an n-mode Gaussian continuous-variable dynamical system.

    This class represents a Gaussian CV system by a quadratic Hamiltonian
    matrix and a Lindblad Gram matrix in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The Hamiltonian matrix defines the coherent quadratic dynamics, while the
    Lindblad Gram matrix defines the dissipative Gaussian channel structure.

    Parameters
    ----------
    hamiltonian_matrix : numpy.ndarray
        The quadratic Hamiltonian matrix of shape (2n, 2n).
    lindblad_matrix : numpy.ndarray
        The Lindblad Gram matrix of shape (2n, 2n).

    Raises
    ------
    TypeError
        If the inputs are not valid NumPy arrays.
    ValueError
        If the Hamiltonian and Lindblad matrices are invalid or dimensionally
        incompatible.

    Notes
    -----
    This class stores the system generator, not the state. Time evolution of
    Gaussian states is defined relative to both matrices.
    """
    def __init__(self, hamiltonian_matrix:npt.NDArray[np.float64],lindblad_matrix:npt.NDArray[np.complex128]):
        """
        Initialize a Gaussian CV system from Hamiltonian and Lindblad matrices.

        Parameters
        ----------
        hamiltonian_matrix : numpy.ndarray
            The quadratic Hamiltonian matrix of shape (2n, 2n).
        lindblad_matrix : numpy.ndarray
            The Lindblad Gram matrix of shape (2n, 2n).

        Raises
        ------
        TypeError
            If the inputs are not valid NumPy arrays.
        ValueError
            If the Hamiltonian and Lindblad matrices are invalid or
            dimensionally incompatible.
        """
        _valid_system(hamiltonian_matrix, lindblad_matrix)
        self.n = (hamiltonian_matrix.shape[0]) // 2
        self._hamiltonian_matrix = hamiltonian_matrix.copy()
        self._lindblad_matrix = lindblad_matrix.copy()

    @classmethod
    def free_evolution(cls,n:Integral, frequency_array:npt.NDArray[np.float64]| None = None) -> "GaussianCVSystem":
        """
        Construct a free-evolution Gaussian system with no dissipation.

        This classmethod returns an n-mode Gaussian system whose Hamiltonian
        consists only of diagonal self-energy terms and whose Lindblad Gram
        matrix is identically zero.

        Parameters
        ----------
        n : Integral
            The number of modes. Must be strictly positive.
        frequency_array : numpy.ndarray or None, optional
            A one-dimensional array of mode frequencies. If fewer than ``n``
            frequencies are supplied, missing entries are set to zero according
            to the rotated-frame convention. If ``None``, all frequencies are
            taken to be zero.

        Returns
        -------
        GaussianCVSystem
            The free-evolution system.

        Raises
        ------
        TypeError
            If ``n`` or ``frequency_array`` has invalid type.
        ValueError
            If the frequency specification is invalid.
        """
        ham = _self_energies(n,frequency_array)
        lind = np.zeros((2*n,2*n), dtype=np.complex128)
        return cls(ham,lind)

    @property
    def hamiltonian_matrix(self) -> npt.NDArray[np.float64]:
        """
        Return a copy of the Hamiltonian matrix.

        Returns
        -------
        numpy.ndarray
            A copy of the quadratic Hamiltonian matrix.
        """
        return self._hamiltonian_matrix.copy()

    @property
    def lindblad_matrix(self) -> npt.NDArray[np.complex128]:
        """
        Return a copy of the Lindblad Gram matrix.

        Returns
        -------
        numpy.ndarray
            A copy of the Lindblad Gram matrix.
        """
        return self._lindblad_matrix.copy()

    def position_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a position-position coupling term to the Hamiltonian.
    
        This adds a quadratic interaction of the form
    
            H += g x_i x_j
    
        in the x-then-p phase-space ordering.
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix += _xixj_term(self.n, subsystem, coupling)
        return self

    def momentum_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a momentum-momentum coupling term to the Hamiltonian.
    
        This adds a quadratic interaction of the form
    
            H += g p_i p_j
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix += _pipj_term(self.n, subsystem, coupling)
        return self

    def position_i_momentum_j_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a cross-quadrature coupling term to the Hamiltonian.
    
        This adds a quadratic interaction of the form
    
            H += g x_i p_j
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The ordered pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix += _xipj_term(self.n, subsystem, coupling)
        return self

    def beamsplitter_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a beamsplitter-type coupling to the Hamiltonian.
    
        This adds the quadratic interaction
    
            H += g (x_i x_j + p_i p_j),
    
        which corresponds to a number-conserving bilinear coupling between modes.
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix += _xixj_term(self.n, subsystem, coupling)
        self._hamiltonian_matrix += _pipj_term(self.n, subsystem, coupling)
        return self

    def squeezer_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a two-mode squeezing coupling to the Hamiltonian.
    
        This adds the quadratic interaction
    
            H += g (x_i x_j - p_i p_j),
    
        corresponding to a non-number-conserving two-mode squeezing interaction.
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix += _xixj_term(self.n, subsystem, coupling)
        self._hamiltonian_matrix -= _pipj_term(self.n, subsystem, coupling)
        return self

    def position_difference_coupling(self, subsystem:tuple[Integral,Integral], coupling:Real):
        """
        Add a position-difference quadratic term to the Hamiltonian.
    
        This adds the interaction
    
            H += g (x_i - x_j)^2
               = g (x_i^2 + x_j^2 - 2 x_i x_j),
    
        implemented through a combination of quadratic terms.
    
        Parameters
        ----------
        subsystem : tuple of (Integral, Integral)
            The pair of mode indices (1-based).
        coupling : Real
            The coupling strength.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
        """
        self._hamiltonian_matrix -= 2*_xixj_term(self.n, subsystem, coupling)
        self._hamiltonian_matrix += _xixj_term(self.n, (subsystem[0],subsystem[0]), coupling)
        self._hamiltonian_matrix += _xixj_term(self.n, (subsystem[1],subsystem[1]), coupling)
        return self

    def multi_position_dissipator(self, subsystem:tuple[Integral,...], decay:Real):
        """
        Add a collective position dissipator to the system.
    
        This method adds a single-environment dissipative channel with collective
        jump operator proportional to
    
            L ∝ sum_i sqrt(decay) x_i,
    
        where the sum runs over the specified subsystem modes.
    
        Parameters
        ----------
        subsystem : tuple of Integral
            The mode indices (1-based) included in the collective dissipator.
        decay : Real
            The decay rate associated with each participating mode.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
    
        Raises
        ------
        TypeError
            If the inputs have invalid type.
        ValueError
            If the subsystem indices are invalid or if the decay rate is invalid.
    
        Notes
        -----
        This is a collective dissipator: all selected modes couple to the same
        environmental channel.
        """
        _valid_indices(self.n, subsystem)
        decay_array = [("x", idx, decay) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self.n, decay_array)
        return self

    def multi_annihilation_dissipator(self, subsystem:tuple[Integral,...], decay:Real):
        """
        Add a collective annihilation dissipator to the system.
    
        This method adds a single-environment dissipative channel with collective
        jump operator proportional to
    
            L ∝ sum_i sqrt(decay) a_i,
    
        where the sum runs over the specified subsystem modes.
    
        Parameters
        ----------
        subsystem : tuple of Integral
            The mode indices (1-based) included in the collective dissipator.
        decay : Real
            The decay rate associated with each participating mode.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
    
        Raises
        ------
        TypeError
            If the inputs have invalid type.
        ValueError
            If the subsystem indices are invalid or if the decay rate is invalid.
    
        Notes
        -----
        This is a collective dissipator: all selected modes couple to the same
        environmental channel.
        """
        _valid_indices(self.n, subsystem)
        decay_array = [("a", idx, decay) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self.n, decay_array)
        return self

    def multi_thermal_dissipator(self, subsystem:tuple[Integral,...], decay:Real, thermal_occupation:Real):
        """
        Add a collective thermal dissipator to the system.
    
        This method adds a collective thermal bath acting on the specified modes,
        with emission and absorption contributions corresponding to thermal
        occupation ``thermal_occupation``. The resulting dissipator is built from
        two independent collective channels:
    
            L_emission   ∝ sum_i sqrt(decay * (thermal_occupation + 1)) a_i,
            L_absorption ∝ sum_i sqrt(decay * thermal_occupation) a_i^†.
    
        Parameters
        ----------
        subsystem : tuple of Integral
            The mode indices (1-based) included in the collective dissipator.
        decay : Real
            The base decay rate.
        thermal_occupation : Real
            The bath thermal occupation number. Must be non-negative.
    
        Returns
        -------
        GaussianCVSystem
            The updated system (self).
    
        Raises
        ------
        TypeError
            If the inputs have invalid type.
        ValueError
            If the subsystem indices are invalid or if ``decay`` or
            ``thermal_occupation`` is invalid.
    
        Notes
        -----
        This is a collective thermal dissipator: all selected modes couple to
        the same thermal environment. Emission and absorption are added as
        separate Lindblad channels.
        """
        _require_nonnegative_real_scalar(thermal_occupation, 'thermal occupation')
        _require_nonnegative_real_scalar(decay, "decay rate")
        _valid_indices(self.n, subsystem)
        emission_array = [("a", idx, decay*(thermal_occupation+1)) for idx in subsystem]
        absorption_array = [("ad", idx, decay*(thermal_occupation)) for idx in subsystem]
        self._lindblad_matrix += _compile_single_lindblad_matrix(self.n, emission_array)
        self._lindblad_matrix += _compile_single_lindblad_matrix(self.n, absorption_array)
        return self

    def generate_drift_and_diffusion(self) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        """
        Generate the Gaussian drift and diffusion matrices.
    
        This method constructs the matrices governing first- and second-moment
        evolution for the current Gaussian CV system in the x-then-p phase-space
        ordering
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        The returned matrices ``(A, D)`` define the Gaussian moment equations
    
            d/dt mean = A @ mean,
            d/dt cov  = A @ cov + cov @ A.T + D.
    
        They are computed from the system Hamiltonian matrix ``H``, Lindblad
        Gram matrix ``M``, and canonical symplectic form ``Ω`` as
    
            A = Ω @ (H + Im(M)),
            D = Ω @ Re(M) @ Ω.T.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            The drift matrix ``A`` and diffusion matrix ``D``, each of shape
            (2n, 2n).
    
        Notes
        -----
        Any residual imaginary parts arising from numerical roundoff are removed
        with ``np.real`` before returning the result.
        """
        omega_matrix = symplectic_matrix(self.n)

        drift = np.real(omega_matrix @ (self.hamiltonian_matrix + np.imag(self.lindblad_matrix)))
        diffusion = np.real(omega_matrix @ np.real(self.lindblad_matrix) @ (omega_matrix.T))

        return (drift, diffusion)

    def copy_system(self) -> "GaussianCVSystem":
        """
        Create a deep copy of the Gaussian CV system.
    
        This method returns a new ``GaussianCVSystem`` instance with copies of
        the Hamiltonian and Lindblad matrices. The returned system is fully
        independent of the original.
    
        Returns
        -------
        GaussianCVSystem
            A new system instance with identical parameters.
    
        Notes
        -----
        The Hamiltonian and Lindblad matrices are copied, so modifying the
        returned system does not affect the original.
        """
        return GaussianCVSystem(self._hamiltonian_matrix.copy(), self._lindblad_matrix.copy())

    def gaussian_channel(self) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        """
        Construct the linear generators for Gaussian moment evolution.
    
        This method returns the matrices governing the evolution of the mean
        vector and the vectorized covariance matrix in the x-then-p phase-space
        ordering
    
        (x_1, ..., x_n, p_1, ..., p_n).
    
        The mean evolves as
    
        d/dt mean = A @ mean,
    
        where ``A`` is the drift matrix.
    
        The covariance evolves according to
    
        d/dt vec(cov) = K @ vec(cov) + vec(D),
    
        where
    
        K = A ⊗ I + I ⊗ A,
    
        and ``D`` is the diffusion matrix. This affine evolution is embedded
        into a linear system by augmenting the state vector as
    
        (vec(cov), 1),
    
        yielding
    
        d/dt [vec(cov); 1] = A_covariance @ [vec(cov); 1].
    
        Parameters
        ----------
        None
    
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
        The pair ``(A, A_covariance)``, where:
        - ``A`` has shape (2n, 2n) and governs mean evolution
        - ``A_covariance`` has shape (4n^2 + 1, 4n^2 + 1) and governs the
        augmented covariance evolution
    
        Notes
        -----
        The augmented covariance generator encodes both drift and diffusion
        in a linear form suitable for matrix-exponential propagation.
        """
        m = int(2*self.n)
        m2 = int(4*self.n**2)
        I = np.identity(m)
        A, D = self.generate_drift_and_diffusion()
        K, D_vec = np.kron(A,I) + np.kron(I, A), D.flatten() 
        
        A_covariance = np.block([
        [K,              D_vec.reshape(m2, 1)],
        [np.zeros((1, m2)), np.zeros((1, 1))]
        ])
        return (A, A_covariance)
        

    def evolve_state(self, state:GaussianCVState, t_eval:npt.NDArray[np.float64]) -> GaussianSolution:
        """
        Evolve a Gaussian state over a specified time grid.
    
        This method evolves the supplied ``GaussianCVState`` under the current
        ``GaussianCVSystem`` at each time in ``t_eval`` using the closed-form
        Gaussian channel generated from the system drift and diffusion matrices.
    
        The evolution is performed in the x-then-p phase-space ordering
    
            (x_1, ..., x_n, p_1, ..., p_n),
    
        with the mean and covariance propagated independently through the
        precompiled Gaussian channel representation.
    
        Parameters
        ----------
        state : GaussianCVState
            The initial Gaussian state to evolve.
        t_eval : numpy.ndarray
            A one-dimensional array of evaluation times. Entries must be real,
            finite, non-negative, non-empty, and strictly increasing.
    
        Returns
        -------
        GaussianSolution
            A solution object containing the evaluation times, evolved mean
            vectors, and evolved covariance matrices.
    
        Raises
        ------
        TypeError
            If ``state`` or ``t_eval`` has invalid type.
        ValueError
            If the state and system are incompatible, if ``t_eval`` is invalid,
            or if any evolved covariance matrix fails the quantum physicality test.
    
        Notes
        -----
        The input state is not modified. The covariance evolution is implemented
        using an augmented vectorized representation of the form
    
            (vec(covariance), 1),
    
        so that affine covariance evolution can be handled by a linear matrix
        exponential.
        """
        _valid_state_system_pair(state, self)
        _valid_t_eval(t_eval)
        x0, c0 = state.mean_vector, state.covariance_matrix
        c0_vec = np.concatenate((c0.flatten(),np.array([1.0])),axis=0)
        A, A_covariance = self.gaussian_channel()
        
        means = []
        covariances = []

        for t in t_eval:
            mean, covariance = _apply_gaussian_channel(A, A_covariance, x0, c0_vec, t)
            
            require_physical_covariance(covariance)
            
            means.append(mean)
            covariances.append(covariance)
        
        return GaussianSolution(t_eval,means,covariances)

def single_pole_ou_embedding(state:GaussianCVState, system:GaussianCVSystem, subsystem:tuple[Integral,...], coupling_types:tuple[str,...], memory_rate:Real, env_freq:Real, decay_rate:Real, thermal_occupation:Real = 0) -> tuple[GaussianCVState, GaussianCVSystem]:
    """
    Construct the single-pole Ornstein–Uhlenbeck pseudomode embedding.

    This function embeds an n-mode Gaussian state and system into an
    (n+1)-mode representation by appending a pseudomode as mode ``n+1``.
    The phase-space convention is x-then-p ordering,

        (x_1, ..., x_n, x_{n+1}, p_1, ..., p_n, p_{n+1}).

    The embedding implements a single-pole Ornstein–Uhlenbeck (OU) environment
    via a Markovian pseudomode model. The appended mode represents the
    environmental memory degree of freedom.

    Construction steps
    ------------------
    - Embed the mean vector, covariance matrix, Hamiltonian, and Lindblad
      Gram matrix into the enlarged (n+1)-mode phase space.
    - Initialize the pseudomode in a thermal state with occupation
      ``thermal_occupation``.
    - Add a thermal dissipator acting on the pseudomode with decay rate

          γ_pseudo = 2 * memory_rate.

    - Add system–pseudomode interaction terms derived from the rotating-wave
      approximation (RWA) interaction

          H_int = L c^† + L^† c,

      where ``c`` is the pseudomode annihilation operator and ``L`` is the
      system coupling operator.

    Coupling conventions
    --------------------
    Each entry of ``coupling_types`` specifies the form of the system operator
    ``L`` for the corresponding mode in ``subsystem``:

    - ``"position"``:
        L = x_i

        H_int = x_i (c^† + c) = sqrt(2) x_i x_{n+1}

        Implemented as a position-position coupling with an additional
        factor of sqrt(2) applied to the coupling strength.

    - ``"momentum"``:
        L = p_i

        H_int = p_i (c^† + c) = sqrt(2) p_i x_{n+1}

        Implemented as a cross-quadrature coupling between p_i and the
        pseudomode position x_{n+1}, with an additional factor of sqrt(2).

    - ``"annihilation"``:
        L = a_i

        H_int = a_i c^† + a_i^† c

        Implemented as a beamsplitter-type coupling

            x_i x_{n+1} + p_i p_{n+1}.

    The effective coupling strength is

        g = sqrt(decay_rate * memory_rate / 2),

    and the sqrt(2) factor required for Hermitian couplings (position and
    momentum) is applied explicitly in those branches.

    The pseudomode self-energy is added with frequency ``env_freq``.

    Parameters
    ----------
    state : GaussianCVState
        The initial n-mode Gaussian state.
    system : GaussianCVSystem
        The corresponding n-mode Gaussian system.
    subsystem : tuple of Integral
        The system mode indices (1-based) coupled to the pseudomode.
    coupling_types : tuple of str
        The coupling type for each subsystem mode. Must contain only
        ``"position"``, ``"momentum"``, or ``"annihilation"``.
    memory_rate : Real
        The OU memory rate. Must be strictly positive.
    env_freq : Real
        The pseudomode frequency.
    decay_rate : Real
        The effective system–environment coupling rate.
    thermal_occupation : Real
        The thermal occupation number of the environment. Must be non-negative.

    Returns
    -------
    tuple of (GaussianCVState, GaussianCVSystem)
        The embedded Gaussian state and system in the enlarged (n+1)-mode space.

    Raises
    ------
    TypeError
        If the inputs have invalid types.
    ValueError
        If the state and system are incompatible, if the embedding parameters
        are invalid, or if any coupling type is not recognized.

    Notes
    -----
    The appended pseudomode is always mode ``n+1``. The embedding produces a
    Markovian system whose reduced dynamics reproduce a single-pole OU
    environment for the specified subsystem couplings.
    """
    _valid_state_system_pair(state, system)
    n = state.n
    _valid_single_pole_input(n, subsystem, coupling_types, memory_rate, env_freq, thermal_occupation)
    _require_nonnegative_real_scalar(decay_rate, "environmental decay")
    embedding_matrix = _embedding_matrix(n)
    mean_vector, covariance_matrix = state.mean_vector, state.covariance_matrix
    hamiltonian_matrix, lindblad_matrix = system.hamiltonian_matrix, system.lindblad_matrix

    embedded_mean = embedding_matrix @ mean_vector
    embedded_covariance = symmetrize_matrix((embedding_matrix @ covariance_matrix) @ embedding_matrix.T)

    embedded_hamiltonian = symmetrize_matrix((embedding_matrix @ hamiltonian_matrix) @ embedding_matrix.T)
    embedded_lindblad = symmetrize_matrix((embedding_matrix @ lindblad_matrix) @ embedding_matrix.T)

    embedded_state = GaussianCVState(embedded_mean, embedded_covariance)
    embedded_state.single_mode_thermal_reset(thermal_occupation,n+1)

    embedded_system = GaussianCVSystem(embedded_hamiltonian, embedded_lindblad)
    pseudo_coupling = np.sqrt(decay_rate*memory_rate/2)
    pseudo_decay = 2*memory_rate
    embedded_system.multi_thermal_dissipator((n+1,), pseudo_decay, thermal_occupation)
    
    for idx in range(len(coupling_types)):
        subsystem_id = subsystem[idx]
        coupling_type = coupling_types[idx]
        
        if coupling_type == "position":
            embedded_system.position_coupling((subsystem_id,n+1), pseudo_coupling*np.sqrt(2))
        elif coupling_type == "momentum":
            embedded_system.position_i_momentum_j_coupling((n+1,subsystem_id), pseudo_coupling*np.sqrt(2))
        elif coupling_type == "annihilation":
            embedded_system.beamsplitter_coupling((subsystem_id,n+1), pseudo_coupling)
        else:
            raise ValueError(f"individual pseudo-system couplings must be 'position', 'momentum', or 'annihilation' got {coupling_type}")
        
    pseudo_array = np.zeros(n+1)
    pseudo_array[n] = env_freq
    embedded_system._hamiltonian_matrix += _self_energies(n+1, pseudo_array)
        
    return embedded_state, embedded_system


    