import numpy as np
import numpy.typing as npt
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.patches import Ellipse

from .conventions import symmetrize_matrix, rotation_matrix, index_list, compress_mean_covariance, mean_subsystem, covariance_subsystem
from ._validation import _valid_mode_number, _valid_nbars_array, _valid_mean_covariance_tuple, _valid_indices, _valid_parameter_tuple, _valid_mean_covariance, _require_real_scalar, _require_nonnegative_real_scalar, _require_positive_integral_scalar, _require_tuple_length, _require_positive_real_scalar
from numbers import Real, Integral, Complex

"""Public"""

def thermal_vacuum_covariance(n: Integral, nbars:npt.NDArray[np.float64]|None=None) -> npt.NDArray[np.float64]:
    """
    Construct the covariance matrix of an n-mode thermal-vacuum product state.

    This function returns the covariance matrix for an n-mode product state
    with thermal occupations ``nbars`` in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    For mode occupations ``nbar_i``, the covariance matrix is diagonal with
    entries

        nbar_i + 1/2

    appearing once in the x block and once in the p block. Thus the returned
    matrix has the form

        diag(v_1, ..., v_n, v_1, ..., v_n),

    where ``v_i = nbar_i + 1/2``.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.
    nbars : numpy.ndarray or None, optional
        A one-dimensional array of thermal occupation numbers for the modes.
        Entries must be real, finite, and non-negative. If ``None``, all modes
        are taken to be in vacuum, corresponding to zero thermal occupation.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The covariance matrix of the thermal-vacuum product state.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or if ``nbars`` is not a valid NumPy array
        when provided.
    ValueError
        If ``n`` is not strictly positive, if ``nbars`` contains invalid values,
        or if more than ``n`` thermal occupations are supplied.

    Warns
    -----
    UserWarning
        If fewer than ``n`` thermal occupations are provided. Missing modes are
        assigned zero thermal occupation.

    Notes
    -----
    This function constructs only the covariance matrix. The corresponding mean
    vector is zero for all modes.
    """
    _valid_mode_number(n)
    
    if nbars is None:
        nbars_full = np.zeros(n)
    else: 
        _valid_nbars_array(nbars)
        if n > len(nbars):
            missing_modes_count = int(n-len(nbars))
            warnings.warn(f"{n}-mode system but only {len(nbars)} thermal occupations provided. Assuming 0 thermal occupation for modes {len(nbars)+1}-{n}.")
            nbars_full = np.append(nbars,np.zeros(missing_modes_count))
        elif n == len(nbars):
            nbars_full = nbars
        else: 
            raise ValueError(f"thermal occupation contains more values than modes. Got {len(nbars)} values, expected {n} values.")
    
    diagonal_elements = nbars_full + 0.5
    return np.diag(np.concatenate((diagonal_elements, diagonal_elements),axis=0))

def thermal_vacuum_mean(n: Integral) -> npt.NDArray[np.float64]:
    """
    Construct the mean vector of an n-mode thermal-vacuum state.

    This function returns the mean vector for an n-mode Gaussian state
    with zero displacement. In the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n),

    the mean vector is identically zero.

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.

    Returns
    -------
    numpy.ndarray of shape (2n,)
        The zero mean vector.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n`` is not strictly positive.

    Notes
    -----
    This corresponds to the mean vector of a thermal or vacuum state.
    """
    _valid_mode_number(n)
    return np.zeros(2*n)

def apply_1_mode_displacement(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], displacement: Complex, mode_id:Integral) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Apply a single-mode displacement to a Gaussian state.

    This function applies a phase-space displacement to one mode of a Gaussian
    state represented by a mean vector and covariance matrix. In the x-then-p
    phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n),

    a complex displacement ``alpha`` acts on the selected mode as

        Δx = sqrt(2) * Re(alpha),
        Δp = sqrt(2) * Im(alpha).

    The mean vector is shifted accordingly, while the covariance matrix is
    unchanged.

    Parameters
    ----------
    mean_vector_covariance_matrix_tuple : tuple of (numpy.ndarray, numpy.ndarray)
        The Gaussian state given as ``(mean_vector, covariance_matrix)``.
    displacement : Complex
        The complex displacement amplitude. Must be finite.
    mode_id : Integral
        The target mode index (1-based).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The displaced Gaussian state as ``(new_mean_vector, covariance_matrix)``.

    Raises
    ------
    TypeError
        If the state is invalid, if ``displacement`` is not complex-valued,
        or if ``mode_id`` is not an integer index.
    ValueError
        If ``displacement`` is not finite or if ``mode_id`` is outside the
        valid range.

    Notes
    -----
    Only the mean vector is modified. The covariance matrix is returned
    unchanged.
    """
    _valid_mean_covariance_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,(mode_id,))

    if not isinstance(displacement,Complex):
        raise TypeError(f"single mode displacement must be complex-valued. Got {type(displacement)}")
    if not np.isfinite(displacement):
        raise ValueError(f"single mode displacement must be finite. Got {displacement}")
    
    x_disp = np.sqrt(2) * np.real(displacement)
    p_disp = np.sqrt(2) * np.imag(displacement)

    idx, idp = index_list(n,(mode_id,))
    
    x_shift_vector = np.zeros(2*n)
    p_shift_vector = np.zeros(2*n)
    
    x_shift_vector[idx] = x_disp
    p_shift_vector[idp] = p_disp
            
    return (mean_vector + x_shift_vector + p_shift_vector, covariance_matrix)

def single_mode_squeeze_matrix(squeeze_magnitude:Real, squeeze_angle:Real) -> npt.NDArray[np.float64]:
    """
    Construct a single-mode squeezing matrix.

    This function returns the 2×2 symplectic matrix representing a
    single-mode squeezing operation in the (x, p) quadrature basis.

    The transformation is given by

        S(r, φ) = R(φ/2) @ diag(exp(-r), exp(r)) @ R(φ/2)^T,

    where r is the squeezing magnitude and φ is the squeezing angle.
    Here R(θ) is the phase-space rotation matrix.

    Parameters
    ----------
    squeeze_magnitude : Real
        The squeezing magnitude r. Must be non-negative.
    squeeze_angle : Real
        The squeezing angle φ in radians. Must be finite.

    Returns
    -------
    numpy.ndarray of shape (2, 2)
        The single-mode squeezing matrix.

    Raises
    ------
    TypeError
        If inputs are not real scalars.
    ValueError
        If ``squeeze_magnitude`` is negative or if inputs are not finite.

    Notes
    -----
    The matrix acts on phase-space vectors ordered as (x, p).
    For r = 0, the identity transformation is returned.
    """
    _require_nonnegative_real_scalar(squeeze_magnitude, 'squeezing magnitude')
    _require_real_scalar(squeeze_angle, 'squeezing angle')
    
    rot_matrix = rotation_matrix(squeeze_angle/2)
    return (rot_matrix @ np.diag([np.exp(-squeeze_magnitude),np.exp(squeeze_magnitude)])) @ rot_matrix.T

def apply_1_mode_squeeze_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                                 squeeze_magnitude_angle_tuple:tuple[Real,Real], mode_id:Integral) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Apply a single-mode squeezing unitary to a Gaussian state.

    This function applies a single-mode squeezing transformation to one mode
    of an n-mode Gaussian state represented by a mean vector and covariance
    matrix in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If the squeezing parameters are ``(r, φ)``, the single-mode symplectic
    transformation is

        S(r, φ) = R(φ/2) @ diag(exp(-r), exp(r)) @ R(φ/2)^T,

    where ``R`` is the single-mode rotation matrix. This 2×2 transformation
    is embedded into the full 2n-dimensional phase space and applied as

        mean -> S_full @ mean
        cov  -> S_full @ cov @ S_full.T

    Parameters
    ----------
    mean_vector_covariance_matrix_tuple : tuple of (numpy.ndarray, numpy.ndarray)
        The Gaussian state given as ``(mean_vector, covariance_matrix)``.
    squeeze_magnitude_angle_tuple : tuple of (Real, Real)
        The squeezing parameters ``(squeeze_magnitude, squeeze_angle)``.
        The squeezing magnitude must be non-negative and the squeezing angle
        must be finite.
    mode_id : Integral
        The target mode index (1-based).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The transformed Gaussian state as
        ``(new_mean_vector, new_covariance_matrix)``.

    Raises
    ------
    TypeError
        If the state, squeezing parameters, or mode index have invalid types.
    ValueError
        If the state is invalid, if the squeezing magnitude is negative,
        if the squeezing angle is not finite, or if ``mode_id`` is outside
        the valid range.

    Notes
    -----
    The transformed covariance matrix is symmetrized numerically before
    being returned. Only the selected mode is squeezed; all other modes
    are left unchanged.
    """
    _valid_mean_covariance_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple

    _valid_parameter_tuple(squeeze_magnitude_angle_tuple,'squeeze magnitude', 'squeeze angle')
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,(mode_id,))
    
    transformed_idx = index_list(n,(mode_id,))
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    
    single_squeeze_matrix = single_mode_squeeze_matrix(squeeze_magnitude, squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) +  (selection_matrix.T) @ (single_squeeze_matrix - np.identity(2)) @ selection_matrix
    
    return (n_mode_squeeze_unitary @ mean_vector, symmetrize_matrix((n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T))


def two_mode_mixing_matrix(coupling_magnitude:Real,coupling_angle:Real) -> npt.NDArray[np.float64]:
    """
    Construct a two-mode passive mixing (beam-splitter) matrix.

    This function returns the real symplectic representation of a two-mode
    passive Gaussian unitary. The transformation is defined by a complex
    2×2 unitary matrix

        U = [[cos(r),  exp(i φ) sin(r)],
             [-exp(-i φ) sin(r), cos(r)]],

    where r is the mixing magnitude and φ is a phase.

    This unitary is mapped to a real 4×4 symplectic matrix acting on the
    phase-space vector ordered as

        (x_1, x_2, p_1, p_2),

    via

        S = [[Re(U), -Im(U)],
             [Im(U),  Re(U)]].

    Parameters
    ----------
    coupling_magnitude : Real
        The mixing magnitude r. Must be non-negative.
    coupling_angle : Real
        The phase φ in radians. Must be finite.

    Returns
    -------
    numpy.ndarray of shape (4, 4)
        The two-mode mixing symplectic matrix.

    Raises
    ------
    TypeError
        If inputs are not real scalars.
    ValueError
        If ``coupling_magnitude`` is negative or if inputs are not finite.

    Notes
    -----
    This transformation is passive (number-conserving) and corresponds to
    a beam-splitter–type interaction between two modes.
    """
    _require_nonnegative_real_scalar(coupling_magnitude, 'mixing magnitude')
    _require_real_scalar(coupling_angle, 'mixing angle')
    unitary_exponential_matrix = np.array([
        [np.cos(coupling_magnitude), np.exp(1j*coupling_angle)*np.sin(coupling_magnitude)],
        [-np.exp(-1j*coupling_angle)*np.sin(coupling_magnitude), np.cos(coupling_magnitude)]
    ])
    return np.kron(np.identity(2),np.real(unitary_exponential_matrix)) + np.kron(np.array([[0,-1],[1,0]]),np.imag(unitary_exponential_matrix))

def apply_2_mode_mix_unitary(mean_vector_covariance_matrix_tuple:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]],
                                 coupling_magnitude_angle_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Apply a two-mode passive mixing unitary to a Gaussian state.

    This function applies a two-mode passive Gaussian unitary to an n-mode
    Gaussian state represented by a mean vector and covariance matrix in the
    x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If the mixing parameters are ``(r, φ)``, the corresponding two-mode
    symplectic transformation is the 4×4 real phase-space representation
    returned by ``two_mode_mixing_matrix(r, φ)``. This transformation is
    embedded into the full 2n-dimensional phase space and applied as

        mean -> S_full @ mean
        cov  -> S_full @ cov @ S_full.T

    Parameters
    ----------
    mean_vector_covariance_matrix_tuple : tuple of (numpy.ndarray, numpy.ndarray)
        The Gaussian state given as ``(mean_vector, covariance_matrix)``.
    coupling_magnitude_angle_tuple : tuple of (Real, Real)
        The mixing parameters ``(mixing_magnitude, mixing_angle)``.
        The mixing magnitude must be non-negative and the mixing angle
        must be finite.
    mode_ids : tuple of (Integral, Integral)
        The pair of mode indices (1-based) defining the two-mode subsystem.
        The two indices must be distinct.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The transformed Gaussian state as
        ``(new_mean_vector, new_covariance_matrix)``.

    Raises
    ------
    TypeError
        If the state, mixing parameters, or mode indices have invalid types.
    ValueError
        If the state is invalid, if the mixing magnitude is negative,
        if the mixing angle is not finite, if the tuple does not have length 2,
        if the indices are not distinct, or if any mode index is out of bounds.

    Notes
    -----
    The transformed covariance matrix is symmetrized numerically before
    being returned. The selected subsystem is ordered as

        (x_i, x_j, p_i, p_j)

    for ``mode_ids = (i, j)``.
    """
    _valid_mean_covariance_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple

    _valid_parameter_tuple(coupling_magnitude_angle_tuple,'mixing magnitude','mixing angle')
    coupling_magnitude, coupling_angle = coupling_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,mode_ids)
    _require_tuple_length(mode_ids, 2, "mixing indices")
    if mode_ids[0] == mode_ids[1]:
        raise ValueError(f"mixing operation is defined on two distinct modes. Expected mode_ids to contain two non-equal Integral values. Got {mode_ids}")
    
    transformed_idx = index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_mix_matrix = two_mode_mixing_matrix(coupling_magnitude,coupling_angle)
    n_mode_mix_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_mix_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_mix_unitary @ mean_vector, symmetrize_matrix((n_mode_mix_unitary @ covariance_matrix) @ n_mode_mix_unitary.T))

def two_mode_squeezing_matrix(squeeze_magnitude:Real,squeeze_angle:Real) -> npt.NDArray[np.float64]:
    """
    Construct a two-mode squeezing (active) symplectic matrix.

    This function returns the 4×4 real symplectic matrix representing a
    two-mode squeezing transformation in the x-then-p phase-space ordering

        (x_1, x_2, p_1, p_2).

    The transformation is parameterized by a squeezing magnitude r and
    squeezing angle φ, with

        μ = cosh(r),
        ν = sinh(r) * exp(i φ).

    Writing ν = ν_r + i ν_i, the resulting matrix is

        [[ μ,  ν_r,  0,  ν_i],
         [ν_r,  μ,  ν_i,  0 ],
         [ 0,  ν_i,  μ, -ν_r],
         [ν_i,  0, -ν_r,  μ ]].

    Parameters
    ----------
    squeeze_magnitude : Real
        The squeezing magnitude r. Must be non-negative.
    squeeze_angle : Real
        The squeezing angle φ in radians. Must be finite.

    Returns
    -------
    numpy.ndarray of shape (4, 4)
        The two-mode squeezing symplectic matrix.

    Raises
    ------
    TypeError
        If inputs are not real scalars.
    ValueError
        If ``squeeze_magnitude`` is negative or if inputs are not finite.

    Notes
    -----
    This transformation is active (non-number-conserving) and generates
    correlations and entanglement between the two modes.
    """
    _require_nonnegative_real_scalar(squeeze_magnitude, 'squeezing magnitude')
    _require_real_scalar(squeeze_angle, 'squeezing angle')
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
    """
    Apply a two-mode squeezing unitary to a Gaussian state.

    This function applies a two-mode active Gaussian unitary to an n-mode
    Gaussian state represented by a mean vector and covariance matrix in the
    x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If the squeezing parameters are ``(r, φ)``, the corresponding two-mode
    symplectic transformation is the 4×4 real phase-space representation
    returned by ``two_mode_squeezing_matrix(r, φ)``. This transformation is
    embedded into the full 2n-dimensional phase space and applied as

        mean -> S_full @ mean
        cov  -> S_full @ cov @ S_full.T

    Parameters
    ----------
    mean_vector_covariance_matrix_tuple : tuple of (numpy.ndarray, numpy.ndarray)
        The Gaussian state given as ``(mean_vector, covariance_matrix)``.
    squeeze_magnitude_angle_tuple : tuple of (Real, Real)
        The squeezing parameters ``(squeeze_magnitude, squeeze_angle)``.
        The squeezing magnitude must be non-negative and the squeezing angle
        must be finite.
    mode_ids : tuple of (Integral, Integral)
        The pair of mode indices (1-based) defining the two-mode subsystem.
        The two indices must be distinct.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The transformed Gaussian state as
        ``(new_mean_vector, new_covariance_matrix)``.

    Raises
    ------
    TypeError
        If the state, squeezing parameters, or mode indices have invalid types.
    ValueError
        If the state is invalid, if the squeezing magnitude is negative,
        if the squeezing angle is not finite, if ``mode_ids`` does not have
        length 2, if the two indices are not distinct, or if any mode index
        is outside the valid range.

    Notes
    -----
    The selected subsystem is ordered as

        (x_i, x_j, p_i, p_j)

    for ``mode_ids = (i, j)``. The transformed covariance matrix is
    symmetrized numerically before being returned.
    """
    _valid_mean_covariance_tuple(mean_vector_covariance_matrix_tuple)
    mean_vector, covariance_matrix = mean_vector_covariance_matrix_tuple
    
    _valid_parameter_tuple(squeeze_magnitude_angle_tuple,'squeeze magnitude','squeeze angle')
    squeeze_magnitude, squeeze_angle = squeeze_magnitude_angle_tuple
    
    n = len(mean_vector)//2
    _valid_mode_number(n)
    _valid_indices(n,mode_ids)
    _require_tuple_length(mode_ids, 2, "two-mode squeeze indices")
    if mode_ids[0] == mode_ids[1]:
        raise ValueError(f"two-mode squeeze operation is defined on two distinct modes. Expected mode_ids to contain two non-equal Integral values. Got {mode_ids}")
    
    transformed_idx = index_list(n,mode_ids)
    selection_matrix = (np.identity(2*n))[transformed_idx,:]
    two_squeeze_matrix = two_mode_squeezing_matrix(squeeze_magnitude,squeeze_angle)
    n_mode_squeeze_unitary = np.identity(2*n) + (selection_matrix.T) @ (two_squeeze_matrix - np.identity(4)) @ selection_matrix
    return (n_mode_squeeze_unitary @ mean_vector, symmetrize_matrix((n_mode_squeeze_unitary @ covariance_matrix) @ n_mode_squeeze_unitary.T))

class GaussianCVState:
    """
    Container for an n-mode Gaussian continuous-variable state.

    This class represents a Gaussian CV state by its mean vector and
    covariance matrix in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The state is validated on construction and internal copies of the
    supplied arrays are stored. The ``mean_vector`` and
    ``covariance_matrix`` properties also return copies, so external code
    cannot mutate the internal state accidentally.

    Parameters
    ----------
    mean_vector : numpy.ndarray
        The Gaussian mean vector of shape (2n,).
    covariance_matrix : numpy.ndarray
        The Gaussian covariance matrix of shape (2n, 2n).

    Raises
    ------
    TypeError
        If the inputs are not valid NumPy arrays.
    ValueError
        If the mean vector and covariance matrix are not valid or are
        dimensionally inconsistent.

    Notes
    -----
    This class stores only first and second moments. It therefore represents
    Gaussian states completely, but does not encode non-Gaussian information.
    """
    def __init__(self, mean_vector: npt.NDArray[np.float64], covariance_matrix: npt.NDArray[np.float64]):
        """
        Initialize a Gaussian CV state from a mean vector and covariance matrix.

        Parameters
        ----------
        mean_vector : numpy.ndarray
            The Gaussian mean vector of shape (2n,).
        covariance_matrix : numpy.ndarray
            The Gaussian covariance matrix of shape (2n, 2n).

        Raises
        ------
        TypeError
            If the inputs are not valid NumPy arrays.
        ValueError
            If the mean vector and covariance matrix are not valid or are
            dimensionally inconsistent.
        """
        _valid_mean_covariance(mean_vector, covariance_matrix)
        self.n = len(mean_vector) // 2
        self._mean_vector = mean_vector.copy()
        self._covariance_matrix = covariance_matrix.copy()

    @classmethod
    def vacuum(cls, n: Integral) -> "GaussianCVState":
        """
        Construct an n-mode vacuum state.

        Parameters
        ----------
        n : Integral
            The number of modes. Must be strictly positive.

        Returns
        -------
        GaussianCVState
            The n-mode vacuum state.

        Raises
        ------
        TypeError
            If ``n`` is not an integer.
        ValueError
            If ``n`` is not strictly positive.
        """
        return cls(thermal_vacuum_mean(n), thermal_vacuum_covariance(n))

    @classmethod
    def thermal(cls, n: Integral, nbars: npt.NDArray[np.float64] | None = None) -> "GaussianCVState":
        """
        Construct an n-mode product thermal state.

        Parameters
        ----------
        n : Integral
            The number of modes. Must be strictly positive.
        nbars : numpy.ndarray or None, optional
            A one-dimensional array of thermal occupation numbers. If fewer
            than ``n`` values are provided, the remaining modes are assigned
            zero thermal occupation. If ``None``, all modes are vacuum.

        Returns
        -------
        GaussianCVState
            The n-mode thermal product state.

        Raises
        ------
        TypeError
            If ``n`` or ``nbars`` has an invalid type.
        ValueError
            If ``n`` is invalid or if ``nbars`` contains invalid values.
        """
        return cls(thermal_vacuum_mean(n), thermal_vacuum_covariance(n, nbars))

    @property
    def mean_vector(self) -> npt.NDArray[np.float64]:
        """
        Return a copy of the mean vector.

        Returns
        -------
        numpy.ndarray
            A copy of the mean vector in x-then-p ordering.
        """
        return self._mean_vector.copy()

    @property
    def covariance_matrix(self) -> npt.NDArray[np.float64]:
        """
        Return a copy of the covariance matrix.

        Returns
        -------
        numpy.ndarray
            A copy of the covariance matrix.
        """
        return self._covariance_matrix.copy()

    def single_mode_displacement(self, displacement:Complex, mode_id:Integral) -> "GaussianCVState":
        """
        Apply a single-mode displacement in place.
    
        This method applies a phase-space displacement to one mode of the state,
        updating the internal mean vector and leaving the covariance matrix
        unchanged. The state uses the x-then-p phase-space ordering
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        Parameters
        ----------
        displacement : Complex
            The complex displacement amplitude. Must be finite.
        mode_id : Integral
            The target mode index (1-based).
    
        Returns
        -------
        GaussianCVState
            The current state instance after in-place update.
    
        Raises
        ------
        TypeError
            If ``displacement`` or ``mode_id`` has invalid type.
        ValueError
            If ``displacement`` is not finite or if ``mode_id`` is out of bounds.
    
        Notes
        -----
        This method mutates the current object and returns ``self`` to support
        chained operations.
        """
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_1_mode_displacement((m0,c0), displacement, mode_id)
        return self
    
    def single_mode_squeeze(self, squeeze_tuple:tuple[Real,Real], mode_id:Integral):
        """
        Apply a single-mode squeezing unitary in place.
    
        This method applies a single-mode Gaussian squeezing transformation to
        one mode of the state, updating both the mean vector and covariance
        matrix in the x-then-p phase-space ordering.
    
        Parameters
        ----------
        squeeze_tuple : tuple of (Real, Real)
            The squeezing parameters ``(squeeze_magnitude, squeeze_angle)``.
        mode_id : Integral
            The target mode index (1-based).
    
        Returns
        -------
        GaussianCVState
            The current state instance after in-place update.
    
        Raises
        ------
        TypeError
            If the squeezing parameters or ``mode_id`` have invalid type.
        ValueError
            If the squeezing parameters are invalid or if ``mode_id`` is out of
            bounds.
    
        Notes
        -----
        This method mutates the current object and returns ``self`` to support
        chained operations.
        """
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_1_mode_squeeze_unitary((m0,c0), squeeze_tuple, mode_id)
        return self
    
    def two_mode_mix(self, coupling_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]) -> "GaussianCVState":
        """
        Apply a two-mode passive mixing unitary in place.
    
        This method applies a two-mode passive Gaussian unitary to the selected
        pair of modes, updating both the mean vector and covariance matrix.
        The selected subsystem is ordered as
    
            (x_i, x_j, p_i, p_j)
    
        in the global x-then-p convention
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        Parameters
        ----------
        coupling_tuple : tuple of (Real, Real)
            The mixing parameters ``(mixing_magnitude, mixing_angle)``.
        mode_ids : tuple of (Integral, Integral)
            The pair of target mode indices (1-based). The two indices must be
            distinct.
    
        Returns
        -------
        GaussianCVState
            The current state instance after in-place update.
    
        Raises
        ------
        TypeError
            If the coupling parameters or mode indices have invalid type.
        ValueError
            If the coupling parameters are invalid, if ``mode_ids`` does not have
            length 2, if the indices are not distinct, or if any mode index is
            out of bounds.
    
        Notes
        -----
        This method mutates the current object and returns ``self`` to support
        chained operations.
        """
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_2_mode_mix_unitary((m0,c0), coupling_tuple, mode_ids)
        return self

    def two_mode_squeeze(self, squeeze_tuple:tuple[Real,Real], mode_ids:tuple[Integral,Integral]) -> "GaussianCVState":
        """
        Apply a two-mode squeezing unitary in place.
    
        This method applies a two-mode active Gaussian unitary to the selected
        pair of modes, updating both the mean vector and covariance matrix.
        The selected subsystem is ordered as
    
            (x_i, x_j, p_i, p_j)
    
        in the global x-then-p convention
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        Parameters
        ----------
        squeeze_tuple : tuple of (Real, Real)
            The squeezing parameters ``(squeeze_magnitude, squeeze_angle)``.
        mode_ids : tuple of (Integral, Integral)
            The pair of target mode indices (1-based). The two indices must be
            distinct.
    
        Returns
        -------
        GaussianCVState
            The current state instance after in-place update.
    
        Raises
        ------
        TypeError
            If the squeezing parameters or mode indices have invalid type.
        ValueError
            If the squeezing parameters are invalid, if ``mode_ids`` does not have
            length 2, if the indices are not distinct, or if any mode index is
            out of bounds.
    
        Notes
        -----
        This method mutates the current object and returns ``self`` to support
        chained operations.
        """
        m0, c0 = self._mean_vector, self._covariance_matrix
        self._mean_vector, self._covariance_matrix = apply_2_mode_squeeze_unitary((m0,c0), squeeze_tuple, mode_ids)
        return self

    def single_mode_thermal_reset(self, nbar:Real, mode_id:Integral) -> "GaussianCVState":
        """
        Reset a single mode to an uncorrelated thermal state.
    
        This method replaces the selected mode with a thermal state of
        occupation ``nbar`` by removing all correlations involving that mode
        and setting its local quadrature variances to
    
            nbar + 1/2.
    
        In the global x-then-p phase-space ordering
    
            (x_1, ..., x_n, p_1, ..., p_n),
    
        a single mode is represented by the two noncontiguous indices
        corresponding to ``x_i`` and ``p_i``. This method therefore zeros the
        rows and columns associated with those indices and then sets the
        diagonal entries for that mode to ``nbar + 1/2``.
    
        Parameters
        ----------
        nbar : Real
            The thermal occupation number. Must be non-negative and finite.
        mode_id : Integral
            The target mode index (1-based).
    
        Returns
        -------
        GaussianCVState
            The current state instance after in-place update.
    
        Raises
        ------
        TypeError
            If ``nbar`` or ``mode_id`` has invalid type.
        ValueError
            If ``nbar`` is negative or not finite, or if ``mode_id`` is out of bounds.
    
        Notes
        -----
        This is a destructive, non-unitary reset. It removes all correlations
        involving the selected mode and leaves the mean vector unchanged.
    
        The state is modified in place and the method returns ``self`` to support
        chained operations.
        """
        _require_positive_integral_scalar(mode_id, "target mode id")
        _require_nonnegative_real_scalar(nbar, "thermal occupation")
        _valid_indices(self.n,(mode_id,))
        idx, idp = index_list(self.n,(mode_id,))

        self._covariance_matrix[idx, :] = 0.0
        self._covariance_matrix[:, idx] = 0.0
        self._covariance_matrix[idp, :] = 0.0
        self._covariance_matrix[:, idp] = 0.0
        self._covariance_matrix[idx, idx] = nbar + 0.5
        self._covariance_matrix[idp, idp] = nbar + 0.5
        return self

    def state_to_vector(self) -> npt.NDArray[np.float64]:
        """
        Convert the Gaussian state to a flattened vector representation.
    
        This method serializes the current state by concatenating the mean
        vector and the flattened covariance matrix into a one-dimensional array:
    
            [mean_vector, covariance_matrix.flatten()]
    
        where the covariance matrix is flattened in row-major (C) order.
    
        Returns
        -------
        numpy.ndarray
            A one-dimensional array of length 2n + (2n)^2 representing the state.
    
        Notes
        -----
        The state is assumed to be ordered in the x-then-p convention
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        This representation is useful for numerical solvers such as ``solve_ivp``
        or other vectorized workflows. Reconstruction can be performed using
        ``extract_mean_covariance``.
        """
        m0, c0 = self.mean_vector, self.covariance_matrix
        return compress_mean_covariance(m0, c0)

    def copy_state(self) -> "GaussianCVState":
        """
        Create a deep copy of the Gaussian state.
    
        This method returns a new ``GaussianCVState`` instance with copies of
        the mean vector and covariance matrix. The returned state is completely
        independent of the original.
    
        Returns
        -------
        GaussianCVState
            A deep copy of the current state.
    
        Notes
        -----
        Modifications to the returned state will not affect the original state.
        """
        return GaussianCVState(self._mean_vector, self._covariance_matrix)

    def plot_state(self, ax=None, n_std=2):
        """
        Plot single-mode phase-space marginals of the Gaussian state.
    
        This method visualizes each mode of the Gaussian state by plotting its
        marginal covariance ellipse in the local phase space
    
            (x_i, p_i),
    
        where the full state uses the global x-then-p ordering
    
            (x_1, ..., x_n, p_1, ..., p_n).
    
        For each mode, the plotted ellipse represents an ``n_std``-standard-
        deviation contour derived from the corresponding 2×2 marginal covariance
        matrix, and the mode mean is shown as a point.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot. If ``None``, a new figure and
            axes are created.
        n_std : Real, optional
            The number of standard deviations used to scale each covariance
            ellipse. Must be strictly positive. Default is 2.
    
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
    
        Raises
        ------
        TypeError
            If ``n_std`` is not a real scalar.
        ValueError
            If ``n_std`` is not finite or not strictly positive.
    
        Notes
        -----
        Each mode is plotted independently using its local marginal mean and
        covariance. Inter-mode correlations are not visualized by this method.
    
        The ellipse orientation is determined by the eigenvectors of the local
        covariance matrix, and the axis lengths are proportional to the square
        roots of its eigenvalues.
        """
        _require_positive_real_scalar(n_std, "number of standard deviations")
        if ax is None:
            fig, ax = plt.subplots()
            
        means =  [mean_subsystem(self.mean_vector, (i,)) for i in range(1,self.n+1)]
        covariances =  [covariance_subsystem(self.covariance_matrix, (i,)) for i in range(1,self.n+1)]
    
        for i in range(self.n):
            mean = means[i]
            covariance = covariances[i]
            eigvals, eigvecs = eigh(covariance)
            
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:,order]
    
            angle = np.degrees(np.arctan2(eigvecs[1,0],eigvecs[0,0]))
    
            width = 2 * n_std * np.sqrt(eigvals[0])
            height = 2 * n_std * np.sqrt(eigvals[1])

            color = plt.cm.tab10(i)
            ax.add_patch(Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                linewidth=2, 
                edgecolor=color
            ))
    
            ax.scatter(*mean, color=color, label=f"mode {i+1}")

        ax.legend()
        ax.set_xlabel("position")
        ax.set_ylabel("momentum")
        ax.set_aspect("equal")
        return ax

    
    
    
    