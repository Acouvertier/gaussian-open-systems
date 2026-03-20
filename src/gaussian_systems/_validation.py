from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric, eigvalsh, ishermitian

Array = npt.NDArray[np.generic]

_ALLOWED_COUPLING_TYPES = ("annihilation", "position", "momentum")

_ALLOWED_DECAY_TYPES = ("a", "x", "p", "ad")

def _require_type(value, expected_type, name:str) -> None:
    """
    Validate that a value is of an expected type.

    This function checks whether ``value`` is an instance of ``expected_type``.
    If not, a ``TypeError`` is raised with a message indicating the name of
    the parameter and the received type.

    Parameters
    ----------
    value : Any
        The object to validate.
    expected_type : type or tuple of types
        The required type(s) for ``value``. This is passed directly to
        ``isinstance`` and may be a single type or a tuple of types.
    name : str
        The name of the variable being validated. Used in the error message.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of ``expected_type``.
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type}, got {type(value)}.")

def _require_finite(value:Real|Array, name:str) -> None:
    """
    Validate that a value is finite.

    This function checks whether ``value`` is either a real scalar or a NumPy
    array and ensures that all entries are finite (i.e., not NaN or infinite).
    Scalars are validated using ``np.isfinite``, and arrays are validated using
    ``np.all(np.isfinite(...))``.

    Parameters
    ----------
    value : Real or numpy.ndarray
        The value to validate. Must be a real scalar or a NumPy array.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    ValueError
        If ``value`` is a scalar or array containing non-finite elements.
    TypeError
        If ``value`` is not a real scalar or NumPy array.
    """
    if isinstance(value, Real):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar. Got {value}")
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite array. Got {value}")
    else:
        raise TypeError(f"{name} must be a real scalar or numpy array. Got {value}")

def _require_real_scalar(value: Real, name: str) -> None:
    """
    Validate that a value is a finite real scalar.

    This function enforces that ``value`` is an instance of ``Real`` (including
    integers) and that it is finite (i.e., not NaN or infinite). It delegates
    type and finiteness checks to ``_require_type`` and ``_require_finite``.

    Parameters
    ----------
    value : Real
        The value to validate. Must be a real scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not a real scalar.
    ValueError
        If ``value`` is not finite.
    """
    _require_type(value, Real, name)
    _require_finite(value, name)

def _require_integral_scalar(value: Integral, name: str) -> None:
    """
    Validate that a value is an integral scalar.

    This function checks whether ``value`` is an instance of ``Integral``
    (e.g., ``int`` or NumPy integer types). No additional checks, such as
    finiteness or bounds, are performed.

    Parameters
    ----------
    value : Integral
        The value to validate. Must be an integer-like scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of ``Integral``.
    """
    _require_type(value, Integral, name)

def _require_nonnegative_real_scalar(value:Real, name: str) -> None:
    """
    Validate that a value is a finite, non-negative real scalar.

    This function enforces that ``value`` is a real scalar (including integers),
    is finite (i.e., not NaN or infinite), and satisfies the constraint
    ``value >= 0``. Type and finiteness checks are delegated to
    ``_require_real_scalar``.

    Parameters
    ----------
    value : Real
        The value to validate. Must be a finite real scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not a real scalar.
    ValueError
        If ``value`` is not finite or is negative.
    """
    _require_real_scalar(value, name)
    sign_criteria = value >= 0
    if not sign_criteria:
        raise ValueError(f"{name} must be non-negative real scalar. Got {value}.")

def _require_positive_real_scalar(value:Real, name: str) -> None:
    """
    Validate that a value is a finite, strictly positive real scalar.

    This function enforces that ``value`` is a real scalar (including integers),
    is finite (i.e., not NaN or infinite), and satisfies the constraint
    ``value > 0``. Type and finiteness checks are delegated to
    ``_require_real_scalar``.

    Parameters
    ----------
    value : Real
        The value to validate. Must be a finite real scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not a real scalar.
    ValueError
        If ``value`` is not finite or is not strictly positive.
    """
    _require_real_scalar(value, name)
    sign_criteria = value > 0
    if not sign_criteria:
        raise ValueError(f"{name} must be positive real scalar. Got {value}.")

def _require_nonnegative_integral_scalar(value:Integral, name: str) -> None:
    """
    Validate that a value is a non-negative integral scalar.

    This function enforces that ``value`` is an instance of ``Integral``
    (e.g., ``int`` or NumPy integer types) and satisfies the constraint
    ``value >= 0``.

    Parameters
    ----------
    value : Integral
        The value to validate. Must be an integer-like scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of ``Integral``.
    ValueError
        If ``value`` is negative.
    """
    _require_integral_scalar(value, name)
    sign_criteria = value >= 0
    if not sign_criteria:
        raise ValueError(f"{name} must be non-negative integer scalar. Got {value}.")

def _require_positive_integral_scalar(value:Integral, name: str) -> None:
    """
    Validate that a value is a strictly positive integral scalar.

    This function enforces that ``value`` is an instance of ``Integral``
    (e.g., ``int`` or NumPy integer types) and satisfies the constraint
    ``value > 0``.

    Parameters
    ----------
    value : Integral
        The value to validate. Must be an integer-like scalar.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of ``Integral``.
    ValueError
        If ``value`` is not strictly positive.
    """
    _require_integral_scalar(value, name)
    sign_criteria = value > 0
    if not sign_criteria:
        raise ValueError(f"{name} must be positive integer scalar. Got {value}.")

def _require_tuple_length(value: tuple, length: Integral, name: str) -> None:
    """
    Validate that a value is a tuple of a specified length.

    This function checks that ``value`` is a tuple and that its length
    matches the expected ``length``.

    Parameters
    ----------
    value : tuple
        The tuple to validate.
    length : Integral
        The required length of the tuple.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is not a tuple.
    ValueError
        If the length of ``value`` does not match ``length``.
    """
    _require_type(value, tuple, name)
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {value}")

def _require_ndarray(arr: Array, name: str) -> None:
    """
    Validate that a value is a NumPy array.

    This function checks whether ``arr`` is an instance of ``numpy.ndarray``.
    No validation of shape, dtype, or contents is performed.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not an instance of ``numpy.ndarray``.
    """
    _require_type(arr, np.ndarray, name)

def _require_ndim(arr: Array, ndim: Integral, name: str) -> None:
    """
    Validate that a NumPy array has a specified number of dimensions.

    This function checks that ``arr`` is a NumPy array and that its number
    of dimensions (``arr.ndim``) matches the expected ``ndim``.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    ndim : Integral
        The required number of dimensions.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr.ndim`` does not equal ``ndim``.
    """
    _require_ndarray(arr,  name)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.ndim}D.")

def _require_real_array(arr: Array, name: str) -> None:
    """
    Validate that an array is real-valued and finite.

    This function enforces that ``arr`` is a NumPy array, that all elements
    are finite (i.e., not NaN or infinite), and that the array is real-valued
    (i.e., not of complex dtype).

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` contains non-finite values or is not real-valued.
    """
    _require_ndarray(arr, name)
    _require_finite(arr, name)
    if not np.isrealobj(arr):
        raise ValueError(f"{name} must be real-valued.")
    
def _require_real_vector(arr:Array, name:str) -> None:
    """
    Validate that an array is a real-valued, finite 1D vector.

    This function enforces that ``arr`` is a NumPy array with exactly one
    dimension (i.e., a vector), that all elements are finite, and that the
    array is real-valued.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` is not one-dimensional, contains non-finite values,
        or is not real-valued.
    """
    _require_ndim(arr, 1, name)
    _require_real_array(arr, name)

def _require_nonnegative_real_vector(arr: Array, name:str) -> None:
    """
    Validate that an array is a real-valued, finite 1D vector with non-negative entries.

    This function enforces that ``arr`` is a one-dimensional NumPy array,
    that all elements are finite and real-valued, and that every entry
    satisfies ``arr[i] >= 0``.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` is not one-dimensional, contains non-finite values,
        is not real-valued, or contains negative entries.
    """
    _require_real_vector(arr, name)
    sign_criteria = np.all(arr >= 0)
    if not sign_criteria:
        raise ValueError(f"{name} must contain only non-negative real values. Got {arr}.")

def _require_positive_real_vector(arr: Array, name:str) -> None:
    """
    Validate that an array is a real-valued, finite 1D vector with strictly positive entries.

    This function enforces that ``arr`` is a one-dimensional NumPy array,
    that all elements are finite and real-valued, and that every entry
    satisfies ``arr[i] > 0``.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` is not one-dimensional, contains non-finite values,
        is not real-valued, or contains non-positive entries.
    """
    _require_real_vector(arr, name)
    sign_criteria = np.all(arr > 0)
    if not sign_criteria:
        raise ValueError(f"{name} must contain only positive real values. Got {arr}.")

def _require_square_matrix(matrix: Array, name: str) -> None:
    """
    Validate that an array is a square 2D matrix.

    This function enforces that ``matrix`` is a two-dimensional NumPy array
    and that it has equal numbers of rows and columns.

    Parameters
    ----------
    matrix : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``matrix`` is not a NumPy array.
    ValueError
        If ``matrix`` is not two-dimensional or is not square.
    """
    _require_ndim(matrix, 2, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")

def _require_even_vector_length(arr: Array, name: str) -> None:
    """
    Validate that an array is a 1D vector with even length.

    This function enforces that ``arr`` is a one-dimensional NumPy array
    and that its length (``arr.shape[0]``) is an even integer.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` is not one-dimensional or does not have even length.
    """
    _require_ndim(arr, 1, name)
    if arr.shape[0] % 2 != 0:
        raise ValueError(f"{name} must have even dimension, got shape {arr.shape}")

def _require_even_matrix_dimension(arr: Array, name: str) -> None:
    """
    Validate that an array is a 2D matrix with even dimensions.

    This function enforces that ``arr`` is a two-dimensional NumPy array
    and that each of its dimensions is an even integer.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to validate.
    name : str
        The name of the variable being validated. Used in error messages.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array.
    ValueError
        If ``arr`` is not two-dimensional or if any dimension is odd.
    """
    _require_ndim(arr, 2, name)
    for dim in arr.shape:
        if dim % 2 != 0:
            raise ValueError(f"{name} must have even dimension, got shape {arr.shape}")

def _require_symmetric(matrix: Array, name: str, *,atol:float = 1e-8, rtol:float = 1e-8) -> None:
    """
    Validate that a matrix is approximately symmetric.

    This function enforces that ``matrix`` is a square NumPy array and that it
    is symmetric within specified absolute and relative tolerances, as determined
    by ``scipy.linalg.issymmetric``.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to validate.
    name : str
        The name of the variable being validated. Used in error messages.
    atol : float, optional
        Absolute tolerance for symmetry comparison. Default is 1e-8.
    rtol : float, optional
        Relative tolerance for symmetry comparison. Default is 1e-8.

    Raises
    ------
    TypeError
        If ``matrix`` is not a NumPy array.
    ValueError
        If ``matrix`` is not square or is not approximately symmetric
        within the specified tolerances.
    """
    _require_square_matrix(matrix, name)
    if not issymmetric(matrix, atol=atol, rtol=rtol):
        raise ValueError(f"{name} must be approximately symmetric. Got {matrix}")

def _require_positive_semidefinite(matrix: Array, name: str, *,atol:float = 1e-8, rtol:float = 1e-8) -> None:
    """
    Validate that a matrix is approximately positive semidefinite.

    This function enforces that ``matrix`` is square and approximately
    Hermitian, then checks that its eigenvalues are non-negative up to an
    absolute tolerance. The Hermiticity check is performed using
    ``scipy.linalg.ishermitian``, and the eigenvalue spectrum is computed
    using ``scipy.linalg.eigvalsh``.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to validate.
    name : str
        The name of the variable being validated. Used in error messages.
    atol : float, optional
        Absolute tolerance used for both the Hermiticity check and the
        positive-semidefinite test. Eigenvalues greater than or equal to
        ``-atol`` are accepted as numerically non-negative. Default is 1e-8.
    rtol : float, optional
        Relative tolerance used in the Hermiticity check. Default is 1e-8.

    Raises
    ------
    TypeError
        If ``matrix`` is not a NumPy array.
    ValueError
        If ``matrix`` is not square, is not approximately Hermitian, or has
        eigenvalues smaller than ``-atol``.

    Notes
    -----
    This validator applies to both real symmetric and complex Hermitian
    matrices, since real symmetric matrices are a special case of Hermitian
    matrices. This function does not check the quantum uncertainty relation.
    """
    _require_square_matrix(matrix, name)
    if not ishermitian(matrix, atol=atol, rtol=rtol):
        raise ValueError(f"{name} must be approximately Hermitian. Got {matrix}")
    eig_spectrum = eigvalsh(matrix)
    if not np.all(eig_spectrum >= -atol):
        raise ValueError(f"{name} must have non-negative eigenvalues. Got {eig_spectrum}")
    

def _valid_mode_number(n: Integral) -> None:
    """
    Validate that the number of modes is a strictly positive integer.

    This function enforces that ``n`` is an integer-like scalar (e.g., ``int`` or
    NumPy integer type) and that it satisfies ``n > 0``.

    Parameters
    ----------
    n : Integral
        The number of modes to validate.

    Raises
    ------
    TypeError
        If ``n`` is not an instance of ``Integral``.
    ValueError
        If ``n`` is not strictly positive.
    """
    _require_positive_integral_scalar(n, 'number of modes')


def _valid_indices(n:Integral, indices:tuple[Integral ,...]) -> None:
    """
    Validate subsystem indices for a system with n modes.

    This function enforces that ``indices`` is a tuple of integer-valued
    indices and that each index lies in the inclusive range ``[1, n]``.
    The number of modes ``n`` must be a strictly positive integer.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    indices : tuple of Integral
        The subsystem indices to validate.

    Raises
    ------
    TypeError
        If ``n`` is not a positive integer, if ``indices`` is not a tuple,
        or if any element of ``indices`` is not an integer.
    ValueError
        If any index in ``indices`` lies outside the range ``[1, n]``.
    """
    name = 'subsystem indices'
    _valid_mode_number(n)
    _require_type(indices, tuple, name)
    if not all(isinstance(i, Integral) for i in indices):
        raise TypeError(f"{name} must be integer valued. Got {indices}")
    if not all(1 <= i <= n for i in indices):
        raise ValueError(f"{name} must be between 1 and {n} inclusive. Got {indices}")

def _valid_mean_vector(mean_vector:Array) -> None:
    """
    Validate a Gaussian mean vector.

    This function enforces that ``mean_vector`` is a one-dimensional NumPy array,
    that all elements are finite and real-valued, and that its length is even.
    The even-length requirement is consistent with phase-space representations
    where variables occur in conjugate pairs.

    Parameters
    ----------
    mean_vector : numpy.ndarray
        The mean vector to validate.

    Raises
    ------
    TypeError
        If ``mean_vector`` is not a NumPy array.
    ValueError
        If ``mean_vector`` is not one-dimensional, contains non-finite values,
        is not real-valued, or does not have even length.
    """
    name = 'mean vector'
    _require_real_vector(mean_vector, name)
    _require_even_vector_length(mean_vector, name)

def _valid_covariance_matrix(covariance_matrix:Array) -> None:
    """
    Validate a classically admissible covariance matrix.

    This function enforces that ``covariance_matrix`` is a NumPy array,
    that all elements are finite and real-valued, that it is two-dimensional
    with even dimensions, and that it is positive semidefinite (within
    numerical tolerance).

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        The covariance matrix to validate.

    Raises
    ------
    TypeError
        If ``covariance_matrix`` is not a NumPy array.
    ValueError
        If ``covariance_matrix`` contains non-finite values, is not real-valued,
        is not two-dimensional, does not have even dimensions, or is not
        positive semidefinite.

    Notes
    -----
    This function checks classical covariance validity only. It does not enforce
    the quantum uncertainty relation required for physical Gaussian states.
    """
    name = 'covariance matrix'
    _require_real_array(covariance_matrix, name)
    _require_even_matrix_dimension(covariance_matrix, name)
    _require_positive_semidefinite(covariance_matrix, name)

def _valid_mean_covariance(mean_vector:Array,covariance_matrix:Array) -> None:
    """
    Validate consistency between a mean vector and covariance matrix.

    This function enforces that ``mean_vector`` is a valid real-valued,
    finite 1D vector with even length, that ``covariance_matrix`` is a
    valid real-valued, finite, even-dimensional positive semidefinite
    matrix, and that their dimensions are consistent.

    Parameters
    ----------
    mean_vector : numpy.ndarray
        The mean vector to validate.
    covariance_matrix : numpy.ndarray
        The covariance matrix to validate.

    Raises
    ------
    TypeError
        If either input is not a NumPy array.
    ValueError
        If ``mean_vector`` or ``covariance_matrix`` fails their respective
        validation checks, or if their dimensions are inconsistent.

    Notes
    -----
    The dimension consistency condition requires that
    ``mean_vector.shape[0] == covariance_matrix.shape[0]``.
    """
    _valid_mean_vector(mean_vector)
    _valid_covariance_matrix(covariance_matrix)
    if mean_vector.shape[0] != covariance_matrix.shape[0]:
        raise ValueError(f"mean vector and covariance matrix dimensions must match, got mean dimension of {mean_vector.shape[0]} and covariance dimension of {covariance_matrix.shape[0]}.")

def _valid_nbars_array(nbars:Array) -> None:
    """
    Validate an array of thermal occupation numbers.

    This function enforces that ``nbars`` is a one-dimensional NumPy array,
    that all elements are finite and real-valued, and that every entry is
    non-negative.

    Parameters
    ----------
    nbars : numpy.ndarray
        The array of thermal occupation numbers to validate.

    Raises
    ------
    TypeError
        If ``nbars`` is not a NumPy array.
    ValueError
        If ``nbars`` is not one-dimensional, contains non-finite values,
        is not real-valued, or contains negative entries.

    Notes
    -----
    This function does not enforce consistency with a specific number of modes.
    """
    name = 'thermal occupations'
    _require_nonnegative_real_vector(nbars,name)

def _valid_mean_covariance_tuple(state: tuple[Array, Array]) -> None:
    """
    Validate a tuple representing a Gaussian state (mean, covariance).

    This function enforces that ``state`` is a tuple of length 2, where the
    first element is a valid mean vector and the second element is a valid
    covariance matrix. It also ensures that the mean vector and covariance
    matrix are dimensionally consistent.

    Parameters
    ----------
    state : tuple of (numpy.ndarray, numpy.ndarray)
        The Gaussian state to validate, given as (mean_vector, covariance_matrix).

    Raises
    ------
    TypeError
        If ``state`` is not a tuple or if its elements are not NumPy arrays.
    ValueError
        If ``state`` does not have length 2, or if the mean vector and
        covariance matrix fail their respective validation checks or are
        dimensionally inconsistent.
    """
    _require_tuple_length(state, 2, 'given state')
    mean_vector, covariance_matrix = state
    _valid_mean_covariance(mean_vector, covariance_matrix)

def _valid_fidelity_input(state1:tuple[Array, Array], state2:tuple[Array, Array]) -> None:
    """
    Validate inputs for Gaussian state fidelity calculations.

    This function enforces that both ``state1`` and ``state2`` are valid
    Gaussian states represented as (mean_vector, covariance_matrix) tuples,
    and that their dimensions are compatible.

    Parameters
    ----------
    state1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first Gaussian state, given as (mean_vector, covariance_matrix).
    state2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second Gaussian state, given as (mean_vector, covariance_matrix).

    Raises
    ------
    TypeError
        If either state is not a valid tuple of NumPy arrays.
    ValueError
        If either state fails mean/covariance validation or if the two states
        have mismatched dimensions.

    Notes
    -----
    Dimension compatibility is determined by requiring
    ``mean1.shape[0] == mean2.shape[0]``.
    """
    _valid_mean_covariance_tuple(state1)
    _valid_mean_covariance_tuple(state2)
    
    mean1, cov1 = state1
    mean2, cov2 = state2
   
    if mean1.shape[0] != mean2.shape[0]:
        raise ValueError(f"states must have matching dimensions, got {mean1.shape[0]} and {mean2.shape[0]}")

def _valid_t_eval(t_eval:Array) -> None:
    """
    Validate a time evaluation grid.

    This function enforces that ``t_eval`` is a one-dimensional NumPy array,
    that all elements are finite, real-valued, and non-negative, that the
    array is non-empty, and that it is strictly increasing.

    Parameters
    ----------
    t_eval : numpy.ndarray
        The time grid to validate.

    Raises
    ------
    TypeError
        If ``t_eval`` is not a NumPy array.
    ValueError
        If ``t_eval`` is not one-dimensional, contains non-finite values,
        is not real-valued, contains negative values, is empty, or is not
        strictly increasing.

    Notes
    -----
    Strict monotonicity is enforced via ``t_eval[i+1] > t_eval[i]`` for all i.
    """
    name = "time grid"
    _require_nonnegative_real_vector(t_eval, name)
    if t_eval.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError(f"{name} must be strictly increasing. Got {t_eval}.")

def _valid_parameter_tuple(parameter_tuple: tuple[Real,Real], name1: str, name2: str) -> None:
    """
    Validate a tuple of two real scalar parameters.

    This function enforces that ``parameter_tuple`` is a tuple of length 2,
    and that each element is a finite real scalar. The two elements are
    validated independently with distinct parameter names for error reporting.

    Parameters
    ----------
    parameter_tuple : tuple of (Real, Real)
        The parameter tuple to validate.
    name1 : str
        The name of the first parameter. Used in error messages.
    name2 : str
        The name of the second parameter. Used in error messages.

    Raises
    ------
    TypeError
        If ``parameter_tuple`` is not a tuple, or if either element is not a
        real scalar.
    ValueError
        If either element is not finite.
    """
    _require_tuple_length(parameter_tuple, 2, 'state/system parameters')
    
    param1, param2 = parameter_tuple
    _require_real_scalar(param1, name1)
    _require_real_scalar(param2, name2)

def _valid_frequency_array(frequency_array: Array) -> None:
    """
    Validate an array of mode frequencies.

    This function enforces that ``frequency_array`` is a one-dimensional
    NumPy array and that all elements are finite and real-valued.

    Parameters
    ----------
    frequency_array : numpy.ndarray
        The array of mode frequencies to validate.

    Raises
    ------
    TypeError
        If ``frequency_array`` is not a NumPy array.
    ValueError
        If ``frequency_array`` is not one-dimensional, contains non-finite
        values, or is not real-valued.

    Notes
    -----
    No sign constraint is enforced. Negative frequencies are permitted to
    support representations in rotated or interaction frames. Users are
    responsible for ensuring physical consistency in such cases
    """
    _require_real_vector(frequency_array, 'mode frequencies')

def _valid_hamiltonian_parameter(coefficient: Real) -> None:
    """
    Validate a Hamiltonian parameter (real scalar coefficient).

    This function enforces that ``coefficient`` is a finite real scalar.
    No additional constraints (e.g., positivity or bounds) are imposed.

    Parameters
    ----------
    coefficient : Real
        The Hamiltonian parameter to validate.

    Raises
    ------
    TypeError
        If ``coefficient`` is not a real scalar.
    ValueError
        If ``coefficient`` is not finite.

    Notes
    -----
    The parameter is labeled as a "coupling coefficient" in error messages,
    but no assumption is made about its sign or magnitude.
    """
    _require_real_scalar(coefficient, 'coupling coefficient')

def _valid_subsystem(n:Integral, mode_id: Integral|tuple[Integral,...]) -> None:
    """
    Validate a subsystem specification for an n-mode system.

    This function enforces that ``mode_id`` specifies a valid subsystem,
    represented either as a single integer mode index or as a tuple of
    integer mode indices. All indices must lie in the inclusive range
    ``[1, n]``.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    mode_id : Integral or tuple of Integral
        The subsystem specification to validate. This may be either a single
        mode index or a tuple of mode indices.

    Raises
    ------
    TypeError
        If ``n`` is not a positive integer, if ``mode_id`` is neither an
        integer nor a tuple, or if any tuple entry is not an integer.
    ValueError
        If any subsystem index lies outside the range ``[1, n]``.

    Notes
    -----
    This function does not enforce uniqueness or ordering of subsystem indices.
    Duplicate indices are permitted unless ruled out elsewhere.
    """
    _valid_mode_number(n)
    if not isinstance(mode_id, tuple) and not isinstance(mode_id, Integral):
        raise TypeError(f"subsystem must be either a single integer mode index or a tuple of integer mode indices, got {type(mode_id)}.")
    if isinstance(mode_id, tuple):
        _valid_indices(n, mode_id)
    else:
        _valid_indices(n,(mode_id,))

def _valid_term_inputs(n:Integral, mode_id: tuple[Integral,Integral], coefficient: Real) -> None:
    """
    Validate inputs for a two-mode Hamiltonian term.

    This function enforces that ``coefficient`` is a finite real scalar,
    that ``mode_id`` is a tuple of length 2, and that both indices define
    a valid subsystem within an ``n``-mode system.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    mode_id : tuple of (Integral, Integral)
        The pair of mode indices defining the coupling.
    coefficient : Real
        The coupling coefficient associated with the term.

    Raises
    ------
    TypeError
        If any input has an invalid type.
    ValueError
        If ``mode_id`` does not have length 2, if indices are out of bounds,
        or if ``coefficient`` is not finite.

    Notes
    -----
    This function does not enforce that the two indices are distinct or ordered.
    """
    _valid_hamiltonian_parameter(coefficient)
    _require_tuple_length(mode_id, 2, 'coupling indices')
    _valid_subsystem(n, mode_id)
    
    
def _valid_hamiltonian_matrix(hamiltonian: Array) -> None:
    """
    Validate a quadratic Hamiltonian matrix.

    This function enforces that ``hamiltonian`` is a NumPy array, that all
    elements are finite and real-valued, that it is two-dimensional with
    even dimensions, and that it is approximately symmetric.

    Parameters
    ----------
    hamiltonian : numpy.ndarray
        The Hamiltonian matrix to validate.

    Raises
    ------
    TypeError
        If ``hamiltonian`` is not a NumPy array.
    ValueError
        If ``hamiltonian`` contains non-finite values, is not real-valued,
        is not two-dimensional, does not have even dimensions, or is not
        approximately symmetric.

    Notes
    -----
    No positive semidefiniteness constraint is imposed. Hamiltonian matrices
    may have indefinite spectra.
    """
    name = 'hamiltonian'
    _require_real_array(hamiltonian, name)
    _require_even_matrix_dimension(hamiltonian, name)
    _require_symmetric(hamiltonian, name)

def _valid_lindblad_gram_matrix(M: Array, *,atol:float=1e-8) -> None:
    """
    Validate a Lindblad Gram matrix.

    This function enforces that ``M`` is a finite, square, even-dimensional
    array and then checks that its Hermitian part is positive semidefinite.
    The Hermitian part is constructed as ``(M + M.conj().T) / 2`` before
    performing the positive-semidefinite check.

    Parameters
    ----------
    M : numpy.ndarray
        The Lindblad Gram matrix to validate.
    atol : float, optional
        Absolute tolerance for the positive-semidefinite check. Default is 1e-8.

    Raises
    ------
    TypeError
        If ``M`` is not a NumPy array.
    ValueError
        If ``M`` contains non-finite values, is not two-dimensional, is not
        square, does not have even dimensions, or if its Hermitian part is not
        positive semidefinite.

    Notes
    -----
    This function does not require the input matrix itself to be exactly
    Hermitian. Instead, validation is performed on the Hermitian part, which
    is often the physically relevant component in Lindblad-type constructions.
    """
    name = 'Lindblad Gram matrix'
    _require_finite(M, name)
    _require_even_matrix_dimension(M, name)

    M_herm = (M + M.conj().T)/2

    _require_positive_semidefinite(M_herm, name, atol=atol)

def _valid_system(ham: Array, M: Array) -> None:
    """
    Validate a Gaussian system specification.

    This function enforces that ``ham`` is a valid Hamiltonian matrix,
    that ``M`` is a valid Lindblad Gram matrix, and that both matrices
    have matching dimensions.

    Parameters
    ----------
    ham : numpy.ndarray
        The Hamiltonian matrix to validate.
    M : numpy.ndarray
        The Lindblad Gram matrix to validate.

    Raises
    ------
    TypeError
        If either input is not a NumPy array.
    ValueError
        If ``ham`` or ``M`` fails their respective validation checks, or
        if their shapes do not match.

    Notes
    -----
    Dimensional compatibility is enforced via ``ham.shape == M.shape``.
    """
    _valid_hamiltonian_matrix(ham)
    _valid_lindblad_gram_matrix(M)
    if ham.shape != M.shape:
        raise ValueError(f"hamiltonian and Lindblad Gram matrix must have matching shapes, got {ham.shape} and {M.shape}.")

def _valid_single_pole_input(n:Integral, subsystem:Integral|tuple[Integral,...], coupling_types:tuple[str,...], memory_rate:Real, env_freq:Real, thermal_occupation:Real) -> None:
    """
    Validate inputs for a single-pole environment model.

    This function enforces that system size, subsystem specification,
    coupling types, and environment parameters are consistent and
    well-formed.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    subsystem : Integral or tuple of Integral
        The subsystem indices to which the environment couples.
    coupling_types : tuple of str
        The types of coupling for each mode in the subsystem. Must have
        the same length as the number of coupled modes.
    memory_rate : Real
        The environmental memory rate. Must be strictly positive.
    env_freq : Real
        The environmental frequency. Must be a finite real scalar.
    thermal_occupation : Real
        The thermal occupation number. Must be non-negative.

    Raises
    ------
    TypeError
        If any input has an invalid type.
    ValueError
        If any parameter violates its constraints, if ``subsystem`` is invalid,
        if ``coupling_types`` has incorrect length, or if any coupling type is
        not in the allowed set.

    Notes
    -----
    The allowed coupling types are defined by ``_ALLOWED_COUPLING_TYPES``.
    No sign constraint is imposed on ``env_freq`` to allow for rotated-frame
    representations.
    """
    _require_nonnegative_real_scalar(thermal_occupation, 'thermal occupation')
    _require_real_scalar(env_freq, 'environmental frequency')
    _require_positive_real_scalar(memory_rate, 'memory rate')
    _valid_subsystem(n, subsystem)
    if isinstance(subsystem,tuple):
        coupled_mode_count = len(subsystem)
    else:
        coupled_mode_count = 1
    _require_tuple_length(coupling_types, coupled_mode_count, 'coupling types')
    
    if not all(elt in _ALLOWED_COUPLING_TYPES for elt in coupling_types):
        raise ValueError(f"pseudo-coupling types must contain either 'annihilation', 'position', or, 'momentum'. Got {coupling_types}")

def _valid_decay_element(decay_element:tuple[str,Integral,Real]) -> None:
    """
    Validate a decay element specification.

    This function enforces that ``decay_element`` is a tuple of length 3
    representing a decay process, given as
    ``(decay_type, mode_id, decay_rate)``.

    Parameters
    ----------
    decay_element : tuple of (str, Integral, Real)
        The decay element to validate.

    Raises
    ------
    TypeError
        If ``decay_element`` is not a tuple of length 3, if ``decay_type``
        is not a string, or if ``mode_id`` is not an integer.
    ValueError
        If ``mode_id`` is not strictly positive, if ``decay_rate`` is
        negative, or if ``decay_type`` is not in the allowed set.

    Notes
    -----
    The allowed decay types are defined by ``_ALLOWED_DECAY_TYPES``.
    This function does not check consistency with a specific system size.
    """
    _require_tuple_length(decay_element, 3, 'decay code')
    decay_str, decay_int, decay_rate = decay_element
    _require_type(decay_str, str, 'decay type')
    _require_positive_integral_scalar(decay_int, 'decaying mode id')
    _require_nonnegative_real_scalar(decay_rate, 'decay rate')
    if decay_str not in _ALLOWED_DECAY_TYPES:
        raise ValueError(f"decay element must contain a string code for the decay type (either 'a', 'ad', 'x', or 'p'). Got {decay_str}")