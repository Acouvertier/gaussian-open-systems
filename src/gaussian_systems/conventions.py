import math
import numpy as np
import numpy.typing as npt
from numbers import Real, Integral 
from scipy.linalg import issymmetric, eigvals

from ._validation import _valid_indices, _valid_mean_covariance, _valid_mode_number, _require_square_matrix, _valid_mean_vector, _valid_covariance_matrix, _require_real_scalar, _require_real_vector, _require_positive_real_scalar, _require_finite

def _x_subsystem(n:Integral , indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    """
    Map mode indices to x-quadrature indices in phase space.

    This function converts a tuple of mode indices (1-based) into the
    corresponding zero-based indices of the position quadratures in the
    x-then-p phase-space ordering.

    For an n-mode system, the phase-space vector is ordered as
    (x_1, ..., x_n, p_1, ..., p_n). The x-quadrature indices for modes
    i are given by i - 1.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    indices : tuple of Integral
        The mode indices (1-based) to extract.

    Returns
    -------
    numpy.ndarray of int
        The zero-based indices corresponding to the selected x-quadratures.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If any index is outside the range [1, n].

    Notes
    -----
    The ordering of the returned indices matches the input order.
    No sorting or deduplication is performed.
    """
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1).astype(int)
        
def _p_subsystem(n:Integral, indices: tuple[Integral , ...]) -> npt.NDArray[int]:
    """
    Map mode indices to p-quadrature indices in phase space.

    This function converts a tuple of mode indices (1-based) into the
    corresponding zero-based indices of the momentum quadratures in the
    x-then-p phase-space ordering.

    For an n-mode system, the phase-space vector is ordered as
    (x_1, ..., x_n, p_1, ..., p_n). The p-quadrature indices for modes
    i are given by (i - 1 + n).

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    indices : tuple of Integral
        The mode indices (1-based) to extract.

    Returns
    -------
    numpy.ndarray of int
        The zero-based indices corresponding to the selected p-quadratures.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If any index is outside the range [1, n].

    Notes
    -----
    The ordering of the returned indices matches the input order.
    No sorting or deduplication is performed.
    """
    _valid_indices(n, indices)
        
    return (np.array(indices) - 1 + n).astype(int)

"""Public"""
    
def index_list(n:Integral, indices: tuple[Integral, ...]) -> npt.NDArray[int]:
    """
    Construct phase-space indices for a subsystem in x-then-p ordering.

    This function returns the indices corresponding to a subsystem defined
    by ``indices`` in an n-mode phase-space vector ordered as
    $(x_1, ..., x_n, p_1, ..., p_n)$.

    For a subsystem with modes $(i_1, ..., i_k)$, the returned indices correspond to
    $(x_{i_1}, ..., x_{i_k}, p_{i_1}, ..., p_{i_k})$,
    preserving the input ordering.

    Parameters
    ----------
    n : Integral
        The total number of modes in the system. Must be strictly positive.
    indices : tuple of Integral
        The mode indices (1-based) defining the subsystem.

    Returns
    -------
    numpy.ndarray of int
        The zero-based indices of the subsystem in phase space, ordered as
        x-components followed by p-components.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If any index is outside the range [1, n].

    Notes
    -----
    The returned ordering is (x-subsystem, p-subsystem), not interleaved.
    No sorting or deduplication of indices is performed.
    """
    x_idx = _x_subsystem(n,indices)
    p_idx = _p_subsystem(n,indices)
    final_idx = np.concatenate((x_idx,p_idx),axis=0).astype(int)
    return final_idx

def symmetrize_matrix(matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Return the symmetric (Hermitian) part of a matrix.

    This function maps a square matrix ``matrix`` to its self-adjoint part:
    ``(matrix + matrix.conj().T) / 2``.

    For real-valued matrices, this reduces to standard symmetrization:
    ``(A + A.T) / 2``. For complex-valued matrices, this produces a Hermitian
    matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix. Must be square and contain only finite values.

    Returns
    -------
    numpy.ndarray
        The symmetric (Hermitian) part of the input matrix.

    Raises
    ------
    TypeError
        If ``matrix`` is not a NumPy array.
    ValueError
        If ``matrix`` is not square or contains non-finite values.

    Notes
    -----
    The output is always self-adjoint (Hermitian).
    """
    _require_square_matrix(matrix, "provided matrix")
    _require_finite(matrix, "provided matrix")
    return (matrix + matrix.conj().T)/2

def rotation_matrix(theta: Real) -> npt.NDArray[np.float64]:
    """
    Construct a single-mode phase-space rotation matrix.

    This function returns the 2×2 rotation matrix acting on a single mode
    in phase space. In the (x, p) quadrature basis, the transformation is

        (x, p) → R(theta) (x, p),

    where
        R(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]].

    Parameters
    ----------
    theta : Real
        The rotation angle in radians. Must be finite.

    Returns
    -------
    numpy.ndarray of shape (2, 2)
        The rotation matrix.

    Raises
    ------
    TypeError
        If ``theta`` is not a real scalar.
    ValueError
        If ``theta`` is not finite.

    Notes
    -----
    This matrix represents a symplectic transformation for a single mode
    in the x-then-p convention.
    """
    _require_real_scalar(theta, "rotation angle")
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta),   np.cos(theta)]
    ])

def symplectic_matrix(n: Integral) -> npt.NDArray[np.float64]:
    """
    Construct the canonical symplectic form for an n-mode system.

    This function returns the 2n × 2n symplectic matrix Ω defined by

        Ω = [[0,  I],
             [-I, 0]],

    where I is the n × n identity matrix.

    In the x-then-p phase-space ordering
    (x_1, ..., x_n, p_1, ..., p_n), this matrix defines the canonical
    commutation relations:

        [R_i, R_j] = i Ω_{ij},

    where R = (x_1, ..., x_n, p_1, ..., p_n).

    Parameters
    ----------
    n : Integral
        The number of modes. Must be strictly positive.

    Returns
    -------
    numpy.ndarray of shape (2n, 2n)
        The symplectic form Ω.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n`` is not strictly positive.

    Notes
    -----
    The matrix is constructed as kron([[0, 1], [-1, 0]], I_n).
    """
    _valid_mode_number(n)
    identity_matrix = np.identity(n)
    w = np.array([
        [0.0 ,1.0],
        [-1.0,0.0]
    ])
    return np.kron(w,identity_matrix)

def compress_mean_covariance(mean_vector:npt.NDArray[np.number], covariance_matrix:npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Flatten a Gaussian state into a single vector.

    This function concatenates a mean vector and covariance matrix into a
    one-dimensional array. The output is given by

    [mean_vector, covariance_matrix.flatten()],

    where the covariance matrix is flattened in row-major (C) order.

    Parameters
    ----------
    mean_vector : numpy.ndarray
    The mean vector of shape (2n,).
    covariance_matrix : numpy.ndarray
    The covariance matrix of shape (2n, 2n).

    Returns
    -------
    numpy.ndarray
        A one-dimensional array of length 2n + (2n)^2 containing the
        concatenated data.

    Raises
    ------
    TypeError
        If inputs are not NumPy arrays.
    ValueError
        If the mean vector and covariance matrix are not valid or
        dimensionally consistent.

    Notes
    -----
    The covariance matrix is flattened in row-major (C) order. To reconstruct
    the original matrix, the number of modes n (or dimension 2n) must be known.
    """
    _valid_mean_covariance(mean_vector,covariance_matrix)
    return np.concatenate((mean_vector, covariance_matrix.flatten()),axis=0)

def extract_mean_covariance(mean_covariance_vector:npt.NDArray[np.number]) -> tuple[npt.NDArray[np.number],npt.NDArray[np.number]]:
    """
    Reconstruct a Gaussian state from a flattened representation.

    This function extracts a mean vector and covariance matrix from a
    one-dimensional array of length 2n + (2n)^2, assumed to be constructed
    via ``compress_mean_covariance``.

    The covariance matrix is reshaped from the flattened data and then
    symmetrized as (A + A.conj().T) / 2 before validation.

    Parameters
    ----------
    mean_covariance_vector : numpy.ndarray
        The flattened representation of the Gaussian state.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The reconstructed mean vector of shape (2n,) and covariance matrix
        of shape (2n, 2n).

    Raises
    ------
    TypeError
        If the input is not a real-valued NumPy vector.
    ValueError
        If the length is incompatible with 2n + 4n^2, or if the reconstructed
        mean vector and covariance matrix are not valid.

    Notes
    -----
    The covariance matrix is assumed to be flattened in row-major (C) order.
    The reconstruction enforces symmetry via projection, so the operation is
    not an exact inverse of ``compress_mean_covariance`` if the input matrix
    was not symmetric.
    """
    _require_real_vector(mean_covariance_vector, "mean-covariance vector")
    element_count = len(mean_covariance_vector)
    n_float = 0.25*(-1 + np.sqrt(1+4*element_count))
    n = int(round(n_float))
    if 4*n*n + 2*n != element_count:
        raise ValueError(f"compressed array length {element_count} is incompatible with expected 2n + 4n^2 form for an n-mode system.")
    _valid_mode_number(n)
    
    mean_vector = mean_covariance_vector[0:2*n]
    covariance_matrix = symmetrize_matrix((mean_covariance_vector[2*n:]).reshape( (2*n,2*n) ))
    
    _valid_mean_covariance(mean_vector,covariance_matrix)

    return ( mean_vector, covariance_matrix )

def mean_subsystem(mean_vector:npt.NDArray[np.number], indices: tuple[Integral, ...]|None = None) -> npt.NDArray[np.number]:
    """
    Extract a subsystem mean vector in x-then-p ordering.

    This function returns the mean vector corresponding to a subset of modes
    from an n-mode Gaussian state. The input mean vector is assumed to be
    ordered as (x_1, ..., x_n, p_1, ..., p_n).

    For a subsystem with modes (i_1, ..., i_k), the returned vector is
    ordered as
    (x_{i_1}, ..., x_{i_k}, p_{i_1}, ..., p_{i_k}).

    Parameters
    ----------
    mean_vector : numpy.ndarray
        The full mean vector of shape (2n,).
    indices : tuple of Integral or None
        The mode indices (1-based) defining the subsystem.

    Returns
    -------
    numpy.ndarray
        The subsystem mean vector of shape (2k,). If indices is None, returns a copy of the original mean vector

    Raises
    ------
    TypeError
        If ``mean_vector`` is not a NumPy array.
    ValueError
        If ``mean_vector`` is not valid or if ``indices`` are out of bounds.

    Notes
    -----
    The returned ordering is (x-subsystem, p-subsystem), not interleaved.
    """
    if indices is None:
        return mean_vector[:]
    _valid_mean_vector(mean_vector)
    n = len(mean_vector)//2
    final_idx = index_list(n, indices)
    return mean_vector[final_idx]

def covariance_subsystem(covariance_matrix:npt.NDArray[np.number], indices: tuple[Integral, ...]|None = None) -> npt.NDArray[np.number]:
    """
    Extract a subsystem covariance matrix in x-then-p ordering.

    This function returns the covariance matrix corresponding to a subset
    of modes from an n-mode Gaussian state. The input covariance matrix is
    assumed to be ordered as (x_1, ..., x_n, p_1, ..., p_n).

    For a subsystem with modes (i_1, ..., i_k), the returned matrix
    corresponds to the quadratures
    (x_{i_1}, ..., x_{i_k}, p_{i_1}, ..., p_{i_k}).

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        The full covariance matrix of shape (2n, 2n).
    indices : tuple of Integral or None
        The mode indices (1-based) defining the subsystem.

    Returns
    -------
    numpy.ndarray
        The subsystem covariance matrix of shape (2k, 2k). If indices is None returns a copy of the covariance matrix.

    Raises
    ------
    TypeError
        If ``covariance_matrix`` is not a NumPy array.
    ValueError
        If ``covariance_matrix`` is not valid or if ``indices`` are out of bounds.

    Notes
    -----
    The returned matrix follows the (x-subsystem, p-subsystem) ordering.
    """
    if indices is None:
        return covariance_matrix[:]
    _valid_covariance_matrix(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
    final_idx = index_list(n, indices)
    return covariance_matrix[np.ix_(final_idx, final_idx)]

def is_physical_covariance_matrix(covariance_matrix:npt.NDArray[np.float64], tol:Real = 1e-8) -> bool:
    """
    Test whether a covariance matrix is physically admissible for a Gaussian state.

    This function checks whether a covariance matrix satisfies the Gaussian
    quantum uncertainty relation

    V + (i/2) Ω >= 0,

    where Ω is the canonical symplectic form for an n-mode system in the
    x-then-p phase-space ordering

    (x_1, ..., x_n, p_1, ..., p_n).

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        The covariance matrix of shape (2n, 2n).
    tol : Real, optional
        Numerical tolerance used in the positivity test. Eigenvalues greater than or equal to ``-tol`` are accepted as non-negative. Default is 1e-8.

    Returns
    -------
    bool
        ``True`` if the covariance matrix satisfies the quantum uncertainty relation within tolerance, and ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``covariance_matrix`` is not a NumPy array or if ``tol`` is not a real scalar.
    ValueError
        If ``covariance_matrix`` is not a valid covariance matrix or if ``tol`` is not strictly positive.

    Notes
    -----
    The covariance matrix is symmetrized internally as ``(V + V.conj().T) / 2`` before testing physicality. This function checks quantum admissibility, not just classical positive semidefiniteness.
    """
    _valid_covariance_matrix(covariance_matrix)
    _require_positive_real_scalar(tol, "positive spectrum tolerance")
        
    n = (covariance_matrix.shape[0])//2
        
    cov_sym = symmetrize_matrix(covariance_matrix)
    
    eigs = np.real(eigvals(cov_sym + 0.5j*symplectic_matrix(n)))
    return np.all(eigs >= -1*tol)

def require_physical_covariance(covariance_matrix:npt.NDArray[np.float64]) -> None:
    """
    Enforce that a covariance matrix is physically admissible.

    This function validates that ``covariance_matrix`` is both:
    1. a valid classical covariance matrix (real, symmetric, positive semidefinite),
    2. a valid quantum Gaussian covariance matrix satisfying the Heisenberg uncertainty relation.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        The covariance matrix of shape (2n, 2n).

    Raises
    ------
    TypeError
        If ``covariance_matrix`` is not a NumPy array.
    ValueError
        If ``covariance_matrix`` is not a valid covariance matrix or fails the Heisenberg uncertainty condition.

    Notes
    -----
    This function performs a strict validation and raises an exception on failure.
    It uses the default numerical tolerance of ``is_physical_covariance_matrix``.
    """
    _valid_covariance_matrix(covariance_matrix)
    if not is_physical_covariance_matrix(covariance_matrix):
        raise ValueError(f"Provided covariance matrix failed the Heisenberg uncertainty condition. Got {covariance_matrix}")






