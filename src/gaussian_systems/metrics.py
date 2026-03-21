import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvals, det, cholesky, solve, eigh, issymmetric, eigvalsh
from numbers import Real

from .conventions import mean_subsystem, covariance_subsystem, symplectic_matrix, symmetrize_matrix, require_physical_covariance
from ._validation import _valid_fidelity_input, _valid_covariance_matrix, _require_square_matrix, _valid_indices, _require_tuple_length, _require_symmetric, _require_positive_real_vector

_ppt_matrix = np.diag([1,1,-1,1])

def _matching_covariances(cov1:npt.NDArray[np.float64], cov2:npt.NDArray[np.float64]) -> None:
    """
    Validate that two covariance matrices are physical and dimensionally compatible.

    This function checks that both input covariance matrices are valid physical
    Gaussian covariance matrices and that they correspond to the same number of
    modes, i.e., have identical dimensions in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    Parameters
    ----------
    cov1 : numpy.ndarray
        The first covariance matrix.
    cov2 : numpy.ndarray
        The second covariance matrix.

    Raises
    ------
    ValueError
        If either covariance matrix is not physical or if the two matrices
        have different dimensions.

    Notes
    -----
    Physicality is enforced through ``require_physical_covariance``, which
    includes symmetry, positive semidefiniteness, and the quantum uncertainty
    relation.
    """
    require_physical_covariance(cov1)
    require_physical_covariance(cov2)
    if (cov1.shape)[0] != (cov2.shape)[0]:
        raise ValueError(f"Covariance matrices must have same dimension. Got system1: {cov1.shape} and system2: {cov2.shape}")

def _lambda_matrix(covariance_matrix:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    """
    Construct the Lambda matrix associated with a covariance matrix.

    This function returns the matrix

        Lambda(V) = - Ω V Ω V,

    where ``V`` is a physical Gaussian covariance matrix and ``Ω`` is the
    canonical symplectic form in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The result is symmetrized numerically before being returned.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        A physical Gaussian covariance matrix of shape (2n, 2n).

    Returns
    -------
    numpy.ndarray
        The symmetrized matrix ``-Ω V Ω V``.

    Raises
    ------
    TypeError
        If ``covariance_matrix`` is not a valid NumPy array.
    ValueError
        If ``covariance_matrix`` is not a physical Gaussian covariance matrix.

    Notes
    -----
    The eigenvalues of this matrix are the squared symplectic eigenvalues
    of ``covariance_matrix`` (up to numerical error and multiplicity).
    This object is useful in Gaussian CV metric calculations such as purity
    and entropy.
    """
    require_physical_covariance(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    return symmetrize_matrix(-omega_matrix @ covariance_matrix @ omega_matrix @ covariance_matrix)

def _gamma_matrix(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    """
    Construct the Gamma matrix for two Gaussian covariance matrices.

    This function returns the matrix

        Gamma(V1, V2) = Ω V1 Ω V2 - (1/4) I,

    where ``V1`` and ``V2`` are physical Gaussian covariance matrices and
    ``Ω`` is the canonical symplectic form in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    Parameters
    ----------
    covariance_matrix1 : numpy.ndarray
        The first physical covariance matrix of shape (2n, 2n).
    covariance_matrix2 : numpy.ndarray
        The second physical covariance matrix of shape (2n, 2n).

    Returns
    -------
    numpy.ndarray
        The matrix ``Ω V1 Ω V2 - (1/4) I``.

    Raises
    ------
    ValueError
        If either covariance matrix is not physical or if the two matrices
        have different dimensions.

    Notes
    -----
    This matrix is an auxiliary object used in Gaussian CV metric calculations,
    particularly fidelity formulas. The ``1/4`` term is consistent with the
    convention that vacuum quadrature variance equals ``1/2``.
    """
    _matching_covariances(covariance_matrix1, covariance_matrix2)
    n = (covariance_matrix1.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    return omega_matrix @ covariance_matrix1 @ omega_matrix @ covariance_matrix2 - 0.25*np.identity(2*n)

def _sigma_matrix(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> npt.NDArray[np.float64]:
    """
    Construct the Sigma matrix for two Gaussian covariance matrices.

    This function returns the matrix

        Sigma(V1, V2) = V1 + V2,

    where ``V1`` and ``V2`` are physical Gaussian covariance matrices in
    the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The result is symmetrized numerically to remove any asymmetry due to
    floating-point error.

    Parameters
    ----------
    covariance_matrix1 : numpy.ndarray
        The first physical covariance matrix of shape (2n, 2n).
    covariance_matrix2 : numpy.ndarray
        The second physical covariance matrix of shape (2n, 2n).

    Returns
    -------
    numpy.ndarray
        The symmetrized matrix ``V1 + V2``.

    Raises
    ------
    ValueError
        If either covariance matrix is not physical or if the two matrices
        have different dimensions.

    Notes
    -----
    This matrix is an auxiliary object used in Gaussian CV metric
    calculations such as fidelity. It is not itself required to satisfy
    the quantum uncertainty relation.
    """
    _matching_covariances(covariance_matrix1, covariance_matrix2)
    return symmetrize_matrix(covariance_matrix1 + covariance_matrix2)

def _logdet_spd(matrix:npt.NDArray[np.float64]) -> np.float64:
    """
    Compute the log-determinant of a symmetric positive-definite matrix.

    This function evaluates

        log(det(M))

    for a symmetric positive-definite matrix ``M`` using a Cholesky
    decomposition

        M = L L^T,

    with

        log(det(M)) = 2 * sum(log(diag(L))).

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric positive-definite matrix.

    Returns
    -------
    float
        The natural logarithm of the determinant of ``matrix``.

    Raises
    ------
    ValueError
        If the matrix is not symmetric or not positive definite.

    Notes
    -----
    This method is numerically more stable than computing
    ``np.log(np.linalg.det(matrix))``, especially for ill-conditioned
    matrices.
    """
    _require_symmetric(matrix, "provided matrix")
    try:
        L = cholesky(matrix)
    except Expection as e:
        raise ValueError(f"cholesky expected a matrix that is approximately positive. Got {matrix}") from e
    logdet_val = 2*np.sum(np.log(np.diagonal(L)))
    return logdet_val

def _fidelity_fixed_parts(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> tuple[np.float64,np.float64]:
    """
    Compute log-domain contributions to Gaussian state fidelity.

    For two Gaussian states with means ``mean1``, ``mean2`` and covariance
    matrices ``covariance1``, ``covariance2``, this function forms

        du = mean2 - mean1,
        sigma = covariance1 + covariance2,

    and returns the pair

        (log(det(sigma)), -1/4 * du.T @ sigma^{-1} @ du).

    These quantities are intended to be combined additively when constructing
    the logarithm of the Gaussian fidelity.

    Parameters
    ----------
    mean_covariance_tuple1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first Gaussian state, given as ``(mean_vector, covariance_matrix)``.
    mean_covariance_tuple2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second Gaussian state, given as ``(mean_vector, covariance_matrix)``.

    Returns
    -------
    tuple of scalar
        A pair containing:
        - ``log(det(sigma))``
        - ``-1/4 * du.T @ sigma^{-1} @ du``

    Raises
    ------
    TypeError
        If either input state has invalid type.
    ValueError
        If either state is invalid or if the two states are dimensionally
        incompatible.

    Notes
    -----
    All quantities are returned in the log-domain to maintain numerical
    stability. The final fidelity should be obtained by exponentiating the
    fully assembled logarithmic expression.
    """
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    mean1, covariance1 = mean_covariance_tuple1
    mean2, covariance2 = mean_covariance_tuple2
    du = mean2 - mean1
    sigma = _sigma_matrix(covariance1,covariance2)
    return (np.real_if_close(_logdet_spd(sigma)),np.real_if_close(-0.25*du@(solve(sigma,du))))

def _calculate_log_lambda(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> np.float64:
    _matching_covariances(covariance_matrix1, covariance_matrix2)
    n = covariance_matrix1.shape[0] // 2
    omega = symplectic_matrix(n)

    sign1, logabsdet1 = np.linalg.slogdet(covariance_matrix1 + 0.5j * omega)
    sign2, logabsdet2 = np.linalg.slogdet(covariance_matrix2 + 0.5j * omega)
    
    return np.real_if_close((2 * n) * np.log(2) + np.log(sign1) + np.log(sign2) + logabsdet1 + logabsdet2)

def _calculate_log_gamma(covariance_matrix1:npt.NDArray[np.float64],covariance_matrix2:npt.NDArray[np.float64])  -> np.float64:
    _matching_covariances(covariance_matrix1, covariance_matrix2)
    n = (covariance_matrix1.shape)[0]//2
    gamma = _gamma_matrix(covariance_matrix1, covariance_matrix2)
    gammaSign, gammaVal = np.linalg.slogdet(gamma)
    """
    Compute the logarithm of the Gamma invariant for Gaussian fidelity.

    This function evaluates

        log(Gamma) =
            2n * log(2)
            + log(det(Ω V1 Ω V2 - (1/4) I)),

    where ``V1`` and ``V2`` are physical Gaussian covariance matrices and
    ``Ω`` is the canonical symplectic form in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The determinant is evaluated using ``numpy.linalg.slogdet`` for
    improved numerical stability.

    Parameters
    ----------
    covariance_matrix1 : numpy.ndarray
        The first physical covariance matrix of shape (2n, 2n).
    covariance_matrix2 : numpy.ndarray
        The second physical covariance matrix of shape (2n, 2n).

    Returns
    -------
    float
        The logarithm of the Gamma invariant.

    Raises
    ------
    ValueError
        If either covariance matrix is not physical or if the two matrices
        have different dimensions.

    Notes
    -----
    For physical covariance matrices, the result should be real up to
    numerical error. This function is intended for internal use in
    log-domain Gaussian fidelity calculations.
    """
    return np.real_if_close((2 * n) * np.log(2) + np.log(gammaSign) + gammaVal)

"""PUBLIC"""

def compute_logarithmic_negativity(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,int] = (1,2)) -> Real:
    """
    Compute the logarithmic negativity of a two-mode Gaussian subsystem.

    This function evaluates the logarithmic negativity of the two-mode
    subsystem specified by ``subsystem`` from a physical Gaussian covariance
    matrix in the x-then-p phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The selected two-mode covariance matrix is ordered as

        (x_i, x_j, p_i, p_j),

    where ``subsystem = (i, j)``. Logarithmic negativity is computed from the
    smallest symplectic eigenvalue of the partially transposed covariance
    matrix:

        E_N = max(0, -log(2 * nu_min_tilde)),

    where ``nu_min_tilde`` is the minimum symplectic eigenvalue after partial
    transposition.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        A physical Gaussian covariance matrix of shape (2n, 2n).
    subsystem : tuple of (int, int), optional
        The two mode indices (1-based) defining the subsystem. Default is
        ``(1, 2)``.

    Returns
    -------
    numbers.Real
        The logarithmic negativity of the selected two-mode subsystem.

    Raises
    ------
    TypeError
        If ``subsystem`` does not have the correct type.
    ValueError
        If ``covariance_matrix`` is not physical, if ``subsystem`` does not
        have length 2, or if the subsystem indices are invalid.

    Notes
    -----
    Partial transposition is implemented in the reduced two-mode x-then-p
    ordering via the fixed matrix

        diag(1, 1, -1, 1),

    corresponding to a sign flip of one momentum quadrature. The formula
    assumes the convention that vacuum quadrature variance equals ``1/2``.
    """
    require_physical_covariance(covariance_matrix)
    n = (covariance_matrix.shape)[0]//2
   
    _require_tuple_length(subsystem, 2, "subsystem")
    _valid_indices(n, subsystem)
    sub_covariance = symmetrize_matrix(covariance_subsystem(covariance_matrix,subsystem))
    omega_matrix = symplectic_matrix(2)
    symplectic_form = 1j*omega_matrix@_ppt_matrix@sub_covariance@_ppt_matrix
    nu_min = np.min(np.abs(eigvals(symplectic_form)))
    log_neg = np.max([0,-np.log(2*nu_min)])
    return log_neg
    
def state_purity(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,...]|None = None) -> Real:
    """
    Compute the Gaussian purity of a state or subsystem.

    This function evaluates the purity

        purity = Tr(rho^2),

    for a Gaussian state specified by its covariance matrix. In the convention
    where vacuum quadrature variance equals ``1/2``, the purity of a subsystem
    with covariance matrix ``V`` is

        purity = 1 / sqrt(det(2 V)).

    The full covariance matrix is assumed to follow the x-then-p ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If a subsystem is specified, the corresponding reduced covariance matrix is
    extracted in the ordering

        (x_{i_1}, ..., x_{i_k}, p_{i_1}, ..., p_{i_k}).

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        A physical Gaussian covariance matrix of shape (2n, 2n).
    subsystem : tuple of int or None, optional
        The mode indices (1-based) defining the subsystem whose purity is to
        be computed. If ``None``, the purity of the full state is returned.

    Returns
    -------
    numbers.Real
        The purity of the selected state or subsystem.

    Raises
    ------
    ValueError
        If ``covariance_matrix`` is not physical or if the subsystem indices
        are invalid.

    Notes
    -----
    For a pure Gaussian state, the purity equals ``1``. Mixed states satisfy
    ``0 < purity < 1``.
    """
    require_physical_covariance(covariance_matrix)
    n =  (covariance_matrix.shape)[0]//2
    if subsystem is None:
        subsystem = tuple(range(1,n+1))
    _valid_indices(n, subsystem)
    return 1/np.sqrt(np.real_if_close(det(2*symmetrize_matrix(covariance_subsystem(covariance_matrix,subsystem)))))

def renyi_two_entropy(covariance_matrix:npt.NDArray[np.float64], subsystem:tuple[int,...]|None = None) -> Real:
    """
    Compute the Rényi-2 entropy of a Gaussian state or subsystem.

    This function evaluates the Rényi-2 entropy

        S_2 = -log(Tr(rho^2)),

    where ``Tr(rho^2)`` is the purity of the state. The purity is computed
    from the covariance matrix under the convention that vacuum quadrature
    variance equals ``1/2``.

    The covariance matrix is assumed to follow the x-then-p ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    If a subsystem is specified, the entropy is computed for the corresponding
    reduced Gaussian state.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        A physical Gaussian covariance matrix of shape (2n, 2n).
    subsystem : tuple of int or None, optional
        The mode indices (1-based) defining the subsystem. If ``None``,
        the entropy of the full state is returned.

    Returns
    -------
    numbers.Real
        The Rényi-2 entropy of the selected state or subsystem.

    Raises
    ------
    ValueError
        If ``covariance_matrix`` is not physical or if the subsystem indices
        are invalid.

    Notes
    -----
    The Rényi-2 entropy is non-negative and equals zero for pure Gaussian
    states.
    """
    return np.real_if_close(-np.log(state_purity(covariance_matrix,subsystem)))

def one_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    """
    Compute the standard Uhlmann fidelity between two one-mode Gaussian states.

    This function evaluates the fidelity between two one-mode Gaussian states
    specified by their mean vectors and covariance matrices in the (x, p)
    phase-space ordering.

    The implementation uses the one-mode closed-form Gaussian fidelity formula
    in terms of

        Delta = det(V1 + V2),
        Lambda = 2^(2n) det(V1 + (i/2) Ω) det(V2 + (i/2) Ω),

    with n = 1. The returned value is the standard Uhlmann fidelity, i.e.
    the square of the root fidelity often used in the Gaussian-state
    literature.

    The denominator is evaluated in rationalized form,

        1 / (sqrt(Delta + Lambda) - sqrt(Lambda))
        = (sqrt(Delta + Lambda) + sqrt(Lambda)) / Delta,

    which is the form used internally.

    Parameters
    ----------
    mean_covariance_tuple1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first one-mode Gaussian state, given as
        ``(mean_vector, covariance_matrix)``.
    mean_covariance_tuple2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second one-mode Gaussian state, given as
        ``(mean_vector, covariance_matrix)``.

    Returns
    -------
    numbers.Real
        The standard Uhlmann fidelity between the two one-mode Gaussian states.

    Raises
    ------
    TypeError
        If either input state has invalid type.
    ValueError
        If either state is invalid, if the two states are dimensionally
        incompatible, or if either covariance matrix is not 2x2.

    Notes
    -----
    This function is restricted to one-mode states. The calculation is
    assembled partly in the log-domain for improved numerical stability,
    then exponentiated at the end.
    """
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    if covariance1.shape[0]//2 != 1:
        raise ValueError(f"Both inputs must single mode states. This means each covariance must be 2x2. Got {covariance1.shape[0]}.")
    covariance2 = mean_covariance_tuple2[1]
    log_lambda = _calculate_log_lambda(covariance1,covariance2)
    log_delta, log_exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    
    log_fid = np.log(np.sqrt(np.exp(log_lambda) + np.exp(log_delta)) + np.exp(0.5*log_lambda)) - log_delta + 2*log_exponential_part 
    return np.exp(log_fid)
    
def two_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    """
    Compute the standard Uhlmann fidelity between two two-mode Gaussian states.

    This function evaluates the fidelity between two two-mode Gaussian states
    specified by their mean vectors and covariance matrices in the x-then-p
    phase-space ordering

        (x_1, x_2, p_1, p_2).

    The implementation uses the closed-form two-mode Gaussian fidelity
    prefactor

        F_0^2(V1, V2)
        = 1 / (sqrt(Gamma) + sqrt(Lambda)
        - sqrt((sqrt(Gamma) + sqrt(Lambda))^2 - Delta)),

    where

        Delta  = det(V1 + V2),
        Lambda = 2^(2n) det(V1 + (i/2) Ω) det(V2 + (i/2) Ω),
        Gamma  = 2^(2n) det(Ω V1 Ω V2 - (1/4) I),

    with n = 2.

    The denominator is evaluated in rationalized form,

        F_0^2(V1, V2)
        = (sqrt(Gamma) + sqrt(Lambda)
        + sqrt((sqrt(Gamma) + sqrt(Lambda))^2 - Delta)) / Delta,

    and combined with the Gaussian displacement contribution. The returned
    value is the standard Uhlmann fidelity, not the root fidelity.

    Parameters
    ----------
    mean_covariance_tuple1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first two-mode Gaussian state, given as
        ``(mean_vector, covariance_matrix)``.
    mean_covariance_tuple2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second two-mode Gaussian state, given as
        ``(mean_vector, covariance_matrix)``.

    Returns
    -------
    numbers.Real
        The standard Uhlmann fidelity between the two two-mode Gaussian states.

    Raises
    ------
    TypeError
        If either input state has invalid type.
    ValueError
        If either state is invalid, if the two states are dimensionally
        incompatible, or if either covariance matrix is not 4x4.

    Notes
    -----
    This function is restricted to two-mode states. The calculation is
    assembled partly in the log-domain for improved numerical stability,
    then exponentiated at the end.
    """
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    if covariance1.shape[0]//2 != 2:
        raise ValueError(f"Both inputs must two mode states. This means each covariance must be 4x4. Got {covariance1.shape[0]}.")
    log_delta, log_exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance2 = mean_covariance_tuple2[1]
    log_lambda = _calculate_log_lambda(covariance1,covariance2)
    log_gamma = _calculate_log_gamma(covariance1,covariance2)
    root_gamma_plus_root_lambda = np.exp(0.5*log_gamma) + np.exp(0.5*log_lambda)
    root_gamma_plus_root_lambda = np.exp(0.5 * log_gamma) + np.exp(0.5 * log_lambda)
    radicand = root_gamma_plus_root_lambda**2 - np.exp(log_delta)
    radicand = np.maximum(radicand, 0.0)

    log_fid = (
        np.log(root_gamma_plus_root_lambda + np.sqrt(radicand))
        - log_delta
        + 2 * log_exponential_part
    )
    return np.exp(log_fid)

def n_mode_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    """
    Compute the standard Uhlmann fidelity between two n-mode Gaussian states.

    This function evaluates the fidelity between two Gaussian states specified
    by their mean vectors and covariance matrices in the x-then-p phase-space
    ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The implementation uses the general auxiliary-matrix formula for the root
    Gaussian fidelity prefactor,

        F_0(V1, V2) = F_tot / det(V1 + V2)^(1/4),

    where

        F_tot = prod_k [w_k^aux + sqrt((w_k^aux)^2 - 1)]^(1/2),

    and the auxiliary matrix is defined by

        W = -2 V i Ω,
        W_aux = -(W1 + W2)^(-1) (I + W2 W1).

    The returned value is the standard Uhlmann fidelity, i.e. the square of
    the root fidelity convention often used in the Gaussian-state literature.

    Parameters
    ----------
    mean_covariance_tuple1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first Gaussian state, given as ``(mean_vector, covariance_matrix)``.
    mean_covariance_tuple2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second Gaussian state, given as ``(mean_vector, covariance_matrix)``.

    Returns
    -------
    numbers.Real
        The standard Uhlmann fidelity between the two Gaussian states.

    Raises
    ------
    TypeError
        If either input state has invalid type.
    ValueError
        If either state is invalid, if the two states are dimensionally
        incompatible, or if the auxiliary spectrum falls outside the expected
        physical domain.

    Notes
    -----
    The calculation is assembled in the log-domain as

        log F = 2 log(F_tot) - (1/2) log(det(V1 + V2))
                - (1/2) du^T (V1 + V2)^(-1) du,

    and exponentiated only at the end for improved numerical stability.
    """
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    log_delta, log_exponential_part = _fidelity_fixed_parts(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    covariance2 = mean_covariance_tuple2[1]
    n = (covariance1.shape)[0]//2
    omega_matrix = symplectic_matrix(n)
    
    w1, w2 = [-2j*cov@omega_matrix for cov in [covariance1, covariance2]]
    w1_w2_inv = solve(w1 + w2,np.identity(2*n))
    w_auxiliary = -  w1_w2_inv @ (np.identity(2*n) + w2@w1) 
    w_eigs = np.real(np.real_if_close(eigvals(w_auxiliary)))

    tol = 1e-10
    log_terms = []
    for wi in w_eigs:
        if wi < 1 - tol:
            raise ValueError(
                f"Expected auxiliary eigenvalues satisfying w >= 1 up to tolerance. Got {wi}."
            )
        wi_eff = max(wi, 1.0)
        log_terms.append(0.5 * np.log(wi_eff + np.sqrt(wi_eff**2 - 1.0)))
    log_f_tot = np.sum(log_terms)
    log_fid = 2*log_f_tot - 0.5*log_delta + 2*log_exponential_part
    return np.exp(log_fid)

def compute_gaussian_fidelity(mean_covariance_tuple1:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], mean_covariance_tuple2:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]) -> Real:
    """
    Compute the standard Uhlmann fidelity between two Gaussian states.

    This function evaluates the Gaussian-state fidelity between two states
    specified by their mean vectors and covariance matrices in the x-then-p
    phase-space ordering

        (x_1, ..., x_n, p_1, ..., p_n).

    The function automatically selects the appropriate implementation based
    on the number of modes:

    - ``n = 1``: one-mode closed-form formula
    - ``n = 2``: two-mode closed-form formula
    - ``n >= 3``: general n-mode auxiliary-matrix formula

    The returned value is the standard Uhlmann fidelity, not the root
    fidelity convention often used in parts of the Gaussian-state literature.

    Parameters
    ----------
    mean_covariance_tuple1 : tuple of (numpy.ndarray, numpy.ndarray)
        The first Gaussian state, given as ``(mean_vector, covariance_matrix)``.
    mean_covariance_tuple2 : tuple of (numpy.ndarray, numpy.ndarray)
        The second Gaussian state, given as ``(mean_vector, covariance_matrix)``.

    Returns
    -------
    numbers.Real
        The standard Uhlmann fidelity between the two Gaussian states.

    Raises
    ------
    TypeError
        If either input state has invalid type.
    ValueError
        If either state is invalid or if the two states are dimensionally
        incompatible.

    Notes
    -----
    This function dispatches to specialized low-mode formulas when available
    and otherwise uses the general n-mode Gaussian fidelity expression.
    """
    _valid_fidelity_input(mean_covariance_tuple1, mean_covariance_tuple2)
    covariance1 = mean_covariance_tuple1[1]
    n = (covariance1.shape)[0]//2
    if n == 1:
        return one_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    elif n == 2:
        return two_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
    else:
        return n_mode_gaussian_fidelity(mean_covariance_tuple1, mean_covariance_tuple2)
        