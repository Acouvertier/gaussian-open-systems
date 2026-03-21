from .initial_state import GaussianCVState
from .conventions import mean_subsystem, covariance_subsystem
from .metrics import compute_logarithmic_negativity, state_purity, renyi_two_entropy, compute_gaussian_fidelity

class GaussianSolution:
    """
    Container for the time-resolved output of Gaussian CV evolution.

    This class stores the time grid together with the corresponding mean
    vectors and covariance matrices produced by ``GaussianCVSystem.evolve_state``.
    It also provides convenience methods for evaluating Gaussian CV metrics
    along the trajectory.

    The phase-space convention throughout is x-then-p ordering,

        (x_1, ..., x_n, p_1, ..., p_n).

    Parameters
    ----------
    t_eval : array-like
        The time grid used for evolution.
    mean_vectors : sequence of numpy.ndarray
        The evolved mean vectors at each time in ``t_eval``.
    covariance_matrices : sequence of numpy.ndarray
        The evolved covariance matrices at each time in ``t_eval``.

    Notes
    -----
    This class is primarily intended as the output container of
    ``GaussianCVSystem.evolve_state``.
    """
    def __init__(self,t_eval,mean_vectors,covariance_matrices):
        """
        Initialize a Gaussian solution container.

        Parameters
        ----------
        t_eval : array-like
            The time grid used for evolution.
        mean_vectors : sequence of numpy.ndarray
            The evolved mean vectors.
        covariance_matrices : sequence of numpy.ndarray
            The evolved covariance matrices.
        """
        self.t_eval = t_eval
        self.mean_vectors = mean_vectors
        self.covariance_matrices = covariance_matrices
    def entanglement_time_trace(self, subsystem:tuple[int,int] = (1,2)):
        """
        Compute logarithmic negativity along the trajectory.

        This method evaluates the logarithmic negativity of the specified
        two-mode subsystem for each covariance matrix in the trajectory.

        Parameters
        ----------
        subsystem : tuple of (int, int), optional
            The two mode indices (1-based) defining the subsystem.
            Default is ``(1, 2)``.

        Returns
        -------
        list of Real
            The logarithmic negativity evaluated at each time point.

        Notes
        -----
        This method uses ``compute_logarithmic_negativity`` on each
        covariance matrix in the trajectory.
        """
        return [compute_logarithmic_negativity(covariance, subsystem) for covariance in self.covariance_matrices]
    def purity_time_trace(self, subsystem:tuple[int,...]|None = None):
        """
        Compute purity along the trajectory.

        This method evaluates the Gaussian purity of either the full state
        or a selected subsystem for each covariance matrix in the trajectory.

        Parameters
        ----------
        subsystem : tuple of int or None, optional
            The subsystem mode indices (1-based). If ``None``, the purity of
            the full state is returned.

        Returns
        -------
        list of Real
            The purity evaluated at each time point.
        """
        return [state_purity(covariance, subsystem) for covariance in self.covariance_matrices]
    def entropy_time_trace(self, subsystem:tuple[int,...]|None = None):
        """
        Compute Rényi-2 entropy along the trajectory.

        This method evaluates the Rényi-2 entropy of either the full state
        or a selected subsystem for each covariance matrix in the trajectory.

        Parameters
        ----------
        subsystem : tuple of int or None, optional
            The subsystem mode indices (1-based). If ``None``, the entropy of
            the full state is returned.

        Returns
        -------
        list of Real
            The Rényi-2 entropy evaluated at each time point.
        """
        return [renyi_two_entropy(covariance, subsystem) for covariance in self.covariance_matrices]
    def fidelity_time_trace_fixed(self, state2:GaussianCVState, subsystem:tuple[int,...]|None = None):
        """
        Compute fidelity to a fixed Gaussian state along the trajectory.

        This method evaluates the Gaussian fidelity between each evolved
        state in the trajectory and a fixed comparison state ``state2``.
        If a subsystem is specified, the same subsystem reduction is applied
        to both the trajectory state and the fixed state before computing
        fidelity.

        Parameters
        ----------
        state2 : GaussianCVState
            The fixed comparison Gaussian state.
        subsystem : tuple of int or None, optional
            The subsystem mode indices (1-based). If ``None``, fidelity is
            computed using the full states.

        Returns
        -------
        list of Real
            The Gaussian fidelity evaluated at each time point.

        Notes
        -----
        This method assumes that the fixed state and each evolved state are
        dimensionally compatible after applying the same subsystem selection.
        """
        return [compute_gaussian_fidelity(
            (mean_subsystem(mean,subsystem), covariance_subsystem(covariance,subsystem)),
            (mean_subsystem(state2.mean_vector,subsystem), covariance_subsystem(state2.covariance_matrix,subsystem))
        ) for mean, covariance in zip(self.mean_vectors, self.covariance_matrices)]