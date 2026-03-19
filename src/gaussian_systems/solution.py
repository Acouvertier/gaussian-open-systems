from .initial_state import GaussianCVState

from .metrics import compute_logarithmic_negativity, state_purity, renyi_two_entropy, compute_gaussian_fidelity

class GaussianSolution:
    def __init__(self,t_eval,mean_vectors,covariance_matrices):
        self.t_eval = t_eval
        self.mean_vectors = mean_vectors
        self.covariance_matrices = covariance_matrices
    def entanglement_time_trace(self, subsystem:tuple[int,int] = (1,2)):
        return [compute_logarithmic_negativity(covariance, subsystem) for covariance in self.covariance_matrices]
    def purity_time_trace(self, subsystem:tuple[int,...]|None = None):
        return [state_purity(covariance, subsystem) for covariance in self.covariance_matrices]
    def entropy_time_trace(self, subsystem:tuple[int,...]|None = None):
        return [renyi_two_entropy(covariance, subsystem) for covariance in self.covariance_matrices]
    def fidelity_time_trace_fixed(self, state2:GaussianCVState, subsystem:tuple[int,...]|None = None):
        return [compute_gaussian_fidelity(
            (mean_subsystem(mean,subsystem), covariance_subsystem(covariance,subsystem)),
            (state2.mean_vector, state2.covariance_matrix)
        ) for mean, covariance in zip(self.mean_vectors, self.covariance_matrices)]