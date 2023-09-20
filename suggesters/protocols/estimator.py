from typing import Protocol

class EstimatorProtocol(Protocol):
    def suggest_estimation_code(self) -> str:
        """
        TODO: needs to be updated to take in a task specification

        Suggest code to run the causal effect analysis

        Returns:
            code str: suggested code for running estimation
        """
        pass