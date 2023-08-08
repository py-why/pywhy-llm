

from typing import List, Protocol
import guidance

class ModelerProtocol(Protocol):

    def suggest_variable_relationships(self, variables: List[str], llm: guidance.llms):
        """
        Suggest the relationships between variables.

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            llm: guidance.llms
                User provided llm access.
        
        Returns:
            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.
        """
        pass

    def suggest_confounders(self, variables: List[str], llm: guidance.llms, treatment: str, outcome: str):
        """
        Suggest confounders

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

            llm: guidance.llms
                User provided llm access.

        Returns:
            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.
        """
        pass










