from typing import Dict, Set, Tuple, Protocol
import guidance

class IdentifierProtocol(Protocol):
    def suggest_backdoor(self, variable_descriptions: Dict[str, str], variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], llm: guidance.llms) -> Set[str]:
        """
        Suggest backdoor path

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.

            llm: guidance.llms
                User provided llm access.

        Returns:
            backdoor_set: Set[str]
                Set of variables in the backdoor set.
        """
        pass

    def suggest_frontdoor(self, variable_descriptions: Dict[str, str], variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], llm: guidance.llms) -> Set[str]:
        """
        Suggest frontdoor path

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.

            llm: guidance.llms
                User provided llm access.

        Returns:
            frontdoor_set: Set[str]
                Set of variables in the frontdoor set.
        """
        pass

    def suggest_iv(self, variable_descriptions: Dict[str, str], variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], llm: guidance.llms) -> Set[str]:
        """
        Suggest instrumental variables

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.

            llm: guidance.llms
                User provided llm access.

        Returns:
            instrumental_variables: Set[str]
                Set of instrumental variables.
        """
        pass

    def suggest_analysis_code(self, variable_descriptions: Dict[str, str], variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], llm: guidance.llms) -> str:
        """
        Suggest code to run identification analysis

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.

            llm: guidance.llms
                User provided llm access.

        Returns:
            code: str
                Code to run the identification analysis.
        """
        pass
