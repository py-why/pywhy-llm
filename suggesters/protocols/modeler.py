from typing import List, Protocol

class ModelerProtocol(Protocol):

    def suggest_pairwise_relationship(self, variable1: str, variable2: str):
        """
        Suggest the relationships between variables.

        Args:
            variable1: str
                A string name of variable1

            variable2: str
                A string name of variable2

        Returns:
            [variable1, variable2, description] if variable1 causes variable2
            [variable2, variable1, description] if variable2 causes variable1
            [None, None, description] if neither causes the other
        """
        pass


    def suggest_variable_relationships(self, variables: List[str]):
        """
        Suggest the relationships between variables.

        Args:
            variables: List[str]
                A list of variable names

            TODO: add a way to pass in longer variable descriptions
        
        Returns:
            variable_relationships: Dict[[cause,effect],description] 
            
            TODO: update to return a NetworkX graph, with the description as metadata for each edge
                
        """
        pass

    def suggest_confounders(self, variables: List[str], treatment: str, outcome: str):
        """
        Suggest confounders

        Args:
            variable_descriptions: Dict[str, str]
                A dictionary mapping variable names to their descriptions.

            variable_relationships: Dict[Tuple[str, str], str]
                A dictionary where the keys are edges, where it's assumed that parent is first, child is second, and the values are an explanation for how their relationship occurs.

        Returns:
            confounders: Set[Tuple[str, str]]
                Set of confounders along with a reasoning or explanation for how the confounding occurs.
        """
        pass










