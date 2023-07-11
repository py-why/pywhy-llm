
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from typing import List, Dict, Set, Protocol
import guidance

class IdentifierProtocol(Protocol):

    def suggest_backdoor(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
        """
        Suggest variables that satisfy the backdoor criterion

        Args:
            variables: List[str]
                List of variables 

            confounders: List[str]
                List of confounders 

            llm: guidance.llms
                User provided llm to access

        Returns:
            backdoor_set: Set[str]
                Set of variables in the backdoor set.
        """
        pass

    def suggest_frontdoor(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
        """
         Suggest variables that satisfy the frontdoor criterion

        Args:
            variables: List[str]
                List of variables 

            confounders: List[str]
                List of confounders 

            llm: guidance.llms
                User provided llm to access

        Returns:
            frontdoor_set: Set[str]
                Set of variables in the frontdoor set.
        """
        pass

    def suggest_iv(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
        """
        Suggest instrumental variables

       Args:
            variables: List[str]
                List of variables 

            confounders: List[str]
                List of confounders 

            llm: guidance.llms
                User provided llm to access

        Returns:
            instrumental_variables: Set[str]
                Set of instrumental variables.
        """
        pass

    def suggest_estimand(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm, backdoor: Set[str] = None, frontdoor: Set[str] = None, iv: Set[str] = None):
        """
        Suggest the estiamnds based off the suggested backdoor, frontdoor, and instrumental variables

        Args:
            variables: List[str]
                List of variables 

            confounders: List[str]
                List of confounders 

            llm: guidance.llms
                User provided llm to access

            backdoor: Set[str]  = None (Optional)
                Set of variables in the backdoor set.

            frontdoor: Set[str] = None (Optional)
                Set of variables in the frontdoor set.  

            iv: Set[str] = None (Optional)  
                Set of instrumental variables.  

        Returns:
            estimand: IdentifiedEstimand
        """
        pass



