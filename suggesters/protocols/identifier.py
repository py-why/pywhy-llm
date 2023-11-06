from typing import List, Dict, Set, Protocol


class IdentifierProtocol(Protocol):
    def suggest_backdoor(
        self,
        treatment,
        outcome,
        factors_list
    ):
        """
        Suggests variables that potentially satisfy the backdoor criterion

        Returns variables that potentially satisfy the backdoor criterion
        """
        pass

    def suggest_frontdoor(
        self,
        treatment,
        outcome,
        factors_list
    ):
        """
        Suggest variables that potentially satisfy the frontdoor criterion
        calls self.suggest_mediators()

        Returns variables that potentially satisfy the frontdoor criterion
        """
        pass

    def suggest_iv(
        self,
        treatment,
        outcome,
        factors_list
    ):
        """
        Suggest potential instrumental variables

        Returns potential instrumental variables
        """
        pass

    def suggest_estimand(
        self,
        treatment,
        outcome,
        llm,
        confounders=None,
        mediators=None,
        ivs=None,
    ):
        """
        Suggest the estiamnds based off the suggested backdoor, frontdoor, and instrumental variables

        Returns an estimand: IdentifiedEstimand
        """
        pass
