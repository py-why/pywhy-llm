from typing import List, Protocol


class ModelerProtocol(Protocol):
    def suggest_relationships(
        self,
        treatment,
        outcome,
        factors_list,
        llm,
        temperature=0.3,
        experts=None,
        stakeholders=None,
        model_type="chat",
        analysis_context=None,
        expert_confidence=0.7,
        relationship_strategy="parent",
    ):
        """
        Suggests the relationships between variables

        Returns list of potential edges
        """
        pass

    def suggest_confounders(
        self,
        treatment,
        outcome,
        factors_list,
        llm,
        temperature=0.3,
        experts=None,
        stakeholders=None,
        model_type="chat",
        analysis_context=None,
        expert_confidence=0.7,
        relationship_strategy="parent",
    ):
        """
        Suggests variables confounding the relationship between treatment and outcome

        Returns list of potential confounders
        """
        pass
