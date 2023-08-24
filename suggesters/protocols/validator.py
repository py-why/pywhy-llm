from typing import Dict, Set, Tuple, Protocol


class ValidatorProtocol(Protocol):
    def critique_graph(
        self,
        edges,
        analysis_context,
        treatment,
        outcome,
        factors_list,
        experts,
        llm,
        stakeholders,
        temperature=0.3,
        expert_confidence=0.7,
        model_type="chat",
    ):
        """
        Critique the graph edges

        Returns original graph edges with each edge critiqued and either validated or refuted, and the new graph edges
        """
        pass

    def suggest_latent_confounders(
        self,
        factors_list,
        treatment,
        outcome,
        llm,
        experts,
        stakeholders,
        analysis_context,
        temperature=0.3,
        expert_confidence=0.7,
        model_type="chat",
    ):
        """
        Suggest potential latent confounders

        Returns list of potential latent confounders
        """
        pass

    def suggest_negative_controls(
        self,
        factors_list,
        treatment,
        outcome,
        llm,
        experts,
        stakeholders,
        analysis_context,
        temperature=0.3,
        expert_confidence=0.7,
        model_type="chat",
    ):
        """
        Suggest potential negative controls

        Returns list of potential negative controls
        """
        pass
