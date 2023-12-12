from typing import List, Dict, Set, Tuple, Protocol
from .protocols import IdentifierProtocol
from .helpers import RelationshipStrategy, ModelType
from .model_suggester import ModelSuggester
from .prompts import prompts as ps
import guidance
import re

# from dowhy import causal_identifier as ci


class IdentificationSuggester(IdentifierProtocol):
    EXPERTS: list() = [
        "cause and effect",
        "causality, you are an intelligent AI with expertise in causality",
    ]
    CONTEXT: str = """causal mechanisms"""

    def suggest_backdoor(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        temperature=0.3,
        prompt_template: str = None,
        stakeholders: list() = None,
        model_type: ModelType = ModelType.Completion,
    ):
        backdoor_set = ModelSuggester.suggest_confounders(
            analysis_context=analysis_context,
            treatment=treatment,
            outcome=outcome,
            factors_list=factors_list,
            experts=experts,
            llm=llm, 
            temperature=temperature,
            prompt_template=prompt_template,
            stakeholders=stakeholders,
            model_type=model_type,
        )
        return backdoor_set

    def suggest_mediators(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        domain_expertise_list: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        temperature=0.3,
        prompt_template: str = None,
        stakeholders: list() = None,
        model_type: ModelType = ModelType.Completion,
    ):
        expert_list: List[str] = list()
        for elements in domain_expertise_list:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_mediators"]
        suggest = guidance(prompt_template)

        mediators: List[str] = list()
        mediators_edges: Dict[Tuple[str, str], int] = dict()
        mediators_edges[(treatment, outcome)] = 1

        edited_factors_list: List[str] = []
        for i in range(len(factors_list)):
            if factors_list[i] != treatment and factors_list[i] != outcome:
                edited_factors_list.append(factors_list[i])

        if len(expert_list) > 1:
            for expert in expert_list:
                mediators_edges, mediators_list = self.request_mediators(
                    suggest=suggest,
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    edited_factors_list=edited_factors_list,
                    temperature=temperature,
                    llm=llm,
                    mediators_edges=mediators_edges,
                )
                for m in mediators_list:
                    if m not in mediators:
                        mediators.append(m)
        else:
            mediators_edges, mediators_list = self.request_mediators(
                suggest=suggest,
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expert_list[0],
                edited_factors_list=edited_factors_list,
                temperature=temperature,
                llm=llm,
                mediators_edges=mediators_edges,
            )

            for m in mediators_list:
                if m not in mediators:
                    mediators.append(m)

        return mediators_edges, mediators

    def request_mediators(
        self,
        suggest,
        treatment,
        outcome,
        analysis_context,
        domain_expertise,
        edited_factors_list,
        temperature,
        llm,
        mediators_edges,
    ):
        mediators: List[str] = list()

        success: bool = False

        while not success:
            try:
                output = suggest(
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    factors_list=edited_factors_list,
                    factor=factor,
                    temperature=temperature,
                    llm=llm,
                )
                mediating_factor = re.findall(
                    r"<mediating_factor>(.*?)</mediating_factor>",
                    output["output"],
                )

                if mediating_factor:
                    for factor in mediating_factor:
                        # to not add it twice into the list
                        if factor in edited_factors_list and factor not in mediators:
                            mediators.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

            for element in mediators:
                if (treatment, element) in mediators_edges and (
                    element,
                    outcome,
                ) in mediators_edges:
                    mediators_edges[(treatment, element)] += 1
                    mediators_edges[(element, outcome)] += 1
                else:
                    mediators_edges[(treatment, element)] = 1
                    mediators_edges[(element, outcome)] = 1

        return mediators_edges, mediators

    def suggest_ivs(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        domain_expertise_list: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        temperature=0.3,
        prompt_template: str = None,
        stakeholders: list() = None,
        model_type: ModelType = ModelType.Completion,
    ):
        """ 
        Note: stakeholders is not included in the prompt but is used to construct the domains of expertise list.
        """
        expert_list: List[str] = list()
        for elements in domain_expertise_list:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)
        
        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_ivs"]
        suggest = guidance(prompt_template)

        ivs: List[str] = list()
        iv_edges: Dict[Tuple[str, str], int] = dict()
        iv_edges[(treatment, outcome)] = 1

        edited_factors_list: List[str] = []
        for i in range(len(factors_list)):
            if factors_list[i] != treatment and factors_list[i] != outcome:
                edited_factors_list.append(factors_list[i])

        if len(expert_list) > 1:
            for expert in expert_list:
                self.request_ivs(
                    suggest=suggest,
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    edited_factors_list=edited_factors_list,
                    temperature=temperature,
                    llm=llm,
                    iv_edges=iv_edges,
                )
        else:
            self.request_ivs(
                suggest=suggest,
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expert_list[0],
                edited_factors_list=edited_factors_list,
                temperature=temperature,
                llm=llm,
                iv_edges=iv_edges,
            )

        return iv_edges, ivs

    def request_ivs(
        self,
        suggest,
        treatment,
        outcome,
        analysis_context,
        domain_expertise,
        edited_factors_list,
        temperature,
        llm,
        iv_edges,
    ):
        ivs: List[str] = list()

        success: bool = False

        while not success:
            try:
                output = suggest(
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    factors_list=edited_factors_list,
                    factor=factor,
                    temperature=temperature,
                    llm=llm,
                )
                iv_factors = re.findall(
                    r"<iv_factor>(.*?)</iv_factor>",
                    output["output"],
                )

                if iv_factors:
                    for factor in iv_factors:
                        # to not add it twice into the list
                        if factor in edited_factors_list and factor not in ivs:
                            ivs.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

            for element in ivs:
                if (element, treatment) in iv_edges:
                    iv_edges[(element, treatment)] += 1
                else:
                    iv_edges[(element, treatment)] = 1

        return iv_edges
