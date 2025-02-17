from typing import List, Dict, Set, Tuple, Protocol
from ..protocols import IdentifierProtocol
from ..helpers import RelationshipStrategy, ModelType
from .model_suggester import ModelSuggester
from ..prompts import prompts as ps
import guidance
import re

# from dowhy import causal_identifier as ci


class IdentificationSuggester(IdentifierProtocol):
    EXPERTS: list() = [
        "cause and effect",
        "causality, you are an intelligent AI with expertise in causality",
    ]
    CONTEXT: str = """causal mechanisms"""

    # def suggest_estimand(
    #     self,
    #     treatment: str,
    #     outcome: str,
    #     factors_list: list(),
    #     llm: guidance.llms,
    #     backdoor: Set[str] = None,
    #     frontdoor: Set[str] = None,
    #     ivs: Set[str] = None,
    #     experts: list() = EXPERTS,
    #     analysis_context: list() = CONTEXT,
    #     stakeholders: list() = None,
    #     temperature=0.3,
    #     model_type: ModelType = ModelType.Completion,
    #     estimand_type: ci.auto_identifier.auto_identifier.EstimandType = ci.auto_identifier.EstimandType.NONPARAMETRIC_ATE,
    # ):
    #     estimands_dict = {}

    #     if len(backdoor) > 0 and backdoor is not None:
    #         estimands_dict["backdoor"] = ci.construct_backdoor_estimand(
    #             treatment_name=treatment, outcome_name=outcome, common_causes=backdoor
    #         )
    #     else:
    #         backdoor_edges, backdoor_set = self.suggest_backdoor(
    #             treatment=treatment,
    #             outcome=outcome,
    #             factors_list=factors_list,
    #             llm=llm,
    #             experts=experts,
    #             analysis_context=analysis_context,
    #             stakeholders=stakeholders,
    #             temperature=temperature,
    #             model_type=model_type,
    #         )
    #         if len(backdoor_set) > 0:
    #             estimands_dict["backdoor"] = backdoor_set
    #         else:
    #             estimands_dict["backdoor"] = None

    #     if len(frontdoor) > 0 and frontdoor is not None:
    #         estimands_dict[
    #             "frontdoor"
    #         ] = ci.auto_identifier.construct_frontdoor_estimand(
    #             treatment_name=treatment,
    #             outcome_name=outcome,
    #             frontdoor_variables_names=frontdoor,
    #         )
    #     else:
    #         frontdoor_edges, frontdoor_set = self.suggest_frontdoor(
    #             treatment=treatment,
    #             outcome=outcome,
    #             factors_list=factors_list,
    #             llm=llm,
    #             experts=experts,
    #             analysis_context=analysis_context,
    #             stakeholders=stakeholders,
    #             temperature=temperature,
    #             model_type=model_type,
    #         )
    #         if len(frontdoor) > 0:
    #             estimands_dict["frontdoor"] = frontdoor_set
    #         else:
    #             estimands_dict["frontdoor"] = None

    #     if len(ivs) > 0 and ivs is not None:
    #         estimands_dict["iv"] = ci.auto_identifier.construct_iv_estimand(
    #             treatment_name=treatment, outcome_name=outcome, instrument_names=ivs
    #         )
    #     else:
    #         ivs_edges, ivs_set = self.suggest_ivs(
    #             treatment=treatment,
    #             outcome=outcome,
    #             factors_list=factors_list,
    #             llm=llm,
    #             experts=experts,
    #             analysis_context=analysis_context,
    #             stakeholders=stakeholders,
    #             temperature=temperature,
    #             model_type=model_type,
    #         )
    #         if len(frontdoor) > 0:
    #             estimands_dict["iv"] = ivs_set
    #         else:
    #             estimands_dict["iv"] = None

    #     estimand = ci.auto_identifier.IdentifiedEstimand(
    #         None,
    #         treatment_variable=treatment,
    #         outcome_variable=outcome,
    #         estimand_type=estimand_type,
    #         estimands=estimands_dict,
    #         backdoor_variables=backdoor,
    #         instrumental_variables=ivs,
    #         frontdoor_variables=frontdoor,
    #     )
    #     return estimand

    def suggest_backdoor(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        stakeholders: list() = None,
        temperature=0.3,
        model_type: ModelType = ModelType.Completion,
    ):
        backdoor_set = ModelSuggester.suggest_confounders(
            analysis_context=analysis_context,
            treatment=treatment,
            outcome=outcome,
            factors_list=factors_list,
            experts=experts,
            llm=llm,
            stakeholders=stakeholders,
            temperature=temperature,
            model_type=model_type,
        )
        return backdoor_set

    def suggest_mediators(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        stakeholders: list() = None,
        temperature=0.3,
        model_type: ModelType = ModelType.Completion,
    ):
        expert_list: List[str] = list()
        for elements in experts:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

        suggest = guidance(ps[model_type.value]["expert_suggests_mediators"])

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
                    expert=expert,
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
                expert=expert_list[0],
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
        expert,
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
                    domain_expertise=expert,
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
        experts: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        stakeholders: list() = None,
        temperature=0.3,
        model_type: ModelType = ModelType.Completion,
    ):
        expert_list: List[str] = list()
        for elements in experts:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

        suggest = guidance(ps[model_type.value]["expert_suggests_mediators"])

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
                    expert=expert,
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
                expert=expert_list[0],
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
        expert,
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
                    domain_expertise=expert,
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
