from typing import List, Dict, Set, Tuple, Protocol
from ..protocols import IdentifierProtocol
from ..helpers import RelationshipStrategy, ModelType
from .model_suggester import ModelSuggester
from ..prompts import prompts as ps
import guidance
from guidance import system, user, assistant, gen
import re


# from dowhy import causal_identifier as ci


class IdentificationSuggester(IdentifierProtocol):
    # EXPERTS: list() = [
    #     "cause and effect",
    #     "causality, you are an intelligent AI with expertise in causality",
    # ]
    CONTEXT: str = """causal mechanisms"""

    def __init__(self, llm):
        if llm == 'gpt-4':
            self.llm = guidance.models.OpenAI('gpt-4')
            self.model_suggester = ModelSuggester('gpt-4')

    # def suggest_estimand(
    #     self,
    #     treatment: str,
    #     outcome: str,
    #     factors_list: list(),
    #     llm: guidance.models,
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
            expertise_list: list(),
            analysis_context: list() = CONTEXT,
            stakeholders: list() = None
    ):
        backdoor_set = self.model_suggester.suggest_confounders(
            treatment=treatment,
            outcome=outcome,
            factors_list=factors_list,
            expertise_list=expertise_list,
            analysis_context=analysis_context,
            stakeholders=stakeholders
        )
        return backdoor_set

    #TODO:implement
    def suggest_frontdoor(
            self,
            treatment: str,
            outcome: str,
            factors_list: list(),
            expertise_list: list(),
            analysis_context: list() = CONTEXT,
            stakeholders: list() = None
    ):
        pass

    def suggest_mediators(
            self,
            treatment: str,
            outcome: str,
            factors_list: list(),
            experts: list(),
            analysis_context: list() = CONTEXT,
            stakeholders: list() = None
    ):
        expert_list: List[str] = list()
        for elements in experts:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

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
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    factors_list=edited_factors_list,
                    mediators_edges=mediators_edges
                )
                for m in mediators_list:
                    if m not in mediators:
                        mediators.append(m)
        else:
            mediators_edges, mediators_list = self.request_mediators(
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expert_list[0],
                factors_list=edited_factors_list,
                mediators_edges=mediators_edges,
            )

            for m in mediators_list:
                if m not in mediators:
                    mediators.append(m)

        return mediators_edges, mediators

    def request_mediators(
            self,
            treatment,
            outcome,
            analysis_context,
            domain_expertise,
            factors_list,
            mediators_edges
    ):
        mediators: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are studying {analysis_context}. You are using your 
                knowledge to help build a causal model that contains all the assumptions about the factors that are directly 
                influencing athe {outcome}. Where a causal model is a conceptual model that describes the causal mechanisms 
                of a system. You will do this by by answering questions about cause and effect and using your domain knowledge 
                in {domain_expertise}. Follow the next two steps, and complete the first one before moving on to the second:"""

                with user():
                    lm += f"""(1) From your perspective as an expert in {domain_expertise}, think step by step as you consider the factors 
                that may interact between the {treatment} and the {outcome}. Use your knowledge as an expert in 
                {domain_expertise} to describe the mediators, if there are any at all, between {treatment} and the 
                {outcome}. Be concise and keep your thinking within two paragraphs. Then provide your step by step chain 
                of thoughts within the tags <thinking></thinking>.  (2) From your perspective as an expert in {domain_expertise},
                 which factor(s) of the following factors, if any at all, has/have a high likelihood of directly influencing and 
                 causing the assignment of the {outcome} and also has/have a high likelihood of being directly influenced and 
                 caused by the assignment of the {treatment}? Which factor(s) of the following factors, if any at all, is/are 
                 on the causal chain that links the {treatment} to the {outcome}? From your perspective as an expert in 
                 {domain_expertise}, which factor(s) of the following factors, if any at all, mediates, is/are on the causal 
                 chain, that links the {treatment} to the {outcome}? Then provide your step by step chain of thoughts within 
                 the tags <thinking></thinking>. factor_names : {factors_list} Wrap the name of the factor(s), if any at all, 
                 that has/have a high likelihood of directly influencing and causing the assignment of the {outcome} and also 
                 has/have a high likelihood of being directly influenced and caused by the assignment of the {treatment} within
                  the tags <mediating_factor>factor_name</mediating_factor>. Where factor_name is one of the items within the 
                  factor_names list. If a factor does not have a high likelihood of mediating, then do not wrap the factor with 
                  any tags. Your step by step answer as an in {domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                mediating_factor = re.findall(
                    r"<mediating_factor>(.*?)</mediating_factor>", output)

                if mediating_factor:
                    for factor in mediating_factor:
                        # to not add it twice into the list
                        if factor in factors_list and factor not in mediators:
                            mediators.append(factor)
                    success = True
                else:
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
            expertise_list: list(),
            analysis_context: list() = CONTEXT,
            stakeholders: list() = None
    ):
        expert_list: List[str] = list()
        for elements in expertise_list:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

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
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    factors_list=edited_factors_list,
                    iv_edges=iv_edges,
                )
        else:
            self.request_ivs(
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expert_list[0],
                factors_list=edited_factors_list,
                iv_edges=iv_edges,
            )

        return iv_edges, ivs

    def request_ivs(
            self,
            treatment,
            outcome,
            analysis_context,
            domain_expertise,
            factors_list,
            iv_edges
    ):
        ivs: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are studying {analysis_context}. 
                        You are using your knowledge to help build a causal model that contains all the assumptions about the factors 
                        that are directly influencing the {outcome}. Where a causal model is a conceptual model that describes the 
                        causal mechanisms of a system. You will do this by by answering questions about cause and effect and using 
                        your domain knowledge in {domain_expertise}. Follow the next two steps, and complete the first one before 
                        moving on to the second:"""

                with user():
                    lm += f"""(1) From your perspective as an expert in {domain_expertise}, think step by step 
                       as you consider the factors that may interact with the {treatment} and do not interact with {outcome}. 
                       Use your knowlegde as an expert in {domain_expertise} to describe the instrumental variable(s), 
                       if there are any at all, that both has/have a high likelihood of influecing and causing the {treatment} and 
                       has/have a very low likelihood of influencing and causing the {outcome}. Be concise and keep your thinking 
                       within two paragraphs.  Then provide your step by step chain of thoughts within the tags 
                       <thinking></thinking>. (2) From your perspective as an expert in {domain_expertise}, which factor(s) of the 
                       following factors, if there are any at all, both has/have a high likelihood of influecing and causing the {
                    treatment} and has/have a very low likelihood of influencing and causing the {outcome}? Which factor(s) of 
                       the following factors, if any at all, has/have a causal link to the {treatment} and has not causal link to 
                       the {outcome}? Which factor(s) of the following factors, if any at all, are (an) instrumental variable(s) 
                       to the causal relationship of the {treatment} causing the {outcome}? Be concise and keep your thinking 
                       within two paragraphs. Then provide your step by step chain of thoughts within the tags 
                       <thinking></thinking>. factor_names : {factors_list} Wrap the name of the factor(s), if there are any at 
                       all, that both has/have a high likelihood of influecing and causing the {treatment} and has/have a very low 
                       likelihood of influencing and causing the {outcome}, within the tags <iv_factor>factor_name</iv_factor>. 
                       Where factor_name is one of the items within the factor_names list. If a factor does not have a high 
                       likelihood of being an instrumental variable, then do not wrap the factor with any tags. Your step by step 
                       answer as an in {domain_expertise}:"""
                with assistant():
                    lm += gen("output")

                output = lm["output"]
                iv_factors = re.findall(r"<iv_factor>(.*?)</iv_factor>", output)

                if iv_factors:
                    for factor in iv_factors:
                        if factor in factors_list and factor not in ivs:
                            ivs.append(factor)
                    success = True
                else:
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
