from typing import Set, Tuple, Dict, List
from ..protocols import ModelerProtocol
import networkx as nx
import guidance
from guidance import system, user, assistant, gen
from ..helpers import RelationshipStrategy, ModelType
import copy
import random
from enum import Enum
from ..prompts import prompts as ps
import os
import re
import csv


class ModelSuggester(ModelerProtocol):

    CONTEXT: str = """causal mechanisms"""

    def __init__(self, llm):
        if llm == 'gpt-4':
            self.llm = guidance.models.OpenAI('gpt-4')

    def suggest_domain_expertises(
            self,
            analysis_context,
            factors_list,
            n_experts: int = 1
    ):

        expertise_list: List[str] = list()
        success: bool = False

        while not success:
            try:
                lm = self.llm

                with system():
                    lm += f"""You are a helpful assistant for recommending useful domain expertises."""
                with user():
                    lm += f"""What domain expertises have the knowledge and experience needed to identify causal 
                    relationships and causal influences between the {analysis_context}? What domain expertises are needed 
                    to work with and reason about the causal influence between {factors_list}? What domain expertises 
                    have the knowledge and experience to reason and answer questions about influence and cause between 
                    such factors? Think about this in a step by step manner and recommend {n_experts} expertises and 
                    provide each one wrapped within the tags, <domain_expertise></domain_expertise>, along with the 
                    reasoning and explanation wrapped between the tags <explanation></explanation>."""
                with assistant():
                    lm += gen("output")

                output = lm["output"]
                expertise = re.findall(r"<domain_expertise>(.*?)</domain_expertise>", output)

                if expertise:
                    for i in range(n_experts):
                        expertise_list.append(expertise[i])
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return expertise_list

    def suggest_domain_experts(
            self,
            analysis_context,
            factors_list,
            n_experts: int = 5
    ):

        experts_list: Set[str] = set()
        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are a helpful assistant for recommending useful domain experts."""
                with user():
                    lm += f"""What domain experts have the knowledge and experience needed to identify causal relationships 
                    and causal influences between the {analysis_context}? What experts are needed to work with and 
                    reason about the causal influence between {factors_list}? What domain experts have the knowledge 
                    and experience to reason and answer questions about influence and cause between such factors? Think 
                    about this in a step by step manner and recommend {n_experts} domain experts and provide each one 
                    wrapped within the tags, <domain_expert></domain_expert>, along with the reasoning and explanation 
                    wrapped between the tags <explanation></explanation>."""
                with assistant():
                    lm += gen("output")

                output = lm["output"]
                experts = re.findall(r"<domain_expert>(.*?)</domain_expert>", output)

                if experts:
                    for i in range(n_experts):
                        experts_list.add(experts[i])
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return experts_list

    def suggest_stakeholders(
            self,
            factors_list,
            n_stakeholders: int = 5,  # must be > 1
            analysis_context=CONTEXT
    ):

        stakeholder_list: List[str] = list()
        success: bool = False

        while not success:
            try:
                lm = self.llm

                with system():
                    lm += "You are a helpful assistant for recommending useful primary and secondary stakeholders."

                with user():
                    lm += f"""What stakeholders have knowledge and experience in and about {analysis_context}? 
                                    What stakeholders can work best with and reason well about the causal influence between 
                {factors_list}? What stakeholders have the knowledge and experience useful to reason within this context? Think about 
                this in a step by step manner and recommend {n_stakeholders} stakeholders. Then provide each useful stakeholder 
                wrapped within the tags, <stakeholder></stakeholder>, along with the reasoning and explanation wrapped between the tags
                <explanation></explanation>."""
                with assistant():
                    lm += gen("output")

                output = lm["output"]

                stakeholder = re.findall(
                    r"<stakeholder>(.*?)</stakeholder>", output)

                if stakeholder:
                    for i in range(n_stakeholders):
                        stakeholder_list.append(stakeholder[i])
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return stakeholder_list

    def suggest_confounders(
            self,
            treatment: str,
            outcome: str,
            factors_list: list(),
            expertise_list: list(),
            analysis_context: str = CONTEXT,
            stakeholders: list() = None
    ):
        expert_list: List[str] = list()
        for elements in expertise_list:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expertise_list.append(elements)

        confounders_edges: Dict[Tuple[str, str], int] = dict()
        confounders_edges[(treatment, outcome)] = 1

        confounders: List[str] = list()

        edited_factors_list: List[str] = []
        for i in range(len(factors_list)):
            if factors_list[i] != treatment and factors_list[i] != outcome:
                edited_factors_list.append(factors_list[i])

        if len(expertise_list) > 1:
            for expert in expertise_list:
                confounders_edges, confounders_list = self.request_confounders(
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    factors_list=edited_factors_list,
                    confounders_edges=confounders_edges
                )

                for m in confounders_list:
                    if m not in confounders:
                        confounders.append(m)
        else:
            confounders_edges, confounders_list = self.request_confounders(
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expertise_list[0],
                factors_list=edited_factors_list,
                confounders_edges=confounders_edges
            )

            for m in confounders_list:
                if m not in confounders:
                    confounders.append(m)

        return confounders_edges, confounders

    def request_confounders(
            self,
            treatment,
            outcome,
            analysis_context,
            domain_expertise,
            factors_list,
            confounders_edges
    ):
        confounders: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are studying {analysis_context}.
                You are using your knowledge to help build a causal model that contains all the assumptions about {
                    analysis_context}. Where a causal model is a conceptual model that describes the causal mechanisms of a 
                        system. You
                will do this by by answering questions about cause and effect and using your domain knowledge in {domain_expertise}."""
                with user():
                    lm += f"""Follow the next two steps, and complete the first one before moving on to the second: (1) 
                                From your perspective as an 
                expert in {domain_expertise}, think step by step as you consider the factors that may interact between the {treatment} 
                and the {outcome}. Use your knowlegde as an expert in {domain_expertise} to describe the confounders, if there are any 
                at all, between the {treatment} and the {outcome}. Be concise and keep your thinking within two paragraphs. Then provide
                your step by step chain of thoughts within the tags <thinking></thinking>. (2) From your perspective as an expert in 
                {domain_expertise}, which factor(s) of the following factors, if any at all, has/have a high likelihood of directly 
                influencing and causing both the assignment of the {treatment} and the {outcome}? Which factor(s) of the following 
                factors, 
                if any at all, have a causal chain that links to the {treatment} to the {outcome}? Which factor(s) of the following 
                factors, 
                if any at all, are a confounder to the causal relationship between the {treatment} and the {outcome}? Be concise and 
                keep your 
                thinking within two paragraphs. Then provide your step by step chain of thoughts within the tags 
                <thinking></thinking>. \n factor_names : 
                {factors_list} Wrap the name of the factor(s), if any at all, that has/have a high likelihood of directly influencing 
                and causing both  the {treatment} and the {outcome}, within the tags 
                <confounding_factor>factor_name</confounding_factor> where 
                factor_name is one of the items within the factor_names list. If a factor does not have a high likelihood of directly 
                confounding, then do not wrap the factor with any tags."""
                with assistant():
                    lm += gen("output")

                output = lm["output"]
                confounding_factors = re.findall(r"<confounding_factor>(.*?)</confounding_factor>", output)

                if confounding_factors:
                    for factor in confounding_factors:
                        # to not add it twice into the list
                        if factor in factors_list and factor not in confounders:
                            confounders.append(factor)
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

            for element in confounders:
                if (element, treatment) in confounders_edges and (
                        element,
                        outcome,
                ) in confounders_edges:
                    confounders_edges[(element, treatment)] += 1
                    confounders_edges[(element, outcome)] += 1
                else:
                    confounders_edges[(element, treatment)] = 1
                    confounders_edges[(element, outcome)] = 1

        return confounders_edges, confounders

    def suggest_parents(
            self,
            analysis_context,
            domain_expertise,
            factor,
            factors_list
    ):
        parent_candidates: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] != factor:
                parent_candidates.append(factors_list[i])

        parents: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are studying {analysis_context}"""

                with user():
                    lm += f"""You are using your knowledge to help build a causal model that 
                            contains all the assumptions about the factors that are directly influencing 
                            and causing the {factor}. Where a causal model is a conceptual model that describes the 
                            causal mechanisms of a system. You will do this by by answering questions about cause and 
                            effect and using your domain knowledge as an expert in {domain_expertise}. Follow the next 
                            two steps, and complete the first one before moving on to the second: (1) From your 
                            perspective 
                            as an expert in {domain_expertise} think step by step as you consider the relevant factor 
                            directly influencing and causing the {factor}. Be concise and keep your thinking within two 
                            paragraphs. Then provide your step by step chain of thoughts within the tags 
                            <thinking></thinking>. 
                            (2) From your perspective as an expert in {domain_expertise} which of the following 
                            factors has 
                            a high likelihood of directly influencing and causing the {factor}? factors list: [
{factors_list}] 
                            For any factors within the list with a high likelihood of directly influencing and causing 
                            the {factor} wrap the name of the factor with the tags 
                            <influencing_factor>factor_name</influencing_factor>. 
                            If a factor does not have a high likelihood of directly influencing and causing the 
{factor}, 
                            then do not wrap the factor with any tags. Your answer as an expert in 
{domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                influencing_factors = re.findall(r"<influencing_factor>(.*?)</influencing_factor", output)

                if influencing_factors:
                    for influencing_factor in influencing_factors:
                        if influencing_factor in parent_candidates and influencing_factor not in parents:
                            parents.append(influencing_factor)
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return parents

    def suggest_children(
            self,
            analysis_context,
            domain_expertise,
            factor,
            factors_list
    ):

        children_candidates: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] != factor:
                children_candidates.append(factors_list[i])

        children: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are studying {analysis_context}"""
                with user():
                    lm += f"""You are using your knowledge to help build a causal model that 
                            contains all the assumptions about the factors that are directly influencing and causing the {factor}. 
                            Where a 
                            causal model is a conceptual model that describes the causal mechanisms of a system. You will do this by by 
                            answering questions about cause and effect and using your domain knowledge as an expert in {
                    domain_expertise}. 
                            Follow the next two steps, and complete the first one before moving on to the second: (1) From your 
                            perspective 
                            as an expert in {domain_expertise} think step by step as you consider which factor(s), from the factors 
                            list, 
                            if any at all, is/are directly influenced and caused by the {factor}. Be concise and keep your thinking 
                            within 
                            two paragraphs. Then provide your step by step chain of thoughts within the tags <thinking></thinking>. (2) 
                            From 
                            your perspective as an expert in {domain_expertise}, which of the following factor(s) from the factors 
                            list, 
                            if any at all, has/have a high likelihood of being directly influenced and caused by the {factor}? What 
                            factor(
                            s) from the factors list, if any at all, is/are affected by the {factor}? factors list: [{factors_list}] 
                            For 
                            any factors within the list with a high likelihood of being directly influenced and caused by the {factor}, 
                            wrap the name of the factor with the tags <influenced_factor>factor_name</influenced_factor>. If a factor 
                            has a 
                            high likelihood of being affected and influenced by the {factor}, then wrap the name of the factor with the 
                            tags <influencing_factor>factor_name</influencing_factor>. Where factor_name is one of the items within the 
                            factor_names list. If a factor does not have a high likelihood of directly influencing and causing the {
                    factor}, then do not wrap the factor with any tags. Your answer as an expert in 
                {domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                influenced_factors = re.findall(r"<influenced_factor>(.*?)</influenced_factor", output)

                if influenced_factors:
                    for influenced_factor in influenced_factors:
                        if influenced_factor in children_candidates and influenced_factor not in children:
                            children.append(influenced_factor)
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return children

    def suggest_pairwise_relationship(
            self,
            domain_expertise,
            factor_a: str,
            factor_b: str,
            analysis_context: str = CONTEXT
    ):

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in {domain_expertise} and are 
                            studying {analysis_context}. You are using your knowledge to help build a causal model that contains 
                            all the 
                            assumptions about {analysis_context}. Where a causal model is a conceptual model that describes the 
                            causal 
                            mechanisms of a system. You will do this by by answering questions about cause and effect and using your 
                            domain 
                            knowledge as an expert in {domain_expertise}."""
                with user():
                    lm += f"""From your perspective as an expert in {domain_expertise}, which of the following is 
                                    most likely true? (A) {factor_a} affects {factor_b}; {factor_a} has a high likelihood of directly 
                                    influencing {factor_b}; and {factor_a} causes {factor_b}. (B) {factor_b} affects {factor_a}; 
                {factor_b} has a high likelihood of directly influencing {factor_a}; and {factor_b} causes {factor_a}. (C) Neither A 
                nor B; {factor_a} does not cause {factor_b}; and {factor_b} does not cause {factor_a}. Select the answer that you as 
                an expert in {domain_expertise} believe has the likelihood of being true. Think step by step and provide your 
                thoughts within the tags <thinking></thinking>. Then select that answer A, B, or C, that is causally correct. When 
                you reach a conclusion, wrap your answer within the tags <answer></answer>. If you are done thinking, provide your 
                answer wrapped within the tags <answer></answer>. e.g. <answer>A</answer>, <answer>B</answer>, or <answer>C</answer>. 
                Your answer as an expert in {domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                answer = re.findall(r"<answer>(.*?)</answer", output)

                if answer:
                    if answer[0] == "A" or answer[0] == "(A)":
                        return factor_a, factor_b

                    elif answer[0] == "B" or answer[0] == "(B)":
                        return factor_b, factor_a

                    elif answer[0] == "C" or answer[0] == "(C)":
                        return None
                    else:
                        success = False

                else:
                    success = False

            except KeyError:
                success = False
                continue

    def suggest_relationships(
            self,
            treatment: str,
            outcome: str,
            factors_list: list(),
            expertise_list: list(),
            analysis_context: str = CONTEXT,
            stakeholders: list() = None,
            relationship_strategy: RelationshipStrategy = RelationshipStrategy.Parent
    ):
        expert_list: List[str] = list()
        for elements in expertise_list:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

        if relationship_strategy == RelationshipStrategy.Parent:
            "loop asking parents program"

            parent_edges: Dict[Tuple[str, str], int] = dict()

            for factor in factors_list:
                if len(expert_list) > 1:
                    for expert in expert_list:
                        suggested_parent = self.suggest_parents(
                            analysis_context=analysis_context,
                            domain_expertise=expert,
                            factor=factor,
                            factors_list=factors_list
                        )
                        for element in suggested_parent:
                            if (
                                    element,
                                    factor,
                            ) in parent_edges and element in factors_list:
                                parent_edges[(element, factor)] += 1
                            else:
                                parent_edges[(element, factor)] = 1
                else:
                    suggested_parent = self.suggest_parents(
                        analysis_context=analysis_context,
                        domain_expertise = expert_list[0],
                        factor=factor,
                        factors_list=factors_list,
                    )

                    for element in suggested_parent:
                        if (element, factor) in parent_edges:
                            parent_edges[(element, factor)] += 1
                        else:
                            parent_edges[(element, factor)] = 1

            return parent_edges

        elif relationship_strategy == RelationshipStrategy.Child:
            "loop asking children program"

            children_edges: Dict[Tuple[str, str], int] = dict()

            for factor in factors_list:
                if len(expert_list) > 1:
                    for expert in expert_list:
                        suggested_children = self.suggest_children(
                            analysis_context=analysis_context,
                            domain_expertise=expert,
                            factor=factor,
                            factors_list=factors_list
                        )
                        for element in suggested_children:
                            if (
                                    element,
                                    factor,
                            ) in children_edges and element in factors_list:
                                children_edges[(element, factor)] += 1
                            else:
                                children_edges[(element, factor)] = 1
                else:
                    suggested_children = self.suggest_parents(
                        analysis_context=analysis_context,
                        domain_expertise=expert_list[0],
                        factor=factor,
                        factors_list=factors_list
                    )

                    for element in suggested_children:
                        if (element, factor) in children_edges:
                            children_edges[(element, factor)] += 1
                        else:
                            children_edges[(element, factor)] = 1

            return children_edges

        elif relationship_strategy == RelationshipStrategy.Pairwise:
            "loop through all pairs asking relationship for"

            pairwise_edges: Dict[Tuple[str, str], int] = dict()

            for factor_a in factors_list:
                for factor_b in factors_list:
                    if factor_a != factor_b:
                        if len(expert_list) > 1:
                            for expert in expert_list:
                                suggested_edge = self.suggest_pairwise_relationship(
                                    domain_expertise=expert,
                                    analysis_context=analysis_context,
                                    factor_a=factor_a,
                                    factor_b=factor_b
                                )

                                if suggested_edge is not None:
                                    if suggested_edge in pairwise_edges:
                                        pairwise_edges[suggested_edge] += 1
                                    else:
                                        pairwise_edges[suggested_edge] = 1
                        else:
                            suggested_edge = self.suggest_pairwise_relationship(
                                domain_expertise=expert_list[0],
                                analysis_context=analysis_context,
                                factor_a=factor_a,
                                factor_b=factor_b
                            )

                            if suggested_edge is not None:
                                if suggested_edge in pairwise_edges:
                                    pairwise_edges[suggested_edge] += 1
                                else:
                                    pairwise_edges[suggested_edge] = 1

            return pairwise_edges

        elif relationship_strategy == RelationshipStrategy.Confounder:
            "one call to confounder program"

            confounders_counter, confounders = self.suggest_confounders(
                analysis_context=analysis_context,
                treatment=treatment,
                outcome=outcome,
                factors_list=factors_list,
                expertise_list=expertise_list,
            )

            return confounders_counter, confounders
