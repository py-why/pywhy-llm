from typing import Set, Tuple, Dict, List
from suggesters.protocols import ModelerProtocol
import networkx as nx
import guidance
from .helpers import RelationshipStrategy, ModelType
import copy
import random
from enum import Enum
from .prompts import prompts as ps
import os
import re
import csv


class ModelSuggester(ModelerProtocol):
    EXPERTS: list() = [
        "answering questions about causality, you are a helpful causality assistant ",
        "causality, you are an intelligent AI with expertise in causality",
        "cause and effect",
    ]
    CONTEXT: str = """causal mechanisms"""

    def suggest_domain_expertises(
        self,
        analysis_context,
        factors_list,
        llm: guidance.llms,
        n_experts: int = 1,
        temperature=0.3,
        prompt_template: str = None,
        model_type: ModelType = ModelType.Completion,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["expertises"])
        suggest = guidance(prompt_template)

        expertise_list: List[str] = list()
        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    factors_list=factors_list,
                    n_experts=n_experts,
                    temperature=temperature,
                    llm=llm,
                )
                expertise = re.findall(
                    r"<domain_expertise>(.*?)</domain_expertise>", output["output"]
                )

                if expertise:
                    for i in range(n_experts):
                        expertise_list.append(expertise[i])
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

        return expertise_list

    def suggest_domain_experts(
        self,
        analysis_context,
        factors_list,
        llm: guidance.llms,
        n_experts: int = 5,
        temperature=0.3,
        prompt_template: str = None,
        model_type: ModelType = ModelType.Chat,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["domain_experts"]
        suggest = guidance(prompt_template)

        experts_list: Set[str] = set()
        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    factors_list=factors_list,
                    n_experts=n_experts,
                    temperature=temperature,
                    llm=llm,
                )
                experts = re.findall(
                    r"<domain_expert>(.*?)</domain_expert>", output["output"]
                )

                if experts:
                    for i in range(n_experts):
                        experts_list.add(experts[i])
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

        return experts_list

    def suggest_stakeholders(
        self,
        factors_list,
        llm: guidance.llms,
        n_experts: int = 5,  # must be > 1
        temperarure=0.3,
        analysis_context=CONTEXT,
        prompt_template: str = None,
        model_type: ModelType = ModelType.Chat,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["stakeholders"]
        suggest = guidance(prompt_template)

        stakeholder_list: List[str] = list()
        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    factors_list=factors_list,
                    n_experts=n_experts,
                    temperature=temperature,
                    llm=llm,
                )
                stakeholder = re.findall(
                    r"<stakeholder>(.*?)</stakeholder>", output["output"]
                )

                if stakeholder:
                    for i in range(n_experts):
                        stakeholder_list.append(stakeholder[i])
                    success = True
                else:
                    llm.OpenAI.cache.clear()
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
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: list() = CONTEXT,
        temperature=0.3,
        prompt_template: str = None,
        stakeholders: list() = None,
        model_type: ModelType = ModelType.Completion,
    ):
        expert_list: List[str] = list()
        for elements in experts:
            expert_list.append(elements)
        if stakeholders is not None:
            for elements in stakeholders:
                expert_list.append(elements)

        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_confounders"]
        suggest = guidance(prompt_template)

        confounders_edges: Dict[Tuple[str, str], int] = dict()
        confounders_edges[(treatment, outcome)] = 1

        confounders: List[str] = list()

        edited_factors_list: List[str] = []
        for i in range(len(factors_list)):
            if factors_list[i] != treatment and factors_list[i] != outcome:
                edited_factors_list.append(factors_list[i])

        if len(expert_list) > 1:
            for expert in expert_list:
                confounders_edges, confounders_list = self.request_confounders(
                    suggest=suggest,
                    treatment=treatment,
                    outcome=outcome,
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    edited_factors_list=edited_factors_list,
                    temperature=temperature,
                    llm=llm,
                    confounders_edges=confounders_edges,
                )

                for m in confounders_list:
                    if m not in confounders:
                        confounders.append(m)
        else:
            confounders_edges, confounders_list = self.request_confounders(
                suggest=suggest,
                treatment=treatment,
                outcome=outcome,
                analysis_context=analysis_context,
                domain_expertise=expert_list[0],
                edited_factors_list=edited_factors_list,
                temperature=temperature,
                llm=llm,
                confounders_edges=confounders_edges,
            )

            for m in confounders_list:
                if m not in confounders:
                    confounders.append(m)

        return confounders_edges, confounders

    def request_confounders(
        self,
        suggest,
        treatment,
        outcome,
        analysis_context,
        domain_expertise,
        edited_factors_list,
        temperature,
        llm,
        confounders_edges,
    ):
        confounders: List[str] = list()

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
                confounding_factors = re.findall(
                    r"<confounding_factor>(.*?)</confounding_factor>",
                    output["output"],
                )

                if confounding_factors:
                    for factor in confounding_factors:
                        # to not add it twice into the list
                        if factor in edited_factors_list and factor not in confounders:
                            confounders.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
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
        factor,
        factors_list,
        domain_expertise,
        llm: guidance.llms,
        temperature=0.3,
        prompt_template: str = None,
        model_type=ModelType.Completion,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_parents"]
        suggest = guidance(prompt_template)

        edited_factors_list: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] not in factor:
                edited_factors_list.append(factors_list[i])

        parents: List[str] = list()

        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    factors_list=edited_factors_list,
                    factor=factor,
                    temperature=temperature,
                    llm=llm,
                )
                influencing_factors = re.findall(
                    r"<influencing_factor>(.*?)</influencing_factor>",
                    output["output"],
                )

                if influencing_factors:
                    for factor in influencing_factors:
                        if factor in edited_factors_list and factor not in parents:
                            parents.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

        return parents

    def suggest_children(
        self,
        analysis_context,
        factor,
        factors_list,
        domain_expertise,
        llm: guidance.llms,
        temperature=0.3,
        prompt_template: str = None,
        model_type=ModelType.Completion,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_children"]
        suggest = guidance(prompt_template)

        edited_factors_list: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] not in factor:
                edited_factors_list.append(factors_list[i])

        children: List[str] = list()

        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    factors_list=edited_factors_list,
                    factor=factor,
                    temperature=temperature,
                    llm=llm,
                )
                influencing_factors = re.findall(
                    r"<influenced_factor>(.*?)</influenced_factor>",
                    output["output"],
                )

                if influencing_factors:
                    for factor in influencing_factors:
                        if factor in edited_factors_list and factor not in children:
                            children.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
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
        llm: guidance.llms,
        temperature=0.3,
        analysis_context: str = CONTEXT,
        prompt_template: str = None,
        model_type=ModelType.Completion,
    ):
        if prompt_template is None:
            prompt_template = ps[model_type.value]["expert_suggests_pairwise_relationships"]
        suggest = guidance(prompt_template)

        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    domain_expertise=domain_expertise,
                    factor_a=factor_a,
                    factor_b=factor_b,
                    temperature=temperature,
                    llm=llm,
                )
                answer = re.findall(
                    r"<answer>(.*?)</answer>",
                    output["output"],
                )

                if answer:
                    if answer[0] == "A" or answer[0] == "(A)":
                        return (factor_a, factor_b)

                    elif answer[0] == "B" or answer[0] == "(B)":
                        return (factor_b, factor_a)

                    elif answer[0] == "C" or answer[0] == "(C)":
                        return None
                    else:
                        llm.OpenAI.cache.clear()
                        success = False

                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

    def suggest_relationships(
        self,
        treatment: str,
        outcome: str,
        factors_list: list(),
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: str = CONTEXT,
        stakeholders: list() = None,
        temperature=0.3,
        prompt_template: str = None,
        model_type: ModelType = ModelType.Completion,
        relationship_strategy: RelationshipStrategy = RelationshipStrategy.Parent,
    ):
        expert_list: List[str] = list()
        for elements in experts:
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
                            factor=factor,
                            factors_list=factors_list,
                            domain_expertise=domain_expertise,
                            llm=llm,
                            temperature=temperature,
                            prompt_template=prompt_template,
                            model_type=model_type,
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
                        factor=factor,
                        factors_list=factors_list,
                        domain_expertise=expert_list[0],
                        llm=llm,
                        temperature=temperature,
                        prompt_template=prompt_template,
                        model_type=model_type,
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
                            factor=factor,
                            factors_list=factors_list,
                            domain_expertise=domain_expertise,
                            llm=llm,
                            temperature=temperature,
                            prompt_template=prompt_template,
                            model_type=model_type,
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
                        factor=factor,
                        factors_list=factors_list,
                        domain_expertise=expert_list[0],
                        llm=llm,
                        temperature=temperature,
                        prompt_template=prompt_template,
                        model_type=model_type,
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
                                    analysis_context=analysis_context,
                                    factor_a=factor_a,
                                    factor_b=factor_b,
                                    domain_expertise=domain_expertise,
                                    llm=llm,
                                    temperature=temperature,
                                    prompt_template=prompt_template,
                                    model_type=model_type,
                                )

                                if suggested_edge is not None:
                                    if suggested_edge in pairwise_edges:
                                        pairwise_edges[suggested_edge] += 1
                                    else:
                                        pairwise_edges[suggested_edge] = 1
                        else:
                            suggested_edge = self.suggest_pairwise_relationship(
                                analysis_context=analysis_context,
                                factor_a=factor_a,
                                factor_b=factor_b,
                                domain_expertise=expert_list[0],
                                llm=llm,
                                temperature=temperature,
                                prompt_template=prompt_template,
                                model_type=model_type,
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
                experts=experts,
                llm=llm,
                stakeholders=stakeholders,
                temperature=temperature,
                prompt_template=prompt_template,
                model_type=model_type,
            )

            return confounders_counter, confounders
