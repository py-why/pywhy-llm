from typing import List, Tuple, Dict, Set
from suggesters.protocols import IdentifierProtocol
from .helpers import RelationshipStrategy, ModelType
from .prompts import prompts as ps
import networkx as nx
import guidance
import copy
import re


class ValidationSuggester(IdentifierProtocol):
    EXPERTS: list() = [
        "causality, you are an intelligent AI with expertise in causality",
        "answering questions about causality, you are a helpful causality assistant ",
    ]
    CONTEXT: str = """causal mechanisms"""

    def suggest_negative_controls(
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

        suggest = guidance(ps[model_type.value]["expert_suggests_negative_controls"])

        negative_controls_counter: Dict[str, int] = dict()
        negative_controls: List[str] = list()

        edited_factors_list: List[str] = []
        for i in range(len(factors_list)):
            if factors_list[i] != treatment and factors_list[i] != outcome:
                edited_factors_list.append(factors_list[i])

        if len(expert_list) > 1:
            for expert in expert_list:
                (
                    negative_controls_counter,
                    negative_controls_list,
                ) = self.request_negative_controls(
                    treatment=treatment,
                    outcome=outcome,
                    program=suggest,
                    edited_factors_list=edited_factors_list,
                    negative_controls_counter=negative_controls_counter,
                    llm=llm,
                    expert=expert,
                    analysis_context=analysis_context,
                    temperature=temperature,
                )
                for m in negative_controls_list:
                    if m not in negative_controls:
                        negative_controls.append(m)
        else:
            (
                negative_controls_counter,
                negative_controls_list,
            ) = self.request_negative_controls(
                treatment=treatment,
                outcome=outcome,
                program=suggest,
                edited_factors_list=edited_factors_list,
                negative_controls_counter=negative_controls_counter,
                llm=llm,
                expert=expert,
                analysis_context=analysis_context,
                temperature=temperature,
            )
            for m in negative_controls_list:
                if m not in negative_controls:
                    negative_controls.append(m)

        return negative_controls_counter, negative_controls

    def request_negative_controls(
        self,
        treatment: str,
        outcome: str,
        program,
        edited_factors_list: list(),
        negative_controls_counter: list(),
        llm: guidance.llms,
        expert: str = EXPERTS[0],
        analysis_context: list() = CONTEXT,
        temperature=0.3,
    ):
        negative_controls_list: List[str] = list()

        success: bool = False
        while not success:
            try:
                output = program(
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    factors_list=edited_factors_list,
                    treatment=treatment,
                    outcome=outcome,
                    temperature=temperature,
                    llm=llm,
                )
                negative_controls = re.findall(
                    r"<negative_control>(.*?)</negative_control>",
                    output["output"],
                )

                if negative_controls:
                    for factor in negative_controls:
                        # to not add it twice into the list
                        if (
                            factor in edited_factors_list
                            and factor not in negative_controls_list
                        ):
                            negative_controls_list.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

            for element in negative_controls_list:
                if element in negative_controls_counter:
                    negative_controls_counter[element] += 1
                else:
                    negative_controls_counter[element] = 1
        return negative_controls_counter, negative_controls_list

    def suggest_latent_confounders(
        self,
        treatment: str,
        outcome: str,
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

        suggest = guidance(ps[model_type.value]["expert_suggests_latent_confounders"])

        latent_confounders_counter: Dict[str, int] = dict()
        latent_confounders: List[str, str] = list()

        if len(expert_list) > 1:
            for expert in expert_list:
                (
                    latent_confounders_counter,
                    latent_confounders_list,
                ) = self.request_latent_confounders(
                    treatment=treatment,
                    outcome=outcome,
                    program=suggest,
                    latent_confounders_counter=latent_confounders_counter,
                    llm=llm,
                    expert=expert,
                    analysis_context=analysis_context,
                    temperature=temperature,
                )
                for m in latent_confounders_list:
                    if m not in latent_confounders:
                        latent_confounders.append(m)
        else:
            (
                latent_confounders_counter,
                latent_confounders_list,
            ) = self.request_latent_confounders(
                treatment=treatment,
                outcome=outcome,
                program=suggest,
                latent_confounders_counter=latent_confounders_counter,
                llm=llm,
                expert=expert,
                analysis_context=analysis_context,
                temperature=temperature,
            )
            for m in latent_confounders_list:
                if m not in latent_confounders:
                    latent_confounders.append(m)

        return latent_confounders_counter, latent_confounders

    def request_latent_confounders(
        self,
        treatment: str,
        outcome: str,
        program,
        latent_confounders_counter: list(),
        llm: guidance.llms,
        expert: str = EXPERTS[0],
        analysis_context: list() = CONTEXT,
        temperature=0.3,
    ):
        latent_confounders_list: List[str] = list()

        success: bool = False
        while not success:
            try:
                output = program(
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    treatment=treatment,
                    outcome=outcome,
                    temperature=temperature,
                    llm=llm,
                )
                latent_confounders = re.findall(
                    r"<confounding_factor>(.*?)</confounding_factor>",
                    output["output"],
                )

                if latent_confounders:
                    for factor in latent_confounders:
                        latent_confounders_list.append(factor)
                    success = True
                else:
                    llm.OpenAI.cache.clear()
                    success = False

            except KeyError:
                success = False
                continue

            for element in latent_confounders_list:
                if element in latent_confounders_counter:
                    latent_confounders_counter[element] += 1
                else:
                    latent_confounders_counter[element] = 1
        return latent_confounders_counter, latent_confounders_list

    def request_parent_critique(
        self,
        analysis_context,
        factor,
        factors_list,
        expert,
        llm: guidance.llms,
        temperature=0.3,
        model_type=ModelType.Completion,
    ):
        suggest = guidance(ps[model_type.value]["expert_critiques_parents"])

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
                    domain_expertise=expert,
                    potential_factors_list=edited_factors_list,
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

    def request_children_critique(
        self,
        analysis_context,
        factor,
        factors_list,
        expert,
        llm: guidance.llms,
        temperature=0.3,
        model_type=ModelType.Completion,
    ):
        suggest = guidance(ps[model_type.value]["expert_critiques_children"])

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
                    domain_expertise=expert,
                    potential_factors_list=edited_factors_list,
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

    def request_pairwise_critique(
        self,
        expert,
        factor_a: str,
        factor_b: str,
        llm: guidance.llms,
        temperature=0.3,
        analysis_context: str = CONTEXT,
        model_type=ModelType.Completion,
    ):
        suggest = guidance(ps[model_type.value]["expert_critiques_pairwise"])

        success: bool = False

        while not success:
            try:
                output = suggest(
                    analysis_context=analysis_context,
                    domain_expertise=expert,
                    a=factor_a,
                    b=factor_b,
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

    def critique_graph(
        self,
        factors_list: List[str],
        edges: List[Tuple[str, str]],
        llm: guidance.llms,
        experts: list() = EXPERTS,
        analysis_context: str = CONTEXT,
        stakeholders: list() = None,
        temperature=0.3,
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
                        suggested_parent = self.request_parent_critique(
                            analysis_context=analysis_context,
                            factor=factor,
                            factors_list=factors_list,
                            expert=expert,
                            llm=llm,
                            temperature=temperature,
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
                    suggested_parent = self.request_parent_critique(
                        analysis_context=analysis_context,
                        factor=factor,
                        factors_list=factors_list,
                        expert=expert_list[0],
                        llm=llm,
                        temperature=temperature,
                        model_type=model_type,
                    )

                    for element in suggested_parent:
                        if (element, factor) in parent_edges:
                            parent_edges[(element, factor)] += 1
                        else:
                            parent_edges[(element, factor)] = 1

            return edges, parent_edges

        elif relationship_strategy == RelationshipStrategy.Child:
            "loop asking children program"

            critiqued_children_edges: Dict[Tuple[str, str], int] = dict()

            for factor in factors_list:
                if len(expert_list) > 1:
                    for expert in expert_list:
                        suggested_children = self.request_children_critique(
                            analysis_context=analysis_context,
                            factor=factor,
                            factors_list=factors_list,
                            expert=expert,
                            llm=llm,
                            temperature=temperature,
                            model_type=model_type,
                        )
                        for element in suggested_children:
                            if (
                                (
                                    element,
                                    factor,
                                )
                                in critiqued_children_edges
                                and element in factors_list
                            ):
                                critiqued_children_edges[(element, factor)] += 1
                            else:
                                critiqued_children_edges[(element, factor)] = 1
                else:
                    suggested_children = self.request_children_critique(
                        analysis_context=analysis_context,
                        factor=factor,
                        factors_list=factors_list,
                        expert=expert_list[0],
                        llm=llm,
                        temperature=temperature,
                        model_type=model_type,
                    )

                    for element in suggested_children:
                        if (element, factor) in critiqued_children_edges:
                            critiqued_children_edges[(element, factor)] += 1
                        else:
                            critiqued_children_edges[(element, factor)] = 1

            return edges, critiqued_children_edges

        elif relationship_strategy == RelationshipStrategy.Pairwise:
            "loop through all pairs asking critique for edge"

            critiqued_pairwise_edges: Dict[Tuple[str, str], int] = dict()

            for factor_a in factors_list:
                for factor_b in factors_list:
                    if factor_a != factor_b:
                        if len(expert_list) > 1:
                            for expert in expert_list:
                                suggested_edge = self.request_pairwise_critique(
                                    analysis_context=analysis_context,
                                    factor_a=factor_a,
                                    factor_b=factor_b,
                                    expert=expert,
                                    llm=llm,
                                    temperature=temperature,
                                    model_type=model_type,
                                )

                                if suggested_edge is not None:
                                    if suggested_edge in critiqued_pairwise_edges:
                                        critiqued_pairwise_edges[suggested_edge] += 1
                                    else:
                                        critiqued_pairwise_edges[suggested_edge] = 1
                        else:
                            suggested_edge = self.request_pairwise_critique(
                                analysis_context=analysis_context,
                                factor_a=factor_a,
                                factor_b=factor_b,
                                expert=expert_list[0],
                                llm=llm,
                                temperature=temperature,
                                model_type=model_type,
                            )

                            if suggested_edge is not None:
                                if suggested_edge in critiqued_pairwise_edges:
                                    critiqued_pairwise_edges[suggested_edge] += 1
                                else:
                                    critiqued_pairwise_edges[suggested_edge] = 1

            return edges, critiqued_pairwise_edges
