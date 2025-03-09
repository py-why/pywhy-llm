import itertools
from typing import List, Tuple, Dict, Set
from ..protocols import IdentifierProtocol
from ..helpers import RelationshipStrategy, ModelType
from ..prompts import prompts as ps
import networkx as nx
import guidance
from guidance import system, user, assistant, gen
import copy
import re


class ValidationSuggester(IdentifierProtocol):
    # EXPERTS: list() = [
    #     "causality, you are an intelligent AI with expertise in causality",
    #     "answering questions about causality, you are a helpful causality assistant ",
    # ]
    CONTEXT: str = """causal mechanisms"""

    def __init__(self, llm):
        if llm == 'gpt-4':
            self.llm = guidance.models.OpenAI('gpt-4')

    def suggest_negative_controls(
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
                    factors_list=edited_factors_list,
                    negative_controls_counter=negative_controls_counter,
                    domain_expertise=expert,
                    analysis_context=analysis_context
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
                factors_list=edited_factors_list,
                negative_controls_counter=negative_controls_counter,
                domain_expertise=expert_list[0],
                analysis_context=analysis_context
            )
            for m in negative_controls_list:
                if m not in negative_controls:
                    negative_controls.append(m)

        return negative_controls_counter, negative_controls

    def request_negative_controls(
            self,
            treatment: str,
            outcome: str,
            factors_list: list(),
            negative_controls_counter: list(),
            domain_expertise: str,
            analysis_context: list() = CONTEXT
    ):
        negative_controls_list: List[str] = list()

        success: bool = False
        while not success:
            try:
                lm = self.llm

                with system():
                    lm += f"""You are an expert in the {domain_expertise} and are 
        studying the {analysis_context}. You are using your domain knowledge to help understand the negative 
        controls for a causal model that contains all the assumptions about the {analysis_context}. Where a causal 
        model is a conceptual model that describes the causal mechanisms of a system. You will do this by answering 
        questions about cause and effect using your domain knowledge in the {domain_expertise}."""

                with user():
                    lm += f"""factor_names: {factors_list} From your
                         perspective as an expert in the {domain_expertise}, what factor(s) from the list of factors, relevant to 
                         the {analysis_context}, should see zero treatment effect when changing the {treatment}? Which factor(s) 
                         from the list of factors, if any at all, relevant to the {analysis_context}, are negative controls on the 
                         causal mechanisms that affect the {outcome} when changing {treatment}? Using your domain knowledge, 
                         which factor(s) from the list of factors, if any at all, relevant to the {analysis_context}, 
                         should we expect to be unaffected by any changes in {treatment}? Which factor(s) from the list of factors, 
                         if any at all, would be surprising if affected by a change in {treatment}? Be concise and keep your 
                         thoughts under two paragraphs. Then provide your step by step chain of thoughts within the tags 
                         <thinking></thinking>. Once you have thought things through, wrap the name of the factor(s) from the list of 
                         factors, that has/have a high likelihood of being negative controls on the causal mechanisms that affect {outcome}
                         when changing {treatment}, within the tags <negative_control>factor_name</negative_control>. Wrap the name 
                         of the factor(s) from the list of factors, that has/have a high likelihood of being unaffected when 
                         changing {treatment}, within the tags <negative_control>factor_name</negative_control>. Where factor_name 
                         is one of the items within the factor_names list. If a factor does not have a high likelihood of being a 
                         negative control relevant to the {analysis_context}, then do not wrap the factor with any tags. Provide 
                         your step by step answer as an expert in the {domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]
                negative_controls = re.findall(
                    r"<negative_control>(.*?)</negative_control>", output)

                if negative_controls:
                    for factor in negative_controls:
                        # to not add it twice into the list
                        if (
                                factor in factors_list
                                and factor not in negative_controls_list
                        ):
                            negative_controls_list.append(factor)
                    success = True
                else:
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
                    latent_confounders_counter=latent_confounders_counter,
                    domain_expertise=expert,
                    analysis_context=analysis_context,
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
                latent_confounders_counter=latent_confounders_counter,
                domain_expertise=expert_list[0],
                analysis_context=analysis_context)
            for m in latent_confounders_list:
                if m not in latent_confounders:
                    latent_confounders.append(m)

        return latent_confounders_counter, latent_confounders

    def request_latent_confounders(
            self,
            treatment: str,
            outcome: str,
            latent_confounders_counter: list(),
            domain_expertise: str,
            analysis_context: list() = CONTEXT
    ):
        latent_confounders_list: List[str] = list()

        success: bool = False
        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are an expert in the {domain_expertise} and are 
                        studying the {analysis_context}. You are using your knowledge to help build a causal model that contains 
                        all the assumptions about the {domain_expertise}. Where a causal model is a conceptual model that describes 
                        the causal mechanisms of a system. You will do this by by answering questions about cause and effect and 
                        using your domain knowledge in the {domain_expertise}."""
                with user():
                    lm += f"""(1) From your perspective as 
                         an expert in the {domain_expertise}, think step by step as you consider the factors that may interact 
                         between the {treatment} and the {outcome}. Use your knowledge as an expert in the {domain_expertise} to 
                         describe the confounders, if there are any at all, between the {treatment} and the {outcome}. Be concise 
                         and keep your thinking within two paragraphs. Then provide your step by step chain of thoughts within the 
                         tags <thinking></thinking>. (2) From your perspective as an expert in the {domain_expertise}, which factor(
                         s), if any at all, has/have a high likelihood of directly influencing and causing both the assignment of the 
                         {treatment} and the {outcome}? Which factor(s), if any at all, have a causal chain that links the {treatment}
                         to the {outcome}? Which factor(s), if any at all, are a confounder to the causal relationship 
                         between the {treatment} and the {outcome}? Be concise and keep your thinking within two paragraphs. Then 
                         provide your step by step chain of thoughts within the tags <thinking></thinking>. Wrap the name of the 
                         factor(s), if any at all, that has/have a high likelihood of directly influencing and causing both the 
                         {treatment} and the {outcome}, within the tags <confounding_factor>factor_name</confounding_factor>. If a 
                         factor does not have a high likelihood of directly confounding, then do not wrap the factor with any tags. 
                         Your step by step answer as an expert in the {domain_expertise}:"""

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                latent_confounders = re.findall(
                    r"<confounding_factor>(.*?)</confounding_factor>", output)

                if latent_confounders:
                    for factor in latent_confounders:
                        latent_confounders_list.append(factor)
                    success = True
                else:
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
            domain_expertise
    ):

        edited_factors_list: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] not in factor:
                edited_factors_list.append(factors_list[i])

        parents: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are a helpful causal assistant and expert in {domain_expertise}, 
                        studying {analysis_context}. Task: identify factors causing {factor}."""
                with user():
                    lm += f"""Steps: (1) 
                        Analyze potential factors [{factors_list}] for factors directly influencing/causing/affecting {
                    factor}. Is relationship direct? Ignore feedback mechanisms/factors not in list. Keep thoughts within 
                        <thinking></thinking> tags. (2) Use prior thoughts to answer: how {factor} influenced/caused/affected by  [
                        {factors_list}]? Is relationship direct? Ignore feedback mechanisms/factors not in list. Wrap 
                        factors highly likely directly influencing/causing/affecting {factor} in 
                        <influencing_factor></influencing_factor> tags. No tags for low likelihood factors. Ignore feedback 
                        mechanisms/factors not in list. Answer as {domain_expertise} expert."""
                with assistant():
                    lm += gen("output")

                output = lm["output"]
                influencing_factors = re.findall(
                    r"<influencing_factor>(.*?)</influencing_factor>", output)

                if influencing_factors:
                    for factor in influencing_factors:
                        if factor in edited_factors_list and factor not in parents:
                            parents.append(factor)
                    success = True
                else:
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
            domain_expertise
    ):

        edited_factors_list: List[str] = []

        for i in range(len(factors_list)):
            if factors_list[i] not in factor:
                edited_factors_list.append(factors_list[i])

        children: List[str] = list()

        success: bool = False

        while not success:
            try:
                lm = self.llm

                with system():
                    lm += f"""You are a helpful causal assistant and expert in {domain_expertise}, 
                        studying {analysis_context}. Task: identify factors caused by {factor}."""

                with user():
                    lm += f"""Steps: (
                        1) Analyze potential factors [{factors_list}] for factors directly influenced/caused/affected by 
                        {factor}. Is relationship direct? Ignore feedback mechanisms/factors not in list. Keep thoughts within 
                        <thinking></thinking> tags. (2) Use prior thoughts to answer: how {factor} influences/causes/affects [{
                    factors_list}]? Is relationship direct? Ignore feedback mechanisms/factors not in list. Wrap 
                        factors highly likely directly influenced/caused/affected by {factor} in 
                        <influenced_factor></influenced_factor> tags. No tags for low likelihood factors. Ignore feedback 
                        mechanisms/factors not in list. Answer as {domain_expertise} expert."""

                with assistant():
                    lm += gen("output")

                output = lm["output"]
                influencing_factors = re.findall(
                    r"<influenced_factor>(.*?)</influenced_factor>", output)

                if influencing_factors:
                    for factor in influencing_factors:
                        if factor in edited_factors_list and factor not in children:
                            children.append(factor)

                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return children

    def request_pairwise_critique(
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
                    lm += f"""You are a helpful causal assistant, expert in {domain_expertise}, 
                        studying {analysis_context}. Task: identify relationship between {factor_a} and {factor_b}."""

                with user():
                    lm += f"""Steps: (1) Does {factor_a} influence/cause/affect {factor_b}? Is relationship direct? Does {factor_b} influence/cause/affect 
                        {factor_a}? Is relationship direct? Ignore feedback mechanisms/factors not in list. Keep thoughts within 
                        <thinking></thinking> tags. (2) Use prior thoughts to select likely answer: (A) {factor_a} influences {factor_b} (B) {
                    factor_b} influences {factor_a} (C) Neither. Wrap answer in <answer></answer>. e.g. <answer>A</answer>, 
                        <answer>B</answer>, <answer>C</answer>. No tags for low likelihood factors. Ignore feedback 
                        mechanisms/factors not in list. Answer as {domain_expertise} expert."""

                with assistant():
                    lm += gen("output")

                output = lm["output"]
                print(output)

                answer = re.findall(
                    r"<answer>(.*?)</answer", output)

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

    def critique_graph(
            self,
            factors_list: List[str],
            edges: Dict[Tuple[str, str], int],
            experts: list(),
            analysis_context: str = CONTEXT,
            stakeholders: list() = None,
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
                            domain_expertise=expert
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
                        domain_expertise=expert_list[0]
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
                            domain_expertise=expert
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
                        domain_expertise=expert_list[0]
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

            for (factor_a, factor_b) in itertools.combinations(factors_list, 2):
                if factor_a != factor_b:
                    if len(expert_list) > 1:
                        for expert in expert_list:
                            suggested_edge = self.request_pairwise_critique(
                                analysis_context=analysis_context,
                                factor_a=factor_a,
                                factor_b=factor_b,
                                domain_expertise=expert
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
                            domain_expertise=expert_list[0]
                        )

                        if suggested_edge is not None:
                            if suggested_edge in critiqued_pairwise_edges:
                                critiqued_pairwise_edges[suggested_edge] += 1
                            else:
                                critiqued_pairwise_edges[suggested_edge] = 1

            return edges, critiqued_pairwise_edges
