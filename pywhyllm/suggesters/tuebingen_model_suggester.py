from typing import List
import guidance
import re
from guidance import system, user, assistant, gen

import sys
from enum import Enum

from .. import ModelSuggester


class Strategy(Enum):
    Straight = 0
    CoT = 1
    ToT_Single = 2
    ToT_Multi = 2


class TuebingenModelSuggester(ModelSuggester):
    def __init__(self, llm):
        super().__init__(llm)

    def suggest_description(
            self, variable, context=None, ask_reference=False
    ):
        generate_description = self._build_description_program(variable)

        success: bool = False
        suggested_description: str

        while not success:
            lm = self.llm
            try:
                with system():
                    lm += generate_description["system"]

                with user():
                    lm += generate_description["user"]

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                description = re.findall(
                    r"<description>(.*?)</description>", output)

                if ask_reference:
                    reference = re.findall(
                        r"<reference>(.*?)</reference>", output)
                    success = True

                    return description, reference

                else:
                    success = True
                    return description

            except KeyError:
                success = False
                continue

    def suggest_onesided_relationship(
            self,
            variable_a,
            description_a,
            variable_b,
            description_b
    ):

        success: bool = False
        suggested_relationship: int

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += f"""You are a helpful assistant on causal reasoning. Your goal is to answer questions about 
                    cause and effect in a factual and concise way."""
                with user():
                    lm += f"""Can changing {variable_a}, {description_a}, change {variable_b}, {description_b}?
                        A. Yes
                        B. No
                        Within one sentence, let's think step-by-step to make sure that we have the right answer.  
                        Then provide your final answer within the tags, <answer>A/B</answer>.
                        """
                with assistant():
                    lm += gen("output")

                output = lm["output"]

                answer = re.findall(r"<answer>(.*?)</answer>", output)

                if answer[0] == "A. Yes" or answer[0] == "A":
                    suggested_relationship = 1
                    success = True
                elif answer[0] == "B. No" or answer[0] == "B":
                    suggested_relationship = 0
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return suggested_relationship

    def _build_description_program(self, variable, use_context=False, ask_reference=False):
        query = {}

        if use_context:
            query["system"] = f"""You are a helpful assistant for writing concise and peer-reviewed descriptions. Your goal is 
            to provide factual and succinct descriptions related to the given concept and context."""
            query["user"] = f"""Using this context about the particular variable, describe the concept of {variable}.
            In one sentence, provide a factual and succinct description of {variable}"""

            if ask_reference:
                query["user"] += f"""Then provide two research papers that support your description.
                Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final 
                answer within the tags, <description></description>, and each research paper within the tags <reference></reference>."""

            else:
                query["user"] += f"""
                    Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final 
                    answer within the tags, <description></description>."""

        else:
            query["system"] = f"""You are a helpful assistant for writing concise and peer-reviewed descriptions. Your goal 
            is to provide factual and succinct description of the given concept."""
            query["user"] = f""" Describe the concept of {variable}.
                    In one sentence, provide a factual and succinct description of {variable}"""

            if ask_reference:
                query["user"] += f""""
                        Then provide two research papers that support your description.
                        Let's think step-by-step to make sure that we have a proper and clear description. Then provide 
                        your final answer within the tags, <description></description>, and each research paper within the 
                        tags <paper></paper>."""
            else:
                query["user"] += f"""
                        Let's think step-by-step to make sure that we have a proper and clear description. Then provide 
                        your final answer within the tags, <description></description>."""
        return query

    def suggest_relationship(
            self,
            variable_a,
            variable_b,
            description_a=None,
            description_b=None,
            domain=None,
            strategy=None,
            ask_reference=False
    ):
        use_description = (
            True if description_a is not None and description_b is not None else False
        )

        program = self._build_relationship_program(
            variable_a,
            description_a,
            variable_b,
            description_b,
            use_domain=domain,
            use_strategy=strategy,
            use_description=use_description,
            ask_reference=ask_reference
        )

        success: bool = False
        suggested_relationship: int

        while not success:
            try:
                lm = self.llm
                with system():
                    lm += program["system"]

                with user():
                    lm += program["user"]

                with assistant():
                    lm += gen("output")

                output = lm["output"]

                answer = re.findall(r"<answer>(.*?)</answer", output)

                if answer[0] == "Yes" or answer[0] == "yes":
                    suggested_relationship = 1
                    success = True
                elif answer[0] == "No" or answer[0] == "no":
                    suggested_relationship = 0
                    success = True
                else:
                    success = False

                if ask_reference is False:
                    return suggested_relationship

                else:
                    reference = re.findall(
                        r"<reference>(.*?)</reference", output)

                    return suggested_relationship, reference

            except KeyError:
                success = False
                continue

    def _build_relationship_program(
            self,
            variable_a,
            description_a,
            variable_b,
            description_b,
            use_domain=None,
            use_strategy=Strategy.ToT_Single,
            use_description=False,
            ask_reference=False
    ):
        query = {}

        if use_domain is not None:
            query["system"] = f"""You are a helpful assistant on causal reasoning and {use_domain}. Your goal is to answer 
            questions about cause and effect in a factual and concise way."""
        else:
            query["system"] = f"""You are a helpful assistant on causal reasoning. Your goal is to answer questions 
            about cause and effect in a factual and concise way."""

        if use_strategy is not None:
            if use_strategy == Strategy.ToT_Single:
                query["user"] = f"""There is a council of three different experts answering this question.
                    Each of the three expert will write down 1 step of their thinking, and share it with the council. 
                    Then all experts will go on to the next step, etc.
                    All experts in the council are arguing to arrive to a true and factual answer. Their goal is to arrive 
                    at consensus but will only do so if an argument is factual and logical.
                    If an expert disagrees with an argument, then they will respectfully explain why with facts and logic. 
                    If any expert realises their argument is wrong at any point, then they will adjust their argument to 
                    be factual and logical.
                    The question is """

                if use_description:
                    query["user"] = f"""can changing {variable_a}, where {description_a}, change {variable_b}, where 
    {description_b}? Answer Yes or No."""
                else:
                    query["user"] = f"""can changing {variable_a} change {variable_b}? Answer Yes or No."""

                if ask_reference:
                    query["user"] += f"""At each step, each expert include a reference to a research paper that supports 
                    their argument. They will provide a one sentence summary of the paper and how it supports their argument. 
                        Then they will answer whether a change in {variable_a} changes {variable_b}. Answer Yes or No.
                        When consensus is reached, thinking carefully and factually, explain the council's answer. Provide 
                        the answer within the tags, <answer>Yes/No</answer>, and the most influential reference within 
                        the tags <reference>Author, Title, Year of publication</reference>.
                        \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of 
                        publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""
                else:
                    query["user"] += """When consensus is reached, thinking carefully and factually, explain the council's answer. 
                    Provide the answer within the tags, <answer>Yes/No</answer>.
                        \n\n\n----------------\n\n\n<answer>Yes</answer>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""

            elif use_strategy == Strategy.CoT:
                if use_description:
                    query["user"] = f"""Can changing {variable_a}, where {description_a}, change {variable_b}, where 
                    {description_b}? """
                else:
                    query["user"] = f"""Can changing {variable_a} change {variable_b}?"""

                if ask_reference:
                    query["user"] += f"""What are three research papers that discuss each of these variables? What do they 
                    say about the relationship they may or may not have? You are to provide the paper title and a one 
                    sentence summary each paper's argument. Then use those arguments as reference to answer whether a change 
                    in {variable_a} changes {variable_b}. Answer Yes or No.
                        Within one sentence, think carefully and factually, explaining your answer. Provide your final 
                        answer within the tags, <answer>Yes/No</answer>, and your references within the tags 
                        <reference>Author, Title, Year of publication</reference>. 
                        \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, 
                        Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""
                else:
                    query["user"] += f"""Answer Yes or No. Within one sentence, you are to think step-by-step to make sure 
                    that you have the right answer. Then provide your final answer within the tags, <answer>Yes/No</answer.
                        \n\n\n----------------\n\n\n<answer>Yes</answer>\n\n\n----------------\n\n\n<answer>No</answer>"""

            elif use_strategy == Strategy.Straight:
                if use_description:
                    query[
                        "user"] = f"""Can changing {variable_a}, where {description_a}, change {variable_b}, where {description_b}? """
                else:
                    query["user"] = f"""Can changing {variable_a} change {variable_b}?"""

                if ask_reference:
                    query["user"] += f"""What are three research papers that discuss each of these variables? What do they 
                    say about the relationship they may or may not have? You are to provide the paper title and a one 
                    sentence summary each paper's argument. Then use those arguments as reference to answer whether a 
                    change in {variable_a} changes {variable_b}. Answer Yes or No.
                        Within one sentence, you are to think step-by-step to make sure that you have the right answer. 
                        Provide your final answer within the tags, <answer>Yes/No</answer>, and your references within 
                        the tags <reference>Author, Title, Year of publication</reference. 
                        \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer>"""

                else:
                    query["user"] += f"""Answer Yes or No. Within one sentence, you are to think step-by-step to make sure 
                    that you have the right answer. Then provide your final answer within the tags, <answer>Yes/No</answer>.
                        \n\n\n----------------\n\n\nExample of output structure: <answer>Yes</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>No</answer>"""

        return query
