from typing import List
import guidance
import re

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
        generate_description = self._build_description_program()

        success: bool = False
        suggested_description: str

        while not success:
            try:
                if context is None:
                    output = generate_description(variable=variable)
                else:
                    output = generate_description(
                        context=context, variable=variable)

                description = re.findall(
                    r"<description>(.*?)</description>", output["output"]
                )

                if ask_reference:
                    reference = re.findall(
                        r"<reference>(.*?)</reference>", output["output"]
                    )
                    success = True

                    return [description[0], description[0]]

                else:
                    success = True
                    return suggested_description

            except KeyError:
                success = False
                continue

    def suggest_onesided_relationship(
            self,
            variable_a,
            description_a,
            variable_b,
            description_b,
            llm: guidance.llms,
            temperature: None = 0.3,
    ):
        suggest_relationship = self._onesided_relationship_program()

        success: bool = False
        suggested_relationship: int

        while not success:
            try:
                output = suggest_relationship(
                    variable_a=variable_a,
                    description_a=description_a,
                    variable_b=variable_b,
                    description_b=description_b,
                    llm=llm,
                    temperature=temperature,
                )

                output = re.findall(r"<answer>(.*?)</answer>", output["output"])

                if output[0] == "A. Yes" or output[0] == "A":
                    suggested_relationship = 1
                    success = True
                elif output[0] == "B. No" or output[0] == "B":
                    suggested_relationship = 0
                    success = True
                else:
                    success = False

            except KeyError:
                success = False
                continue

        return suggested_relationship

    def _build_description_program(self, use_context=False, ask_reference=False):
        query = {}

        if use_context:
            query["system"] = f"""You are a helpful assistant for writing concise and peer-reviewed descriptions. Your goal is 
            to provide factual and succinct descriptions related to the given concept and context."""
            query["user"] = f"""Using this context about the particular variable, describe the concept of {variable}.
            In one sentence, provide a factual and succinct description of {variable}"""

            if ask_reference:
                query += """
                Then provide two research papers that support your description.
                Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final answer within the tags, <description></description>, and each research paper within the tags <reference></reference>.
                {{~/user}}

                {{#assistant~}}
                {{gen 'output' temperature=temperature}}
                {{~/assistant}}"""
            else:
                query += """
                Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final answer within the tags, <description></description> .
                {{~/user}}

                {{#assistant~}}
                {{gen 'output' temperature=temperature}}
                {{~/assistant}}"""

        else:
            query = """
            {{#system~}} 
            You are a helpful assistant for writing concise and peer-reviewed descriptions. Your goal is to provide factual and succinct description of the given concept. 
            {{~/system}}
            
            {{#user~}}
            Describe the concept of {{variable}}.
            In one sentence, provide a factual and succinct description of {{variable}}.
            """

            if ask_reference:
                query += """
                Then provide two research papers that support your description.
                Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final answer within the tags, <description></description>, and each research paper within the tags <paper></paper>.
                {{~/user}}

                {{#assistant~}}
                {{gen 'output' temperature=temperature}}
                {{~/assistant}}"""
            else:
                query += """
                Let's think step-by-step to make sure that we have a proper and clear description. Then provide your final answer within the tags, <description></description> .
                {{~/user}}

                {{#assistant~}}
                {{gen 'output' temperature=temperature}}
                {{~/assistant}}"""

        return guidance(query)

    def _onesided_relationship_program(self):
        return guidance(
            """
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in a factual and concise way.
        {{~/system}}

        {{#user~}}
        Can changing {{variable_a}}, {{description_a}}, change {{variable_b}}, {{description_b}}?
        A. Yes
        B. No
        Within one sentence, let's think step-by-step to make sure that we have the right answer.  Then provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'output' temperature=temperature}}
        {{~/assistant}} 
        """
        )

    def suggest_relationship(
            self,
            variable_a,
            variable_b,
            llm,
            description_a=None,
            description_b=None,
            domain=None,
            strategy=None,
            ask_reference=False,
            temperature=0.3,
    ):
        use_description = (
            True if description_a is not None and description_b is not None else False
        )

        program = self._build_relationship_program(
            use_domain=domain,
            use_strategy=strategy,
            use_description=use_description,
            ask_reference=ask_reference,
        )

        success: bool = False
        suggested_relationship: int

        while not success:
            try:
                if use_description:
                    output = program(
                        variable_a=variable_a,
                        description_a=description_a,
                        variable_b=variable_b,
                        description_b=description_b,
                        llm=llm,
                        temperature=temperature,
                    )
                else:
                    output = program(
                        variable_a=variable_a,
                        variable_b=variable_b,
                        llm=llm,
                        temperature=temperature,
                    )

                answer = re.findall(r"<answer>(.*?)</answer>", output["output"])

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
                        r"<reference>(.*?)</reference>", output["output"]
                    )

                    return (suggested_relationship, reference)

            except KeyError:
                success = False
                continue

    def _build_relationship_program(
            self,
            use_domain=None,
            use_strategy=Strategy.ToT_Single,
            use_description=False,
            ask_reference=False,
    ):
        system = ""
        user = ""
        assistant = """{{#assistant~}} {{gen 'output' temperature=temperature}} {{~/assistant}}"""

        if use_domain is not None:
            system = (
                    """{{#system~}} You are a helpful assistant on causal reasoning and """
                    + use_domain
                    + """. Your goal is to answer questions about cause and effect in a factual and concise way. {{~/system}}"""
            )

        else:
            system = """{{#system~}} You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in a factual and concise way. {{~/system}}"""

        if use_strategy is not None:
            if use_strategy == Strategy.ToT_Single:
                user += """{{#user~}} There is a council of three different experts answering this question.
                Each of the three expert will write down 1 step of their thinking, and share it with the council. 
                Then all experts will go on to the next step, etc.
                All experts in the council are arguing to arrive to a true and factual answer. Their goal is to arrive at concensus but will only do so if an argument is factual and logical.
                If an expert disagrees with an argument, then they will respectfully explain why with facts and logic. 
                If any expert realises their argument is wrong at any point, then they will adjust their argument to be factual and logical.
                The question is """

                if use_description:
                    user += """can changing {{variable_a}}, where {{description_a}}, change {{variable_b}}, where {{description_b}}? Answer Yes or No."""
                else:
                    user += """can changing {{variable_a}} change {{variable_b}}? Answer Yes or No."""

                if ask_reference:
                    user += """At each step, each expert include a reference to a research paper that supports their argument. They will provide a one sentence summary of the paper and how it supports their argument. 
                    Then they will answer whether a change in{{variable_a}} changes {{variable_b}}. Answer Yes or No.
                    When concensus is reached, thinking carefully and factually, explain the council's answer. Provide the answer within the tags, <answer>Yes/No</answer>, and the most influential reference within the tags <reference>Author, Title, Year of publication</reference>.
                    \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""
                else:
                    user += """When concensus is reached, thinking carefully and factually, explain the council's answer. Provide the answer within the tags, <answer>Yes/No</answer>.
                    \n\n\n----------------\n\n\n<answer>Yes</answer>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""

            elif use_strategy == Strategy.CoT:
                if use_description:
                    user += """{{#user~}} Can changing {{variable_a}}, where {{description_a}}, change {{variable_b}}, where {{description_b}}? """
                else:
                    user += """{{#user~}} Can changing {{variable_a}} change {{variable_b}}?"""

                if ask_reference:
                    user += """What are three research papers that discuss each of these variables? What do they say about the relationship they may or may not have? You are to provide the paper title and a one sentence summary each paper's argument. Then use those arguments as reference to answer whether a change in{{variable_a}} changes {{variable_b}}. Answer Yes or No.
                    Within one sentence, think carefully and factually, explaining your answer. Provide your final answer within the tags, <answer>Yes/No</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference>.
                    \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""
                else:
                    user += """Answer Yes or No. Within one sentence, you are to think step-by-step to make sure that you have the right answer. Then provide your final answer within the tags, <answer>Yes/No</answer.
                    \n\n\n----------------\n\n\n<answer>Yes</answer>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""

            elif use_strategy == Strategy.Straight:
                if use_description:
                    user += """{{#user~}} Can changing {{variable_a}}, where {{description_a}}, change {{variable_b}}, where {{description_b}}? """
                else:
                    user += """{{#user~}} Can changing {{variable_a}} change {{variable_b}}?"""

                if ask_reference:
                    user += """What are three research papers that discuss each of these variables? What do they say about the relationship they may or may not have? You are to provide the paper title and a one sentence summary each paper's argument. Then use those arguments as reference to answer whether a change in{{variable_a}} changes {{variable_b}}. Answer Yes or No.
                    Within one sentence, you are to think step-by-step to make sure that you have the right answer. Provide your final answer within the tags, <answer>Yes/No</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference. \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""

                else:
                    user += """Answer Yes or No. Within one sentence, you are to think step-by-step to make sure that you have the right answer. Then provide your final answer within the tags, <answer>Yes/No</answer>.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>Yes</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>No</answer> {{~/user}}"""

        return guidance(system + user + assistant)
