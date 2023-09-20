from typing import List, Tuple, Dict
from suggesters.protocols import ModelerProtocol
import networkx as nx
import guidance
from enum import Enum
import re


class Strategy(Enum):
    Straight = 0
    CoT = 1
    ToT_Single = 2
    ToT_Multi = 2


class Relationship(Enum):
    One_sided = 1
    Two_sided = 2


class ModelSuggester(ModelerProtocol):
    def suggest_description(
        self,
        variables: List[str],
        llm: guidance.llms,
        context=None,
        ask_reference=False,
        temperature=0.3,
    ):
        generate_description = self._build_description_program()

        success: bool = False
        suggested_description: Dict[str, Dict[str, str]]

        for variable in variables:
            success = False
            while not success:
                try:
                    if context is None:
                        output = generate_description(
                            variable=variable, llm=llm, temperature=temperature
                        )
                    else:
                        output = generate_description(
                            context=context,
                            variable=variable,
                            llm=llm,
                            temperature=temperature,
                        )

                    description = re.findall(
                        r"<description>(.*?)</description>", output["output"]
                    )

                    suggested_description["description"] = description[0]

                    if ask_reference:
                        reference = re.findall(
                            r"<reference>(.*?)</reference>", output["output"]
                        )

                        suggested_description["reference"] = reference[0]
                        success = True

                    return suggested_description

                except KeyError:
                    success = False
                    continue

    def _build_description_program(self, use_context=False, ask_reference=False):
        query = ""

        if use_context:
            query = """
            {{#system~}} 
            You are a helpful assistant for writing concise and peer-reviewed descriptions. Your goal is to provide factual and succinct descriptions related to the given concept and context. 
            {{~/system}}
            
            {{#user~}}
            {{context}}
            Using this context about the particular variable, describe the concept of {{variable}}.
            In one sentence, provide a factual and succinct description of {{variable}}"""

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

    def suggest_confounders(
        self,
        variables,
        #llm: guidance.llms,
        treatment: str,
        outcome: str,
        domain=None,  # , explanation=False
    ):
        generate_confounders = self._build_confounder_program(use_domain=domain)

        success: bool = False
        suggested_confounders: Dict[str, str]

        for variable in variables:
            if variable != treatment and variable != outcome:
                success = False
                while not success:
                    try:
                        output = generate_confounders(
                            treatment=treatment,
                            outcome=outcome,
                            variable=variable
                            #llm=llm,
                        )

                        confounder = re.findall(
                            r"<answer>(.*?)</answer>", output["confounders"]
                        )
                        # explanation = re.findall(r'<explanation>(.*?)</explanation>', output['confounders'])
                        print(confounder)
                        if confounder[0] == "A" or confounder[0] == "A. Yes":
                            suggested_confounders.append(variable)

                        success = True

                    except KeyError:
                        success = False
                        continue

        return suggested_confounders

    def _build_confounder_program(
        self,
        use_domain="",
        # use_strategy=Strategy.Straight,
        # use_description=None,
        # ask_explanation=False
    ):
        assistant = """{{#assistant~}} {{gen 'output' temperature=temperature}} {{~/assistant}}"""

        # set systen
        system = ""
        if use_domain == "" or use_domain == None:
            system = """{{#system~}} You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in a factual and concise way. {{~/system}}"""
        else:
            system = (
                """{{#system~}} You are a helpful assistant on causal reasoning and """
            )
            +use_domain
            (
                +""". Your goal is to answer questions about cause and effect in """
                + use_domain
                + """ in a factual and concise way. {{~/system}}"""
            )

        # set user
        user = """{{#user~}}
        Is it true that changing {{variable}} can change {{treatment}} and {{outcome}}? 
        A. Yes
        B. No
        Answer Yes or No. Think step-by-step to make sure that you are concise and accurate in your answer.
        Then provide your final answer within the tags, <answer>Yes/No</answer>.
        \n\n\n----------------\n\n\nExample of output structure: <answer>Yes</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>No</answer>
        {{~/user}}"""

        # set assistant
        assistant = """{{#assistant~}}
        {{gen 'confounders' temperature=temperature}}
        {{~/assistant}}"""

        return guidance(assistant + user + assistant)

    def suggest_relationship(
        self,
        variable_set_a: List[str],
        variable_set_b: List[str],
        llm,
        description_a=None,
        description_b=None,
        domain=None,
        strategy: Strategy = None,
        comparison_type=Relationship.Two_sided,
        ask_reference=False,
        consider_feedback_loop=False,
        temperature=0.3,
    ):
        use_description = (
            True if description_a is not None and description_b is not None else False
        )

        program: guidance()

        if comparison_type == Relationship.One_sided:
            program = self._build_one_sided_relationship_program(
                use_domain=domain,
                use_strategy=strategy,
                use_description=use_description,
                ask_reference=ask_reference,
                temperature=temperature,
            )
        elif comparison_type == Relationship.Two_sided:
            program = self._build_two_sided_relationship_program(
                use_domain=domain,
                use_strategy=strategy,
                use_description=use_description,
                ask_reference=ask_reference,
                consider_feedback_loop=True,
            )

        success: bool = False
        suggested_relationship: Dict[Tuple[str, str], Dict[str, str]]

        for variable_a in variable_set_a:
            for variable_b in variable_set_b:
                if variable_a != variable_b:
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

                            answer = re.findall(
                                r"<answer>(.*?)</answer>", output["output"]
                            )
                            temp_dict: Dict[str, str] = {}
                            if comparison_type is Relationship.One_sided:
                                if answer[0] == "Yes" or answer[0] == "yes":
                                    temp_dict["relationship"] = "R"  # Right: a causes b
                                elif answer[0] == "No" or answer[0] == "no":
                                    temp_dict[
                                        "relationship"
                                    ] = "N"  # No: a does not cause b
                                else:
                                    success = False

                            elif comparison_type is Relationship.Two_sided:
                                if consider_feedback_loop:
                                    if answer[0] == "A":  # Right
                                        temp_dict["relationship"] = "R"
                                    elif answer[0] == "B":  # Left
                                        temp_dict["relationship"] = "L"
                                    elif answer[0] == "C":  # Both
                                        temp_dict["relationship"] = "B"
                                    elif answer[0] == "D":  # Neither/No
                                        temp_dict["relationship"] = "N"
                                    else:
                                        success = False
                                else:
                                    if answer[0] == "A":  # Right
                                        temp_dict["relationship"] == "R"
                                    elif answer[0] == "B":  # Left
                                        temp_dict["relationship"] = "L"
                                    elif answer[0] == "C":  # Neither
                                        temp_dict["relationship"] = "N"
                                    else:
                                        success = False

                            if ask_reference:
                                reference = re.findall(
                                    r"<reference>(.*?)</reference>", output["output"]
                                )
                                temp_dict["reference"] = reference[0]

                            suggested_relationship[(variable_a, variable_b)] = temp_dict

                            success = True

                        except KeyError:
                            success = False
                            continue

        return suggested_relationship

    def _build_one_sided_relationship_program(
        self,
        use_domain=None,
        use_strategy=Strategy.ToT_Single,
        use_description=False,
        ask_reference=False,
    ):
        system = ""
        user = ""
        assistant = """{{#assistant~}} {{gen 'output' temperature=temperature}} {{~/assistant}}"""

        if use_domain is None:
            system = """{{#system~}} You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in a factual and concise way. {{~/system}}"""
        else:
            system = (
                """{{#system~}} You are a helpful assistant on causal reasoning and """
            )
            +use_domain
            (
                +""". Your goal is to answer questions about cause and effect in """
                + use_domain
                + """ in a factual and concise way. {{~/system}}"""
            )

        if use_strategy is not None:
            """
            Using Single Prompt ToT Ttrategy
            https://github.com/dave1010/tree-of-thought-prompting
            """
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
                    Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>Yes/No</answer, and your references within the tags <reference>Author, Title, Year of publication</reference>.
                    \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""
                else:
                    user += """Answer Yes or No. Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>Yes/No</answer.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>Yes</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>No</answer> {{~/user}}"""

            elif use_strategy == Strategy.Straight:
                if use_description:
                    user += """{{#user~}} Can changing {{variable_a}}, where {{description_a}}, change {{variable_b}}, where {{description_b}}? """
                else:
                    user += """{{#user~}} Can changing {{variable_a}} change {{variable_b}}?"""

                if ask_reference:
                    user += """What are three research papers that discuss each of these variables? What do they say about the relationship they may or may not have? You are to provide the paper title and a one sentence summary each paper's argument. Then use those arguments as reference to answer whether a change in{{variable_a}} changes {{variable_b}}. Answer Yes or No.
                    Within one sentence, you are to think step-by-step to make sure that you have the right answer. Provide your final answer within the tags, <answer>Yes/No</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference. \n\n\n----------------\n\n\n<answer>Yes</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\n<answer>No</answer> {{~/user}}"""

                else:
                    user += """Answer Yes or No. Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>Yes/No</answer>.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>Yes</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>No</answer> {{~/user}}"""

        return guidance(system + user + assistant)

    def _build_two_sided_relationship_program(
        self,
        use_domain="",
        use_strategy=Strategy.Straight,
        use_description=False,
        ask_reference=False,
        consider_feedback_loop=False,
    ):
        assistant = """{{#assistant~}} {{gen 'output' temperature=temperature}} {{~/assistant}}"""

        # set systen
        system = ""
        if use_domain is None:
            system = """{{#system~}} You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in a factual and concise way. {{~/system}}"""
        else:
            system = (
                """{{#system~}} You are a helpful assistant on causal reasoning and """
            )
            +use_domain
            (
                +""". Your goal is to answer questions about cause and effect in """
                + use_domain
                + """ in a factual and concise way. {{~/system}}"""
            )

        # set user
        user = """"""
        if use_strategy is not None:
            if use_strategy == Strategy.ToT_Single:
                user += """{{#user~}} There is a council of three different experts answering this question.
                Each of the three expert will write down 1 step of their thinking, and share it with the council. 
                Then all experts will go on to the next step, etc.
                All experts in the council are arguing to arrive at a true and factual answer. Their goal is to reach concensus but will only do so if an argument is factual and logical.
                If an expert disagrees with an argument, then they will respectfully explain why with facts and logic. 
                If any expert realises their argument is wrong at any point, then they will adjust their argument to be factual and logical.
                The question is """

                if use_description:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}}, where {{description_a}}, and {{variable_b}}, where {{description_b}}, makes most factual sense? 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. Both A and B.
                    D. None."""
                else:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense? 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. Both A and B.
                    D. None."""

                if ask_reference:
                    user += """At the end of each step, each expert include a reference to a research paper that supports their argument. They will provide a one sentence summary of the paper and how it supports their argument. 
                    Which of the following cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense? Answer A, B, C, or D.
                    When concensus is reached, thinking carefully and factually, explain the council's answer. Provide the answer within the tags, <answer>A/B/C/D</answer>, and the most influential reference within the tags <reference>Author, Title, Year of publication</reference>. 
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer>\n<reference>Author, Title, Year of publication</reference> {{~/user}}"""
                else:
                    user += """Which of the following cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense? Answer A, B, C, or D. When concensus is reached, thinking carefully and factually, explain the council's answer. Provide the answer within the tags, <answer>A/B/C/D</answer>.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer> {{~/user}}"""

            elif use_strategy == Strategy.CoT:
                if use_description:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}}, where {{description_a}}, and {{variable_b}}, where {{description_b}}, make most factual sense? 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. Both A and B.
                    D. None."""
                else:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense? 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. Both A and B.
                    D. None."""

                if ask_reference:
                    user += """What are three research papers that discuss each of these variables? What do they say about the relationship they may or may not have? You are to provide the paper title and a one sentence summary each paper's argument. Then use those arguments as reference to answer whether a change in{{variable_a}} changes {{variable_b}}. Answer Yes or No.
                    Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>Yes/No</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference>.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer>\n<reference>Author, Title, Year of publication</reference> {{~/user}}"""
                else:
                    user += """Answer Yes or No. Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>Yes/No</answer.
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer> {{~/user}}"""

            elif use_strategy == Strategy.Straight:
                # add variable description
                if use_description:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}}, where {{description_a}}, and {{variable_b}}, where {{description_b}}, make most factual sense?"""

                # no variable description
                else:
                    user += """{{#user~}} Which of the following cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense?"""

                # consider "both" in answer option
                if consider_feedback_loop:
                    user += """ 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. Both A and B.
                    D. None."""

                # don't consider "both" in answer options
                else:
                    user += """ 
                    A. Changing {{variable_a}} changes {{variable_b}}?
                    B. Changing {{variable_b}} changes {{variable_a}}?
                    C. None."""

                # request research paper support
                if ask_reference:
                    user += """What are three research papers that discuss each of these variables? What do they say about the relationship they may or may not have? You are to provide the paper title and a one sentence summary each paper's argument. Then use those arguments as reference to identify which of the cause and effect situations between {{variable_a}} and {{variable_b}} make most factual sense."""

                    # fourth option in examples
                    if consider_feedback_loop:
                        user += """Answer A, B, C, or D.
                   Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>A/B/C/D</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference. 
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer>\n<reference>Author, Title, Year of publication</reference> {{~/user}}"""

                    # three options in examples
                    else:
                        user += """Answer A, B, or C.
                   Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>A/B/C/D</answer>, and your references within the tags <reference>Author, Title, Year of publication</reference. 
                    \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n<reference>Author, Title, Year of publication</reference>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n<reference>Author, Title, Year of publication</reference> {{~/user}}"""

                # Don't request research paper support
                else:
                    user += """Think carefully and factually to identify which of the cause and effect situations between {{variable_a}} and {{variable_b}} makes most factual sense."""

                    # consider four examples
                    if consider_feedback_loop:
                        user += """Answer A, B, C, D. Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>A/B/C/D</answer>.
                        \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>D</answer>{{~/user}}"""

                    # consider three examples
                    else:
                        user += """Answer A, B, C. Think step-by-step to make sure that you are concise and accurate in your answer. Then provide your final answer within the tags, <answer>A/B/C</answer>.
                        \n\n\n----------------\n\n\nExample of output structure: <answer>A</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>B</answer>\n\n\n----------------\n\n\nExample of output structure: <answer>C</answer>{{~/user}}"""

        return guidance(system + user + assistant)
