from typing import List, Tuple, Dict
from ..protocols import ModelerProtocol
import networkx as nx
import guidance
from enum import Enum
import re
import itertools
import os
from guidance import system, user, assistant, gen


class SimpleModelSuggester:
    """
    A class that provides methods for suggesting causal relationships and confounding factors between variables.

    This class uses the guidance library to interact with LLMs, and assumes that guidance.llm has already been initialized to the user's preferred LLM 

    Methods:
    - suggest_pairwise_relationship(variable1: str, variable2: str) -> List[str]: 
        Suggests the causal relationship between two variables and returns a list containing the cause, effect, and a description of the relationship.
    - suggest_relationships(variables: List[str]) -> Dict[Tuple[str, str], str]: 
        Suggests the causal relationships between all pairs of variables in a list and returns a dictionary containing the cause-effect pairs and their descriptions.
    - suggest_confounders(variables: List[str], treatment: str, outcome: str) -> List[str]: 
        Suggests the confounding factors that might influence the relationship between a treatment and an outcome, given a list of variables that have already been considered.
    """

    def __init__(self, llm):
        if (llm == 'gpt-4'):
            self.llm = guidance.models.OpenAI('gpt-4')

    # new ver
    def suggest_pairwise_relationship(self, variable1: str, variable2: str):
        """
            Suggests a cause-and-effect relationship between two variables.

            Args:
                variable1 (str): The name of the first variable.
                variable2 (str): The name of the second variable.

            Returns:
                list: A list containing the suggested cause variable, the suggested effect variable, and a description of the reasoning behind the suggestion.  If there is no relationship between the two variables, the first two elements will be None.
            """

        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            lm += f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.
            A. {variable1} causes {variable2} B. {variable2} causes {variable1} C. neither {variable1} nor {variable2} cause each other."""

        with assistant():
            lm += gen("description")

        # with user():
        #     lm += "Now what is your final answer: A, B, or C? Just give me a single letter."
        #
        # with assistant():
        #     lm += gen("answer")

        description = lm['description']
        print(description)
        answer = re.findall(r'<answer>(.*?)</answer>', description)
        answer = [ans.strip() for ans in answer]
        answer_str = "".join(answer)

        if (answer_str == "A"):
            return [variable1, variable2, description]
        elif (answer_str == "B"):
            return [variable2, variable1, description]
        elif (answer_str == "C"):
            return [None, None, description]  # maybe we want to save the description in this case too
        else:
            assert False, "Invalid answer from LLM: " + answer_str

    def suggest_relationships(self, variables: List[str]):
        """
        Given a list of variables, suggests relationships between them by querying for pairwise relationships.

        Args:
            variables (List[str]): A list of variable names.

        Returns:
            dict: A dictionary of edges found between variables, where the keys are tuples representing the causal relationship between two variables,
            and the values are the strength of the relationship.
        """
        relationships = {}
        total = (len(variables) * (len(variables) - 1) / 2)
        i = 0
        for (var1, var2) in itertools.combinations(variables, 2):
            i += 1
            print(f"{i}/{total}: Querying for relationship between {var1} and {var2}")
            y = list(self.suggest_pairwise_relationship(var1, var2))
            if (y[0] == None):
                print(f"\tNo relationship found between {var1} and {var2}")
                continue
            print(f"\t{y[0]} causes {y[1]}")
            relationships[(y[0], y[1])] = y[2]

        return relationships

    # new ver
    def suggest_confounders(self, variables: List[str], treatment: str, outcome: str) -> List[str]:

        """
            Suggests potential confounding factors that might influence the relationship between the treatment and outcome variables.

            Args:
                variables (List[str]): A list of variables that have already been considered.
                treatment (str): The name of the treatment variable.
                outcome (str): The name of the outcome variable.

            Returns:
                List[str]: A list of potential confounding factors.
            """

        lm = self.llm

        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            lm += f"""What latent confounding factors might influence the relationship between {treatment} and {outcome}?

            We have already considered the following factors {variables}.  Please do not repeat them.

            List the confounding factors between {treatment} and {outcome} enclosing the name of each factor in <conf> </conf> tags.
            """
        with assistant():
            lm += gen("latents")

        latents = lm['latents']
        latents_list = re.findall(r'<conf>(.*?)</conf>', latents)

        return latents_list
