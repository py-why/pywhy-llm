from typing import List, Tuple, Dict
from ..protocols import ModelerProtocol
import networkx as nx
import guidance
from guidance import system, user, assistant, gen
from enum import Enum
import re
import itertools


class SimpleIdentificationSuggester:

    def __init__(self, llm):
        if (llm == 'gpt-4'):
            self.llm = guidance.models.OpenAI('gpt-4')

    def suggest_iv(self, factors, treatment, outcome):
        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            lm += f"""Which factors in {factors} might be valid instrumental variables for identifying the effect of {treatment} on {outcome}?

            List the factors that are possible instrumental variables in <iv> </iv> tags."""
        with assistant():
            lm += gen("iv")

        ivs = lm['iv']
        ivs_list = re.findall(r'<iv>(.*?)</iv>', ivs)

        return ivs_list

    def suggest_backdoor(self, factors, treatment, outcome):
        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            lm += f"""Which set or subset of factors in {factors} might satisfy the backdoor criteria for identifying the effect of {treatment} on {outcome}?

            List the factors satisfying the backdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.
            """
        with assistant():
            lm += gen("backdoors")

        backdoors = lm['backdoors']
        backdoors_list = re.findall(r'<backdoor>(.*?)</backdoor>', backdoors)

        return backdoors_list

    def suggest_frontdoor(self, factors, treatment, outcome):
        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            lm += f"""Which set or subset of factors in {factors} might satisfy the frontdoor criteria for identifying the effect of {treatment} on {outcome}?

            List the factors satisfying the frontdoor criteria enclosing the name of each factor in <frontdoor> </frontdoor> tags.
            """
        with assistant():
            lm += gen("frontdoors")

        frontdoors = lm['frontdoors']
        frontdoors_list = re.findall(r'<frontdoor>(.*?)</frontdoor>', frontdoors)

        return frontdoors_list

