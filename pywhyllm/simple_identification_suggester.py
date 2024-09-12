from typing import List, Tuple, Dict
import networkx as nx
import guidance
from enum import Enum
import re
import itertools

class SimpleIdentificationSuggester:

    def suggest_backdoor(
        self,
        factors_list,
        treatment,
        outcome
    ):
        prompt = """
        {{#system~}} 
        You are a helpful assistant for causal reasoning.
        {{~/system}}
        
        {{#user~}}
        Which set or subset of factors in {{factors}} might satisfy the backdoor criteria for identifying the effect of {{treatment}} on {{outcome}}?

        List the factors satisfying the backdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.
        {{~/user}}

        {{#assistant}}
        {{~gen 'backdoors'}}
        {{~/assistant}}
        """ 
        program = guidance(prompt)

        executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)

        if( executed_program._exception is not None):
            raise executed_program._exception
        
        backdoors = executed_program['backdoors']
        backdoors_list = re.findall(r'<backdoor>(.*?)</backdoor>', backdoors)

        return backdoors_list


    def suggest_frontdoor(
        self,
        factors_list,
        treatment,
        outcome
    ):
        prompt = """
        {{#system~}} 
        You are a helpful assistant for causal reasoning.
        {{~/system}}
        
        {{#user~}}
        Which set or subset of factors in {{factors}} might satisfy the frontdoor criteria for identifying the effect of {{treatment}} on {{outcome}}?

        List the factors satisfying the frontdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.
        {{~/user}}

        {{#assistant}}
        {{~gen 'frontdoors'}}
        {{~/assistant}}
        """ 
        program = guidance(prompt)

        executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)

        if( executed_program._exception is not None):
            raise executed_program._exception
        
        frontdoors = executed_program['frontdoors']
        frontdoors_list = re.findall(r'<frontdoor>(.*?)</frontdoor>', frontdoors)

        return frontdoors_list


    def suggest_iv(
        self,
        factors_list,
        treatment,
        outcome
    ):
        prompt = """
        {{#system~}} 
        You are a helpful assistant for causal reasoning.
        {{~/system}}
        
        {{#user~}}
        Which factors in {{factors}} might be valid instrumental variables for identifying the effect of {{treatment}} on {{outcome}}?

        List the factors that are possible instrumental variables in <iv> </iv> tags.
        {{~/user}}

        {{#assistant}}
        {{~gen 'ivs'}}
        {{~/assistant}}
        """ 
        program = guidance(prompt)

        executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)

        if( executed_program._exception is not None):
            raise executed_program._exception
        
        ivs = executed_program['ivs']
        ivs_list = re.findall(r'<iv>(.*?)</iv>', ivs)

        return ivs_list
