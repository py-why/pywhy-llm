from typing import List, Tuple, Dict
from suggesters.protocols import ModelerProtocol
import networkx as nx
import guidance
from enum import Enum
import re
import itertools

class SimpleModelSuggester:

    def suggest_pairwise_relationship(self, variable1: str, variable2: str):
        prompt = """
        {{#system~}} 
        You are a helpful assistant for causal reasoning.
        {{~/system}}
        
        {{#user~}}
        Which cause-and-effect-relationship is more likely?
        A. {{variable1}} causes {{variable2}}
        B. {{variable2}} causes {{variable1}}
        C. neither {{variable1}} nor {{variable2}} cause each other
        First let's very succinctly think about each option.
        Then I'll ask you to provide your final answer A, B, or C.
        {{~/user}}

        {{#assistant}}
        {{~gen 'description'}}
        {{~/assistant}}

        {{#user~}}
        Now what is your final answer: A, B, or C?
        {{~/user}}

        {{#assistant}}
        {{~#select 'answer'}}A{{~or}}B{{~or}}C{{~/select}}
        {{~/assistant}}
        """

        program = guidance(prompt)
        executed_program = program(variable1=variable1, variable2=variable2)
        if( executed_program._exception is not None):
            raise executed_program._exception

        description = executed_program['description']
        answer = executed_program['answer']

        if( answer == "A"):
            return [variable1, variable2, description]
        elif( answer == "B"):
            return [variable2, variable1, description]
        elif(answer == "C"):
            return [None, None, description] # maybe we want to save the description in this case too
        else:
            assert False, "Invalid answer from LLM: " + answer

    def suggest_relationships(self, variables: List[str]):
        relationships = {}
        total = (len(variables) * (len(variables)-1)/2)
        i=0
        for(var1, var2) in itertools.combinations(variables, 2):
            i+=1
            print(f"{i}/{total}: Querying for relationship between {var1} and {var2}")
            y = self.suggest_pairwise_relationship(var1, var2)
            if( y[0] == None ):
                print(f"\tNo relationship found between {var1} and {var2}")
                continue
            print(f"\t{y[0]} causes {y[1]}")
            relationships[(y[0], y[1])] = y[2]

        return relationships
    
    def suggest_confounders(self, variables: List[str], treatment: str, outcome: str):
        prompt = """
        {{#system~}} 
        You are a helpful assistant for causal reasoning.
        {{~/system}}
        
        {{#user~}}
        What latent confounding factors might influence the relationship between {{treatment}} and {{outcome}}?

        We have already considered the following factors {{variables}}.  Please do not repeat them.

        List the confounding factors between {{treatment}} and {{outcome}} enclosing the name of each factor in <conf> </conf> tags.
        {{~/user}}

        {{#assistant}}
        {{~gen 'latents'}}
        {{~/assistant}}
        """ 
        program = guidance(prompt)

        executed_program = program(variables=str(variables), treatment=treatment, outcome=outcome)

        if( executed_program._exception is not None):
            raise executed_program._exception
        
        latents = executed_program['latents']
        latents_list = re.findall(r'<conf>(.*?)</conf>', latents)

        return latents_list


