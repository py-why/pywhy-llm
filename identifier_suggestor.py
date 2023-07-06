from typing import List, Dict, Set, Tuple, Protocol
from protocols import IdentifierProtocol
import dodiscover as dd
import guidance
import re

class IdentifierSuggestor(IdentifierProtocol):

    def suggest_backdoor(self, variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], treatment: str, outcome: str, llm: guidance.llm) -> Set[str]:
        
        program = self.backdoor_program()

        output = program(treatment=treatment, outcome=outcome, edges=variable_relationships.keys(), llm=llm)

        backdoor_set = re.findall(r'<variable>(.*?)</variable>', output['backdoor_set'])

        return set(backdoor_set)
    
    def suggest_frontdoor(self, variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], treatment: str, outcome: str, llm: guidance.llm) -> Set[str]:
         
        program = self.frontdoor_program()

        output = program(treatment=treatment, outcome=outcome, edges=variable_relationships.keys(), llm=llm)

        frontdoor_set = re.findall(r'<variable>(.*?)</variable>', output['frontdoor_set'])

        return set(frontdoor_set)
    
    def suggest_iv(self, variable_relationships: Dict[Tuple[str, str], str], confounders: Set[Tuple[str, str]], treatment: str, outcome: str, llm: guidance.llm) -> Set[str]:
         
        program = self.iv_program()

        output = program(treatment=treatment, outcome=outcome, edges=variable_relationships.keys(), llm=llm)

        iv_set = re.findall(r'<variable>(.*?)</variable>', output['iv_set'])

        return set(iv_set)
    
    def backdoor_program(self) -> guidance._program.Program:
            
            return guidance(    
            '''
            {{#system~}}
            You are a causal inference expert and your task is to identify the backdoor set for a specific causal relationship in a directed acyclic graph (DAG). The DAG is represented as a list of edges, where each edge represents a causal relationship between two variables. Your task is to provide me with the backdoor set for a given causal relationship.

            The backdoor set for a causal relationship in a DAG is defined as a set of variables that satisfies two conditions:

            1. It blocks all directed paths from the treatment variable to the outcome variable.
            2. It does not contain any descendants of the treatment variable.

            To assist you, I will provide you with the causal relationship of interest and the DAG represented by a list of edges between variables.

            Your task if to provide me with the set of variables that satisfy the backdoor criterion for the given causal relationship, should any exist. 
            
            -----------------------------------------------------------------------------
            Input: 

                List of edges:
                    (variable_1, variable_2)
                    ...
                    (variable_n, variable_m)

                Causal relationship of interest:
                    treatment: variable_a
                    outcome: variable_k

            Output:

                If backdoor set exists for the given causal relationship:
                    <variable>variable_1</variable>
                    ...
                    <variable>variable_n</variable>

                If no backdoor set exists for the given causal relationship:
                    <null>   
            {{~/system}}

            {{#user~}}
            List of edges:
            {{edges}}

            Causal relationship of interest:
            treatment: {{treatment}}
            outcome: {{outcome}}
            {{~/user}}
                        
            {{#assistant~}}
            {{gen 'backdoor_set' temperature=0.7}}
            {{~/assistant}}
            ''') 

    def frontdoor_program(self) -> guidance._program.Program:
        
        return guidance(    
        '''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. Your task is to identify the frontdoor set for a specific causal relationship in a directed acyclic graph (DAG). The DAG is represented as a list of edges, where each edge represents a causal relationship between two variables. Your task is to provide me with the frontdoor set for a given causal relationship.

        The frontdoor set for a causal relationship in a DAG is defined as a set of variables that satisfies the following conditions:

        1. It satisfies the frontdoor criterion, meaning it blocks all frontdoor paths from the causal variable to the outcome variable.
        2. It does not contain any colliders between the causal variable and the outcome variable.
        3. It is not affected by the outcome variable.
        
        -----------------------------------------------------------------------------
        Input: 

            List of edges:
                (variable_1, variable_2)
                ...
                (variable_n, variable_m)

            Causal relationship of interest:
                treatment: variable_a
                outcome: variable_k

        Output:

            If frontdoor set exists for the given causal relationship:
                <variable>variable_1</variable>
                ...
                <variable>variable_n</variable>

            If no frontdoor set exists for the given causal relationship:
                <null>   
        {{~/system}}

        {{#user~}}
        List of edges:
        {{edges}}

        Causal relationship of interest:
        treatment: {{treatment}}
        outcome: {{outcome}}
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'frontdoor_set' temperature=0.7}}
        {{~/assistant}}
        ''') 

    def iv_program(self) -> guidance._program.Program:
        
        return guidance(    
        '''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. Your task is to identify the instrumental variables causing the treatment in a specific causal relationship in a directed acyclic graph (DAG), where the DAG is represented as a list of edges, where each edge denotes a causal relationship between two variables. 

        In the field of causal inference, instrumental variables play a crucial role in estimating causal effects when facing unobserved confounding.
        Instrumental variables are defined as variables that satisfy the following conditions:

        1. They cause the treatment variable of interest.
        2. They are not caused by the outcome variable.
        3. They only affect the outcome variable indirectly through their effect on the treatment variable, they have no direct effect on the outcome variable.
        4. They are not caused by any confounders of the causal relationship of interest.
        
        -----------------------------------------------------------------------------
        Input: 

            List of edges:
                (variable_1, variable_2)
                ...
                (variable_n, variable_m)

            Causal relationship of interest:
                treatment: variable_a
                outcome: variable_k

        Output:

            If instrumental variable exists for the given causal relationship:
                <variable>variable_1</variable>

            If instrumental variables exist for the given causal relationship:
                <variable>variable_1</variable>
                ...
                <variable>variable_n</variable>

            If no instrumental variable exists for the given causal relationship:
                <null>   
        {{~/system}}

        {{#user~}}
        List of edges:
        {{edges}}

        Causal relationship of interest:
        treatment: {{treatment}}
        outcome: {{outcome}}
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'iv_set' temperature=0.7}}
        {{~/assistant}}
        ''') 