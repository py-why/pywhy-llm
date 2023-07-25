

from typing import List, Tuple
from protocols import ModelerProtocol
import networkx as nx
import guidance
import re

class ModelSuggester(ModelerProtocol):       

    def suggest_confounders(self, variables: List[str], llm: guidance.llms, treatment: str, outcome: str):

        generate_confounders = self._confounder_program()


        success : bool = False
        suggested_confounders : list = []

        for variable in variables:
            if variable != treatment and variable != outcome:
                success = False
                while not success: 
                    try:
                        output = generate_confounders(
                            treatment=treatment, 
                            outcome=outcome, 
                            variable=variable,
                            llm=llm)

                        confounder = re.findall(r'<answer>(.*?)</answer>', output['confounders'])
                        # explanation = re.findall(r'<explanation>(.*?)</explanation>', output['confounders'])
                        print(confounder)
                        if confounder[0] == 'A' or confounder[0] == 'A. Yes':
                            suggested_confounders.append(variable)

                        success = True
                    
                    except KeyError:
                        success = False
                        continue

        return suggested_confounders
    
    def suggest_relationships(self, variables: List[str], llm: guidance.llms):

        generate_relationships = self._pairwise_relationship_program()


        relationships: List[Tuple[str, str]] = []
        success: bool = False

        for variable_a in variables:

            for variable_b in variables:

                if variable_a != variable_b and (variable_a, variable_b) not in relationships and (variable_b, variable_a) not in relationships:
                    success = False
                    while not success: 
                        try:
                            output = generate_relationships(variable_a=variable_a, variable_b=variable_b, llm=llm)

                            relationship = re.findall(r'<answer>(.*?)</answer>', output['relationship'])
                            print(relationship) 
                            if relationship[0] == 'A' or relationship[0] == 'A. Yes':
                                relationships.append((variable_a, variable_b))             
                                
                            success = True
                        
                        except KeyError:
                            success = False
                            continue
        
        g = nx.DiGraph()
        g.add_nodes_from(variables)
        g.add_edges_from(relationships)
                                
        return g

    def _confounder_program(self):

        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in arctic sea ice and atmosphere sciences in a factual and concise way.
        {{~/system}}

        {{#user~}}
        Is it true that changing {{variable}} can change {{treatment}} and {{outcome}}? 
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'confounders' temperature=0.3}}
        {{~/assistant}} 
        ''')
   
    def _pairwise_relationship_program(self):
                
        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in arctic sea ice and atmosphere sciences in a factual and concise way.
        {{~/system}}

        {{#user~}}
        Can changing {{variable_a}} change {{variable_b}}?
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Then provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationship' temperature=0.3}}
        {{~/assistant}} 

        ''')
    













