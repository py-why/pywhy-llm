from typing import List, Dict, Tuple
from protocols import ModelerProtocol
import guidance
import re

SYSTEM = "You are a helpful assistant on causal reasoning. Your goal is to answer questionsfactually and concisely about cause and effect in {{field}}"

class ModelSuggestor(ModelerProtocol):

    def suggest_descriptions(self, variable_names: List[str], llm: guidance.llms) -> Dict[str, str]:
    
        variables_and_descriptions : Dict[str, str] = {}

        generate_description = self.observed_variable_description_program()  
        success : bool = False

        
        for variable_name in variable_names:
            success = False
            while not success: 
                try:
                    output = generate_description(variable=variable_name, llm=llm)

                    variables_and_descriptions[variable_name] = output['description']

                    success = True  
                except KeyError:
                    success = False
                    continue
        return variables_and_descriptions


    def observed_variable_description_program(self) -> guidance._program.Program:
        
        generate_description = guidance(    
        '''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about arctic sea ice and atmosphere sciences in a factual and concise way. You are helping me better understand a dataset by providing me with a description of the variables. Let's think step-by-step to make sure that we have the right description. 
        {{~/system}}

        {{#user~}}
        {{variable}}
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'description' temperature=0.3}}
        {{~/assistant}}
        ''') 

        return generate_description


    def suggest_confounders(self, variables_and_descriptions: Dict[str, str], llm: guidance.llms, treatment: str, outcome: str) -> List[str]:
        generate_confounders = self.confounder_program()

        success : bool = False
        suggested_confounders : list = []

        for variable in variables_and_descriptions.keys():
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
    

    def confounder_program(self) -> guidance._program.Program:
        generate_confounders = guidance('''
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

        return generate_confounders
    

    def suggest_variable_relationships(
        self, 
        variables_and_descriptions: Dict[str, str], 
        llm: guidance.llms) -> List[Tuple[str, str]]:

        generate_relationships = self.pairwise_relationship_program()

        relationships: List[Tuple[str, str]] = []
        success: bool = False

        for variable_a in variables_and_descriptions.keys():

            for variable_b in variables_and_descriptions.keys():

                if variable_a != variable_b and (variable_a, variable_b) not in relationships and (variable_b, variable_a) not in relationships:
                    success = False
                    while not success: 
                        try:
                            output = generate_relationships(variables_and_descriptions=variables_and_descriptions, variable_a=variable_a, variable_b=variable_b, llm=llm)

                            relationship = re.findall(r'<answer>(.*?)</answer>', output['relationship'])
                            print(relationship) 
                            if relationship[0] == 'A' or relationship[0] == 'A. Yes':
                                relationships.append((variable_a, variable_b))             
                                
                            success = True
                        
                        except KeyError:
                            success = False
                            continue

        return relationships


    def pairwise_relationship_program(self) -> guidance._program.Program:

        generate_relationships = guidance('''
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
        return generate_relationships 