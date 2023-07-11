from typing import List, Dict, Set, Tuple, Protocol
from protocols import IdentifierProtocol
from dowhy.causal_identifier.auto_identifier import construct_backdoor_estimand, construct_frontdoor_estimand, construct_iv_estimand, EstimandType
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
import guidance
import re

class IdentificationSuggester(IdentifierProtocol):

    def suggest_estimand(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm, backdoor: Set[str] = None, frontdoor: Set[str] = None, iv: Set[str] = None):

        estimands_dict = {}
        
        if backdoor == None:
            backdoor = self.suggest_backdoor(variables=variables, treatment=treatment, outcome=outcome, llm=llm)
        if len(backdoor) > 0:
            estimands_dict["backdoor"] = construct_backdoor_estimand(treatment, outcome, backdoor)
        else:
            estimands_dict["backdoor"] = None   

        if frontdoor == None:
            frontdoor = self.suggest_frontdoor(variables=variables, treatment=treatment, outcome=outcome, llm=llm)   
        if len(frontdoor) > 0:
            estimands_dict["frontdoor"] = construct_frontdoor_estimand(treatment, outcome, frontdoor)  
        else:
            estimands_dict["frontdoor"] = None    

        if iv == None:
            iv = self.suggest_iv(variables=variables, treatment=treatment, outcome=outcome, llm=llm)
        if len(iv) > 0:
            estimands_dict["iv"] = construct_iv_estimand(treatment, outcome, iv)
        else:
            estimands_dict["iv"] = None

        estimand = IdentifiedEstimand(
            None,
            treatment_variable=treatment,
            outcome_variable=outcome,
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            estimands=estimands_dict,
            backdoor_variables=backdoor,
            instrumental_variables=iv,
            frontdoor_variables=frontdoor,
        )
        return estimand

    def suggest_backdoor(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
        
        program = self.backdoor_program()
        
        backdoor_set : Set[str] = []

        for variable in variables: 
            if variable != treatment and variable != outcome:
                success = False
                while not success: 
                    try:
                        output = program(treatment=treatment, outcome=outcome, variable=variable, llm=llm)

                        z = re.findall(r'<answer>(.*?)</answer>', output['backdoor_set'], re.DOTALL)

                        if z[0] == 'A' or z[0] == 'A. Yes':
                            backdoor_set.append(variable)
                            
                        success = True

                    except KeyError:
                        success = False
                        continue 

                    except IndexError:
                        success = False
                        continue 

        return backdoor_set
    
    def suggest_frontdoor(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
         
        program = self.frontdoor_program()

        frontdoor_set : Set[str] = []

        for variable in variables: 
            if variable != treatment and variable != outcome:
                success = False
                while not success: 
                    try:
                        output = program(treatment=treatment, outcome=outcome, variable=variable, llm=llm)

                        z = re.findall(r'<answer>(.*?)</answer>', output['frontdoor_set'], re.DOTALL)

                        if len(z) > 0: 
                            if z[0] == 'A' or z[0] == 'A. Yes':
                                frontdoor_set.append(variable)
                            
                            success = True

                        else:
                            success = False

                    except KeyError:
                        success = False
                        continue 

        return frontdoor_set
    
    def suggest_iv(self, variables: List[str], treatment: str, outcome: str, llm: guidance.llm):
         
        program = self.iv_program()
        
        iv_set : set[str] = []

        for variable in variables: 
            if variable != treatment and variable != outcome:
                success = False
                while not success: 
                    try:
                        output = program(treatment=treatment, outcome=outcome, variable=variable, llm=llm)

                        z = re.findall(r'<answer>(.*?)</answer>', output['iv_set'], re.DOTALL)

                        if z: 
                            if z[0] == 'A' or z[0] == 'A. Yes':
                                iv_set.append(variable)
                            
                            success = True

                        else:
                            success = False


                    except KeyError:
                        success = False
                        continue 

        return iv_set
    
    def backdoor_program(self):
            
        return guidance(    
            '''
            {{#system~}}
            You are a helpful assistant on causal reasoning and arctic sea ice and atmosphere sciences. Your goal is to answer questions factually and concisely about whether and how a causal effect can be estimated using your domain knowledge on artic sea ice and atmosphere sciences.
            {{~/system}} 

            {{#user~}}
            Is {{variable}} a confounder that when conditioned on can unbias the {{treatment}} -> {{outcome}} causal effect estimation? 
            A. Yes
            B. No
            Let's think step-by-step to make sure that we have the right answer. Keep your argument and references to no more than one paragraph, otherwise you lose points, and wrap it within the tags, <explanation>...</explanation>. Then provide your final answer within the tags, <answer>A/B</answer>.
            {{~/user}}
                        
            {{#assistant~}}
            {{gen 'backdoor_set' temperature=0.5}}
            {{~/assistant}}
            ''')          

    def frontdoor_program(self):
        
        return guidance( 
        '''
        {{#system~}}
        You are a helpful assistant on causal reasoning and arctic sea ice and atmosphere sciences. Your goal is to answer questions factually and concisely about whether and how a causal effect can be estimated using your domain knowledge on artic sea ice and atmosphere sciences.
        {{~/system}} 

        {{#user~}}
        Is {{variable}} a mediator that when conditioned on can unbias the {{treatment}} -> {{outcome}} causal effect estimation? 
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Keep your argument and references to no more than one paragraph, otherwise you lose points, and wrap it within the tags, <explanation>...</explanation>. Then provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'frontdoor_set' temperature=0.7}}
        {{~/assistant}}
        ''') 

    def iv_program(self):
        
        return guidance(    
        '''
        {{#system~}}
        You are a helpful assistant on causal reasoning and arctic sea ice and atmosphere sciences. Your goal is to answer questions factually and concisely about whether and how a causal effect can be estimated using your domain knowledge on artic sea ice and atmosphere sciences.
        {{~/system}} 

        {{#user~}}
        Is {{variable}} an instrumental variable that when conditioned on can unbias the {{treatment}} -> {{outcome}} causal effect estimation? 
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Keep your argument and references to no more than one paragraph, otherwise you lose points, and wrap it within the tags, <explanation>...</explanation>. Then provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'iv_set' temperature=0.3}}
        {{~/assistant}}
        ''') 