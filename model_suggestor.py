from typing import List, Dict, Set, Tuple, Protocol
from protocols import ModelerProtocol
import dodiscover as dd
import guidance
import re

class ModelSuggestor(ModelerProtocol):
    

    context_builder : dd.context.Context

    def __init__(self):
        self.context_builder = dd.make_context()


    def suggest_descriptions(self, variable_names: List[str], llm: guidance.llms) -> Dict[str, str]:
    
        variables_and_descriptions : Dict[str, str] = {}

        generate_description = self.observed_variable_description_program()  

        for variable_name in variable_names:

            output = generate_description(variable=variable_name, llm=llm)

            variables_and_descriptions[variable_name] = output['description']

        self.context_builder.observed_variables(variable_names)
        self.context_builder.observed_variables_descriptions(variables_and_descriptions)
    
        return variables_and_descriptions


    def observed_variable_description_program(self) -> guidance._program.Program:
        
        generate_description = guidance(    
        '''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. You are helping me better understand a dataset by providing me with a description of the variables. I will provide you with the name of a column variable within the dataset and you will provide a description for that column variable. Let's take it step by step to make sure the description is relevant, succinct, and clear. 
        Here is an example:
        ------------------------------------------
        variable_name
        description
        ------------------------------------------
        age
        The age of the patient in years. 
        {{~/system}}

        {{#user~}}
        {{variable}}
        {{~/user}}
                    
        {{#assistant~}}
        {{gen 'description' temperature=0.7}}
        {{~/assistant}}
        ''') 

        return generate_description


    def suggest_confounders(self, variables_and_descriptions: Dict[str, str], llm: guidance.llms, treatment: str, outcome: str) -> Dict[str, str]:

        all_vars_but_t_o = ', '.join(f'{key}: {value}' for key, value in variables_and_descriptions.items() if key not in [treatment, outcome])

        generate_confounders = self.confounder_program()
        
        output = generate_confounders(
        treatment=treatment, 
        outcome=outcome, 
        variables_and_descriptions=all_vars_but_t_o,
        llm=llm)

        # Find all occurrences of confounders, explanations, and categories
        confounders = re.findall(r'<confounder>(.*?)</confounder>', output['confounders'])
        explanations = re.findall(r'<explanation>(.*?)</explanation>', output['confounders'])

        # Combine confounders, explanations, and categories into a dictionary
        suggested_confounders = dict(zip(confounders, explanations))

        return suggested_confounders
    
    
    def confounder_program(self) -> guidance._program.Program:
        generate_confounders = guidance('''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. Given a relationship of interest between a treatment variable, an outcome variable and a list of potential confounders, your task is to identify confounders present in the list.

        To be included as a confounder, a variable must satisfy the following conditions:

        1. The variable directly causes the treatment assignment.
        2. The variable directly causes the outcome of interest.
        3. The variable is not affected/caused by the treatment itself.
        4. The variable is not affected/caused by the outcome itself.
        5. The variable is not on the causal pathway between the treatment and the outcome.

        You will be provided with a list of potential confounders, and your task is to identify variables that satisfy the above conditions, i.e. are confouding the relationship between the treatment and the outcome.
        These confounders should be variables that, if unaccounted for, could lead to biased or incorrect estimates of the causal effect. Your response should include a list of these variables and an explanation of why they are considered confounders. 
        ------------------------------------------
        Input:
            treatment
                treatment_variable_name
                description of treatment_variable

            outcome
                outcome_variable_name
                description of outcome_variable

            variables under consideration
                name_of_first_variable: Description of first variable.
                name_of_second_variable: Description of second variable.
                ...
                name_of_nth_confounder: Description of nth variable.          

        Output: 
            <confounder>name_of_first_variable</confounder>: <explanation>Description of nth variable. 
            Explanation for why and how the selected variable is or is not a confounder.</explanation>

            ...

            <confounder>name_of_nth_confounder</confounder>: <explanation>Description of nth variable. 
            Explanation for why and how the selected variable is or is not a confounder.</explanation>
        {{~/system}}

        {{#user~}}
        Treatment: {{treatment}}
        Outcome: {{outcome}}

        Dataset schema with descriptions
        {{variables_and_descriptions}}

        What confounders are present in this list?
        {{~/user}}

        {{#assistant~}}
        {{gen 'confounders' temperature=0.7}}
        {{~/assistant}} 
        ''')

        return generate_confounders
    

    def suggest_latent_confounders(self, variables_and_descriptions: Dict[str, str], llm: guidance.llms, treatment: str, outcome: str) ->  Dict[str, str]:

        generate_latent_confounders = self.latent_confounder_program()
        
        output = generate_latent_confounders(
        treatment=treatment, 
        outcome=outcome, 
        variables_and_descriptions=variables_and_descriptions,
        llm=llm)

        # Find all occurrences of confounders, explanations, and categories
        confounders = re.findall(r'<latent_confounder>(.*?)</latent_confounder>', output['latent_confounders'])
        explanations = re.findall(r'<explanation>(.*?)</explanation>', output['latent_confounders'])

        # Combine confounders and explanations into a dictionary
        latent_confounders = dict(zip(confounders, explanations))    

        return latent_confounders
    
    
    def latent_confounder_program(self) -> guidance._program.Program:
        generate_latent_confounders = guidance('''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. Given a relationship of interest between a treatment variable, an outcome variable, and a list of confounders, your task is to identify confounders missing from this list.  

        To be included as a confounder, a variable must satisfy the following conditions:

        1. The variable directly causes the treatment assignment.
        2. The variable directly causes the outcome of interest.
        3. The variable is not affected/caused by the treatment itself.
        4. The variable is not affected/caused by the outcome itself.
        5. The variable is not on the causal pathway between the treatment and the outcome.

        Given a list of variables under consideration for the list of confounders, your task is to identify confounders missing from this list. 
        These confounders should be variables that, if unaccounted for, could lead to biased or incorrect estimates of the causal effect. Your response should include a list of these variables and an explanation of why they are considered confounders. 
        ------------------------------------------
        Input:
            treatment
                treatment_variable_name
                description of treatment_variable

            outcome
                outcome_variable_name
                description of outcome_variable

            variables under consideration
                name_of_first_variable: Description of first variable.
                name_of_second_variable: Description of second variable.
                ...
                name_of_nth_confounder: Description of nth variable.          

        Output: 
            <confounder>name_of_first_variable</confounder>: <explanation>Description of nth variable. 
            Explanation for why and how the selected variable is or is not a confounder.</explanation>

            ...

            <confounder>name_of_nth_confounder</confounder>: <explanation>Description of nth variable. 
            Explanation for why and how the selected variable is or is not a confounder.</explanation>
        {{~/system}}

        {{#user~}}
        Treatment: {{treatment}}
        Outcome: {{outcome}}

        Dataset schema with descriptions
        {{variables_and_descriptions}}

        What confounders are missing from this list?
        {{~/user}}

        {{#assistant~}}
        {{gen 'latent_confounders' temperature=0.7}}
        {{~/assistant}} 
        ''')

        return generate_latent_confounders


    def suggest_variable_relationships(
        self, 
        variables_and_descriptions: Dict[str, str], 
        llm: guidance.llms, latent_confounders_descriptions: Dict[str, str] = None) -> List[Tuple[str, str]]:

        if latent_confounders_descriptions:
            relevant_variables_and_descriptions = {key: value for each in latent_confounders_descriptions for key, value in variables_and_descriptions.items() if each in key}
        else:
            relevant_variables_and_descriptions = variables_and_descriptions
        
        generate_relationships = self.pairwise_relationship_program()

        relationships: List[Tuple[str, str]] = []
        success: bool = False

        for variable_a in relevant_variables_and_descriptions.keys():

            for variable_b in relevant_variables_and_descriptions.keys():

                if variable_a != variable_b and (variable_a, variable_b) not in relationships and (variable_b, variable_a) not in relationships:
                    success = False
                    while not success: 
                        try:
                            output = generate_relationships(variables_and_descriptions=relevant_variables_and_descriptions, variable_a=variable_a, variable_b=variable_b, llm=llm)

                            if not re.search(r'<null>', output['relationships']):

                                parent_match = re.search(r'<parent>(.*?)</parent>', output['relationships'])
                                child_match = re.search(r'<child>(.*?)</child>', output['relationships'])

                                relationships.append((parent_match.group(1), child_match.group(1)))

                            success = True
                        
                        except KeyError:
                            success = False
                            continue

        return relationships


    
    def pairwise_relationship_program(self) -> guidance._program.Program:

        generate_relationships = guidance('''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. Given two variables along with their description, your task is to identify which variable is the parent and which is the child. During your analysis, consider various causal factors to guide your assessment.
        ------------------------------------------
        Input:

            Dataset with descriptions
                name_of_first_variable: Description of first variable.
                ...
                name_of_nth_confounder: Description of nth variable.     

            Selected variables
                first_variable
                second_variable    

        Output (if there is a causal relationship):
            <parent>variable_a</parent>
            <child>variable_b</child>

        Output (if there is no causal relationship): 
            <null>
        {{~/system}}

        {{#user~}}
        Selected variables
        {{variable_a}}
        {{variable_b}}

        Does {{variable_a}} directly cause {{variable_b}}?
        Or does {{variable_b}} directly cause {{variable_a}}?
        Consider various causal factors to guide your assessment and only answer if you are confident in your response.
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationships' temperature=0.7}}
        {{~/assistant}} 

        ''')
        return generate_relationships