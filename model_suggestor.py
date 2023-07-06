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
        confounders = re.findall(r'<confounder>(.*?)</confounder>', output['confounder_explanation'])
        explanations = re.findall(r'<explanation>(.*?)</explanation>', output['confounder_explanation'])
        categories = re.findall(r'<category>(.*?)</category>', output['confounder_explanation'])

        # Combine confounders, explanations, and categories into a dictionary
        suggested_confounders = {confounder: explanation
               for confounder, explanation, category in zip(confounders, explanations, categories)
               if category == 'True'}

        return suggested_confounders
    
    
    def confounder_program(self) -> guidance._program.Program:
        generate_confounders = guidance('''
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. I will provide you with context about the dataset by showing you all the columns and their associated descriptions. You will then go varaible by variable and explain whether the variable is confounding the relationship between the treatment and the outcome and categorize it appropriately. Where a confounder is a common cause of both the treatment and the outcome. In essence, a confounder directly causes both the treatment and the outcome. 
        ------------------------------------------
        Input:
            treatment
                treatment_variable_name
                description of treatment_variable

            outcome
                outcome_variable_name
                description of outcome_variable

            Dataset schema with descriptions
                name_of_first_variable: Description of first variable.
                name_of_second_variable: Description of second variable.
                ...
                name_of_nth_confounder: Description of nth variable.          

        Output: 
            <confounder>name_of_first_variable</confounder>: <explanation>Description of first variable.</explanation>
            <category>True or false</category>

            <confounder>name_of_second_variable</confounder>: <explanation>Description of second variable.</explanation>
            <category>True or false</category>

            ...

            <confounder>name_of_nth_confounder</confounder>: <explanation>Description of nth variable. 
            Explanation for why and how the selected variable is or is not a confounder.</explanation>
            <category>True or false</category>
        {{~/system}}

        {{#user~}}
        Treatment: {{treatment}}
        Outcome: {{outcome}}

        Dataset schema with descriptions
        {{variables_and_descriptions}}

        What variables directly influence both the treatment and the outcome?
        {{~/user}}

        {{#assistant~}}
        {{gen 'confounder_explanation' temperature=0.7}}
        {{~/assistant}} 
        ''')

        return generate_confounders
    



    def suggest_latent_confounders(self, variables_and_descriptions: Dict[str, str], llm: guidance.llms, treatment: str, outcome: str) ->  Tuple[Dict[str, str], Dict[str, str]]:

        generate_latent_confounders = self.latent_confounder_program()
        
        output = generate_latent_confounders(
        treatment=treatment, 
        outcome=outcome, 
        variables_and_descriptions=variables_and_descriptions,
        llm=llm)

        # Find all occurrences of confounders, explanations, and categories
        confounders = re.findall(r'<latent_confounder>(.*?)</latent_confounder>', output['latent_confounder_explanation'])
        descriptions = re.findall(r'<description>(.*?)</description>', output['latent_confounder_explanation'])
        explanations = re.findall(r'<explanation>(.*?)</explanation>', output['latent_confounder_explanation'])

        # Combine confounders and descriptions into a dictionary
        latent_confounders_descriptions = {latent_confounder: description
                for latent_confounder, description in zip(confounders, descriptions)}
        
        # Combine confounders and explanations into a dictionary
        latent_confounders = {latent_confounder: explanation
                for latent_confounder, explanation in zip(confounders, explanations)}
        
        self.context_builder.latent_variables({latent_confounder: None for latent_confounder in latent_confounders.keys()})
        self.context_builder.latent_variables_descriptions(latent_confounders_descriptions)        

        return (latent_confounders_descriptions, latent_confounders)
    
    
    def latent_confounder_program(self) -> guidance._program.Program:
        generate_confounders = guidance('''
        {{#system~}}
        You are a knowledgeable assistant with expertise in causal inference, dedicated to helping identify unobserved confounders, also known as latent confounders, that are not explicitly included in my dataset. These latent confounders are variables that are not directly observed or measured but could have a significant influence on the relationship between the treatment and outcome variables of interest. To begin, I will provide you with a comprehensive overview of the dataset, including all available columns, their respective data types, and descriptions. Additionally, I will specify the treatment and outcome variables for analysis.

        In the context of causal inference, a latent confounder refers to a variable that is not explicitly present in the dataset but is a potential common cause of both the treatment and outcome variables. It is a hidden factor that can affect both the treatment assignment and the outcome measurement, leading to biased estimates if not properly accounted for.

        When suggesting latent confounders, it is important to consider whether there might already be another variable in the dataset that closely resembles the latent confounder being suggested. If so, it is important to explain why the suggested latent confounder is different from the existing variable and why it is still important to consider it as a potential confounder.

        Your expertise in causal inference will be crucial in suggesting these latent confounders, being specific about their nature and measurement, and explaining how they may impact the causal relationship between the treatment and outcome variables. By identifying and considering these hidden confounding factors accurately, we can improve the validity and accuracy of our causal analysis, gaining deeper insights into the true causal effects
        ------------------------------------------
        Input:
            Dataset schema with descriptions
                name_of_first_variable: Description of first variable.
                name_of_second_variable: Description of second variable.
                ...
                name_of_nth_confounder: Description of nth variable.     

            treatment
                treatment_variable_name

            outcome
                outcome_variable_name 

        Output: 
            <latent_confounder>name_of_first_latent_confounder</latent_confounder>
            <description>Description of first latent confounder.</description>
            <explanation>Explanation for how first latent confounder influences the causal relationship between treatment and outcome variable.</explanation>

            <latent_confounder>name_of_second_latent_confounder</latent_confounder>
            <description>Description of second latent confounder.</description>
            <explanation>Explanation for how second latent confounder influences the causal relationship between treatment and outcome variable.</explanation>    

            ...

            <confounder>name_of_nth_confounder</confounder>
            <description>Description of nth latent confounder.</description>
            <explanation>Explanation for how nth latent confounder influences the causal relationship between treatment and outcome variable.</explanation>
        {{~/system}}

        {{#user~}}
        Dataset schema with descriptions
        {{variables_and_descriptions}}

        Treatment: {{treatment}}
        Outcome: {{outcome}}

        What latent or unobserved confounders (if any at all) are influencing the causal relationship between {{treatment}} and {{outcome}}?
        {{~/user}}

        {{#assistant~}}
        {{gen 'latent_confounder_explanation' temperature=0.7}}
        {{~/assistant}} 
        ''')

        return generate_confounders


    def suggest_variable_relationships(
        self, 
        variables_and_descriptions: Dict[str, str], 
        llm: guidance.llms, latent_confounders_descriptions: Dict[str, str] = None) -> Dict[Tuple[str, str], str]:

        relevant_variables_and_descriptions = {**variables_and_descriptions, **latent_confounders_descriptions}

        generate_relationships = self.pairwise_relationship_program()

        relationships_and_descriptions: Dict[Tuple[str, str], str] = {}

        for variable_a, desc in relevant_variables_and_descriptions.items():

            for variable_b, desc in relevant_variables_and_descriptions.items():

                if variable_a != variable_b:
                    
            
                    output = generate_relationships(variables_and_descriptions=relevant_variables_and_descriptions, variable_a=variable_a, variable_b=variable_b, llm=llm)

                    parent = re.findall(r'<parent>(.*?)</parent>', output['relationships'])
                    child = re.findall(r'<child>(.*?)</child>', output['relationships'])
                    explanation = re.findall(r'<explanation>(.*?)</explanation>', output['relationships'])

                    if child and parent and explanation:
                        relationships_and_descriptions[(parent[0], child[0])] = explanation[0]

        return relationships_and_descriptions

    
    def pairwise_relationship_program(self) -> guidance._program.Program:

        generate_relationships = guidance('''
        {{#system~}}
        As an expert in causal inference, your role is to guide me in discovering the causal graph representing my dataset. Let's work together to uncover the underlying causal relationships.

        To begin, I will provide you with a comprehensive overview of the dataset by presenting you with the columns of the dataset along with their associated descriptions. These descriptions offer valuable insights into the variables and their significance within the dataset.

        Now, let's focus on two specific variables of interest. Your task is to identify whether there is a direct causal relationship between these variables and, if so, determine the direction of causality. In essence, you need to identify which variable acts as the parent and which one acts as the child in the causal relationship. It is essential to carefully analyze the relationships and explain your reasoning step by step. This process ensures that the relationship you identify is meaningful in a causal sense.

        During your analysis, consider various factors to guide your assessment. These factors include:

        1. Direction of causality: Determine the direction of causality between the variables. Identify which variable influences the other, or if there is a bidirectional relationship.

        2. Temporal relationships: Consider the temporal order of events. Assess whether one variable precedes the other in time, as causal relationships typically exhibit a temporal sequence.

        3. Conditional independence: Evaluate whether the relationship holds even after accounting for other relevant factors. Assess if the relationship is independent of potential confounding variables.

        By taking into account the direction of causality, temporal relationships, and conditional independence, you can enhance the rigor and validity of your analysis. Following this step-by-step approach, we can unravel the causal connections and construct an accurate causal graph for the dataset.

        Your expertise in causal inference will play a pivotal role in identifying the direction of causality and providing insightful explanations. Let's collaborate to uncover the hidden causal structure behind our dataset.
        ------------------------------------------
        Input:

            Dataset schema with descriptions
                name_of_first_variable: Description of first variable.
                name_of_second_variable: Description of second variable.
                ...
                name_of_nth_confounder: Description of nth variable.     

            Selected variables
                first_variable
                second_variable    

        Output (if there is a causal relationship):
            <parent>variable_a</parent>
            <child>variable_b</child>
            <explanation>Explanation for why and how variable_a is the parent of variable_b.</explanation>

        Output (if there is no causal relationship): 
            <null>
        {{~/system}}

        {{#user~}}
        Dataset schema with descriptions
        {{variables_and_descriptions}}

        Selected variables
        {{variable_a}}
        {{variable_b}}
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationships' temperature=0.7}}
        {{~/assistant}} 
        
        ''')

        return generate_relationships