from typing import List, Tuple, Dict
from protocols import IdentifierProtocol    
import guidance
import re



class ValidationSuggestor(IdentifierProtocol):

    def critique_graph(self, edges: List[Tuple[str, str]], llm: guidance.llms) ->  Tuple[Dict[Tuple[str, str], str], Tuple[str, str]]:
        
        generator = self.critique_graph_program()
        discriminator = self.critique_program()

        critiqued_edges : Dict[Tuple[Tuple[str, str], Tuple[str, str]], str] = {}

        for (x, y) in edges: 
            success = False
            while not success: 
                try:
                    output_1 = generator(variable_a=x, variable_b=y, llm=llm)

                    explanation_1 = re.findall(r'<explanation>(.*)</explanation>', output_1['relationship'], re.DOTALL)

                    answer_1 = re.findall(r'<answer>(.*)</answer>', output_1['relationship'], re.DOTALL)

                    output_2 = discriminator(variable_a=x, variable_b=y, answer=answer_1[0], explanation=explanation_1[0], llm=llm)

                    answer_2 = re.findall(r'<answer>(.*?)</answer>', output_2['relationship'], re.DOTALL)

                    explanation_2 = re.findall(r'<explanation>(.*?)</explanation>', output_2['relationship'], re.DOTALL)
                    
                    if answer_2[0] == 'A':
                        key = ((x, y), (x, y))
                        critiqued_edges[key] = explanation_2

                    elif answer_2[0] == 'B':
                        key = ((x, y), (y, x))
                        critiqued_edges[key] = explanation_2
                        edges.remove((x, y))
                        edges.append((y, x))

                    elif answer_2[0] == 'C':
                        key = ((x, y), ("Deleted", "Deleted"))
                        critiqued_edges[key] = explanation_2
                        edges.remove((x, y)) 

                    success = True

                except KeyError:
                    success = False
                    continue 

                except IndexError:
                    success = False
                    continue
        
        return (critiqued_edges, edges)

    def critique_graph_program(self) -> guidance._program.Program:
        
        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to factually and concisely answer questions about cause and effect relationships using your domain knowledge on artic sea ice and atmosphere sciences.
        {{~/system}}

        {{#user~}}
        Which cause and effect relationship is more likely?
        A. {{variable_a}} causes {{variable_b}}?
        B. {{variable_b}} causes {{variable_a}}?
        C. Neither. No causal relationship exists. 
        Let's think step-by-step to make sure that we have the right answer. Keep your argument to no more than one paragraph, otherwise you lose points, and wrap it within the tags, <explanation>...</explanation>. Then provide your final answer within the tags, <answer>A/B/C</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationship' temperature=0.3}}
        {{~/assistant}}                     
        ''')

    def critique_program(self) -> str:
        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to factually and concisely answer questions about cause and effect relationships using your domain knowledge on artic sea ice and atmosphere sciences.
        {{~/system}}

        {{#user~}}
        Analyze the output from an AI assistant. Is the final answer {{answer}} consistent with the reasoning provided by the assistant?
                        
        Question
        Which cause and effect relationship is more likely?
        A. {{variable_a}} causes {{variable_b}}?
        B. {{variable_b}} causes {{variable_a}}?
        C. Neither. No causal relationship exists. 
        Let's think step-by-step to make sure that we have the right answer. Keep your argument to no more than one paragraph, otherwise you lose points, and wrap it within the tags, <explanation>...</explanation>. Then provide your final answer within the tags, <answer>A/B/C</answer>.
            
        AI Assistant Explanation
        {{explanation}}          
                        
        Critique the AI Assistant's argument by providing your own individual version of the argument. Provide your explanations within the tags, <explanation>...</explanation> and answer your final answer to the question within the tags, <answer>A/B/C</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationship' temperature=0.3}}
        {{~/assistant}}                     
        ''')    











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
        "You are a helpful assistant on causal reasoning. Your goal is to answer questions factually and concisely about cause and effect in {{field}}"
        {{~/system}}

        {{#user~}}
        Is it true that changing {{variable_a}} can directly change {{variable_b}}? 
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Provide your final answer within the tags, <answer>A/B</answer>, and explanations within the tags, <explanation>...</explanation>.
        {{~/user}}
        
        {{#system~}}
        You are a helpful assistant with expertise in causal inference. You will be given a causal relationship and a list of variables. Your task is to identify any confounding variables present in the list. Your task is to suggest confounding variables that are not in the list. Your task is to identify missing confounders, i.e.variables that when changed, directly change both the treatment and outcome variables. 
        Wrap confounders with a <confounder> tag and explanations with an <explanation> tag.
        {{~/system}}

        {{#user~}}
        Treatment: {{treatment}}
        Outcome: {{outcome}}

        Dataset schema with descriptions
        {{variables_and_descriptions}}

        What confounders are missing from this list?
        {{~/user}}

        {{#assistant~}}
        {{gen 'latent_confounders' temperature=0.3}}
        {{~/assistant}} 
        ''')

        return generate_latent_confounders