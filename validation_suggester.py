from typing import List, Tuple, Dict
from protocols import IdentifierProtocol    
import networkx as nx   
import guidance
import copy
import re

class ValidationSuggester(IdentifierProtocol):

    def critique_graph(self, graph: nx.DiGraph, llm: guidance.llms):
        
        generator = self._critique_graph_program()
        discriminator = self._critique_program()

        critiqued_edges : Dict[Tuple[Tuple[str, str], Tuple[str, str]], str] = {}

        critiqued_graph = copy.deepcopy(graph)

        for edge in list(critiqued_graph.edges): 
            x = edge[0]
            y = edge[1]
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
                        graph.remove_edge(x, y)
                        graph.add_edge(y, x)

                    elif answer_2[0] == 'C':
                        key = ((x, y), ("Deleted", "Deleted"))
                        critiqued_edges[key] = explanation_2
                        graph.remove_edge(x, y) 

                    success = True

                except KeyError:
                    success = False
                    continue 

                except IndexError:
                    success = False
                    continue

        return (critiqued_edges, critiqued_graph)

    def _critique_graph_program(self):
        
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

    def _critique_program(self):

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

    def suggest_latent_confounders(self, llm: guidance.llms, treatment: str, outcome: str):

        generate_latent_confounders = self._latent_confounder_program()
        
        output = generate_latent_confounders(
        treatment=treatment, 
        outcome=outcome,
        llm=llm)

        # Find all occurrences of confounders, explanations, and categories
        confounders = re.findall(r'<latent_confounder>(.*?)</latent_confounder>', output['latent_confounders'])

        # Combine confounders and explanations into a dictionary
        latent_confounders = list(confounders)    

        return latent_confounders
    
    def _latent_confounder_program(self):

        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in arctic sea ice and atmosphere sciences in a factual and concise way.
        {{~/system}}

        {{#user~}}
        In a causal observational study of arctic sea ice and atmosphere science where we wish to measure the causal effect of {{outcome}} on {{treatment}}, we are trying to add unobserved common causes. To help validate our analysis, what are some examples of specific latent confounders that we may bias our results if we do not account for them? Be specific about the confounders you mention.
        Let's think step-by-step to make sure that we have the right answer. Explanations should not excede one sentence, otherwise you lose points. For each latent confounder, provide your explanation within the tags, <explanation>...</explanation> and the confounder within the tags, <latent_confounder>...</latent_confounder>
        {{~/user}}

        {{#assistant~}}
        {{gen 'latent_confounders' temperature=0.3}}
        {{~/assistant}} 
        ''')

    def suggest_negative_controls(self, variables: List[str], llm: guidance.llms, treatment: str, outcome: str):

        suggest_negative_controls = self._suggest_negative_controls_program()
        
        negative_controls = []
        not_controls = []

        for variable in variables:

            if variable is not treatment and variable is not outcome:
                success = False
                while not success: 
                    try:
                        output = suggest_negative_controls(
                        treatment=treatment, 
                        outcome=outcome, 
                        variable=variable,
                        llm=llm)
                        # Find all occurrences of confounders, explanations, and categories
                        negative_control = re.findall(r'<answer>(.*?)</answer>', output['negative_control'])

                        if negative_control == 'A' or negative_control == 'A. Yes':
                            negative_controls.append(variable)
                        else:
                            not_controls.append(variable)
                        

                        success = True

                    except KeyError:
                        success = False
                        continue 

                    except IndexError:
                        success = False
                        continue

        return (negative_controls, not_controls)

    def _suggest_negative_controls_program(self): 
        
        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in arctic sea ice and atmosphere sciences in a factual and concise way.
        {{~/system}}

        {{#user~}}
        In a causal observational study of arctic sea ice and atmosphere science where we wish to measure the causal effect of {{treatment}} on {{outcome}}, we want to identify whether {{variable}} is a negative control where we might expect to see zero treatment effect. Is {{variable}} a likely negative control?
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Provide your explanations within the tags, <explanation>...</explanation> and confounders within the tags, <answer>A/B</answer>. Explanations should be specific and not excede one sentence, otherwise you lose points. 
        {{~/user}}

        {{#assistant~}}
        {{gen 'negative_control' temperature=0.3}}
        {{~/assistant}} 
        ''')