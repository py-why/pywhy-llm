
from model_suggester import ModelSuggester
import guidance

class ToTModelSuggester(ModelSuggester):       

    def _pairwise_relationship_program(self):
        
        '''
        Uses tree of thought prompting to choose best answer for pairwise relationship
        ''' 

        return guidance('''
        {{#system~}}
        You are a helpful assistant on causal reasoning. Your goal is to answer questions about cause and effect in arctic sea ice and atmosphere sciences in a factual and concise way.
        {{~/system}}

        {{#user~}}
        Can changing {{variable_a}} change {{variable_b}}?
        In one sentence, make a compelling and factual argument for the relationship between {{variable_a}} and {{variable_b}} being true. And in another sentence, make a compelling argument for it being false. 
        If no arguemnt can be made for either option, then answer "I don't know" for that option.
        Show your work by explaining how you came to your conclusion. 
        {{~/user}}

        {{#assistant~}}
        {{gen 'thoughts' temperature=0.3}}
        {{~/assistant}} 
                        
        {{#user~}}
        Now, critique the arguments for and against, and consider whether changing {{variable_a}} changes {{variable_b}}.
        A. Yes
        B. No
        Let's think step-by-step to make sure that we have the right answer. Then provide your final answer within the tags, <answer>A/B</answer>.
        {{~/user}}

        {{#assistant~}}
        {{gen 'relationship' temperature=0.3}}
        {{~/assistant}} 
        ''')
    













