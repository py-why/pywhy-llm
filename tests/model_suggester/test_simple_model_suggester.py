import pytest
import openai

from pywhyllm.suggesters.simple_model_suggester import SimpleModelSuggester

class TestSimpleModelSuggester(object):

    def test_pairwise_relationship(self):
        # TODO: add suport for a smaller model than gpt-4 that can be loaded locally.
        with pytest.raises(openai.OpenAIError):
            modeler = SimpleModelSuggester('gpt-4')
            result = modeler.suggest_pairwise_relationship("ice cream sales", "shark attacks")
        
