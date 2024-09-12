import unittest
import pytest
from guidance.models._mock import MockChat

import pywhyllm

class TestModelSuggester(unittest.TestCase):

    def test_pairwise_relationship(self):
        llm = MockChat()
        m = pywhyllm.SimpleModelSuggester()
        model_edges = m.suggest_pairwise_relationship(llm, "smoking", "cancer")
        


if __name__ == "__main__":
    unittest.main()
