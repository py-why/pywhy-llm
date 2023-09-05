# PyWhy-LLM: Leveraging Large Language Models for Causal Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
## Introduction

PyWhy-LLM is an innovative library designed to augment human expertise by seamlessly integrating Large Language Models (LLMs) into the causal analysis process. It empowers users with access to knowledge previously only available through domain experts. As part of the DoWhy community, we aim to investigate and harness the capabilities of LLMs for enhancing causal analysis process.

## Documentation and Tutorials

For detailed usage instructions and tutorials, refer to [Notebook](link_here).

## Installation

To install PyWhy-LLM, you can use pip:

```bash
pip install pywhy-llm
```

To install dependencies of this repository you can use [micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html) or conda:
```bash
micromamba create -f environment.yaml
micromamba activate pywhy-llm
```
You can also install it locally with:
```
pip install -e .
```

## Usage

PyWhy-LLM seamlessly integrates into your existing causal inference process. Import the necessary classes and start exploring the power of LLM-augmented causal analysis.

```python
from pywhy-llm import ModelSuggester, IdentificationSuggester, ValidationSuggester

```


### Modeler

```python
# Create instance of Modeler
modeler = Modeler()

# Suggest a set of potential confounders
suggested_confounders = modeler.suggest_confounders(variables=_variables, treatment=treatment, outcome=outcome, llm=gpt4)

# Suggest pair-wise relationship between variables
suggested_dag = modeler.suggest_relationships(variables=selected_variables, llm=gpt4)

plt.figure(figsize=(10, 5))
nx.draw(suggested_dag, with_labels=True, node_color='lightblue')
plt.show()
```



### Identifier


```python
# Create instance of Identifier
identifier = Identifier()

# Suggest a backdoor set, front door set, and iv set
suggested_backdoor = identifier.suggest_backdoor(llm=gpt4, treatment=treatment, outcome=outcome, confounders=suggested_confounders)
suggested_frontdoor = identifier.suggest_frontdoor(llm=gpt4, treatment=treatment, outcome=outcome,  confounders=suggested_confounders)
suggested_iv = identifier.suggest_iv(llm=gpt4, treatment=treatment, outcome=outcome,  confounders=suggested_confounders)

# Suggest an estimand based on the suggester backdoor set, front door set, and iv set
estimand = identifier.suggest_estimand(confounders=suggested_confounders, treatment=treatment, outcome=outcome, backdoor=suggested_backdoor, frontdoor=suggested_frontdoor, iv=suggested_iv, llm=gpt4)  
```



### Validator


```python
# Create instance of Validator
validator = Validator()

# Suggest a critique of the provided DAG
suggested_critiques_dag = validator.critique_graph(graph=suggested_dag, llm=gpt4)

# Suggest latent confounders
suggested_latent_confounders = validator.suggest_latent_confounders(treatment=treatment, outcome=outcome, llm=gpt4)

# Suggest negative controls
suggested_negative_controls = validator.suggest_negative_controls(variables=selected_variables, treatment=treatment, outcome=outcome, llm=gpt4)

plt.figure(figsize=(10, 5))
nx.draw(suggested_critiques_dag, with_labels=True, node_color='lightblue')
plt.show()

```

## License

PyWhy-LLM is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions, feedback, or inquiries, please reach out to [Emre Kiciman](mailto:emrek@microsoft.com) and [Rose De Sicilia](mailto:t-rdesicilia@microsoft.com).

---

By leveraging LLMs and formalizing human-LLM collaboration, PyWhy-LLM takes causal inference to new heights. Explore its potential and join us in making causal analysis more accessible and insightful.
