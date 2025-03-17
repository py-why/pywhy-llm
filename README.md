# PyWhy-LLM: Leveraging Large Language Models for Causal Analysis
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
## Introduction

PyWhy-LLM is an innovative library designed to augment human expertise by seamlessly integrating Large Language Models (LLMs) into the causal analysis process. It empowers users with access to knowledge previously only available through domain experts. As part of the DoWhy community, we aim to investigate and harness the capabilities of LLMs for enhancing causal analysis process.

## Documentation and Tutorials

For detailed usage instructions and tutorials, refer to [Notebook](link_here).

## Installation

To install PyWhy-LLM, you can use pip:

```bash
pip install pywhyllm
```

## Usage

PyWhy-LLM seamlessly integrates into your existing causal inference process. Import the necessary classes and start exploring the power of LLM-augmented causal analysis.

```python
from pywhyllm import ModelSuggester, IdentificationSuggester, ValidationSuggester

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

## Contributors ✨
This project welcomes contributions and suggestions. For a guide to contributing and a list of all contributors, check out [CONTRIBUTING.md](https://github.com/py-why/pywhyllm/blob/main/CONTRIBUTING.md>). Our contributor code of conduct is available [here](https://github.com/py-why/governance/blob/main/CODE-OF-CONDUCT.md>).

If you encounter an issue or have a specific request for DoWhy, please [raise an issue](https://github.com/py-why/pywhyllm/issues).
