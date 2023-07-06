from typing import List, Dict, Set, Tuple, Protocol
from protocols import IdentifierProtocol
import dodiscover as dd
import guidance
import re

class AnalysisSuggestor(IdentifierProtocol):

    backdoor_prompt = """You are a causal inference expert and your task is to identifying the backdoor set for a specific causal relationship in a directed acyclic graph (DAG). The DAG is represented as a list of edges, where each edge represents a causal relationship between two variables. Your task is to provide me with the backdoor set for a given causal relationship.

    The backdoor set for a causal relationship in a DAG is defined as a set of variables that satisfies two conditions:

    1. It blocks all directed paths from the causal variable to the outcome variable.
    2. It does not contain any descendants of the causal variable.

    Please provide the backdoor set for the following causal relationship: [Causal Relationship of Interest]

    To clarify, the DAG is represented by the following list of edges: [List of Edges]

    Note: If no backdoor set exists for the given causal relationship, please indicate that as well."""

    frontdoor_prompt = """You are a helpful assistant with expertise in causal inference. Your task is to identify the frontdoor set for a specific causal relationship in a directed acyclic graph (DAG). The DAG is represented as a list of edges, where each edge represents a causal relationship between two variables. Your task is to provide me with the frontdoor set for a given causal relationship.

    The frontdoor set for a causal relationship in a DAG is defined as a set of variables that satisfies the following conditions:

    It satisfies the backdoor criterion, meaning it blocks all backdoor paths from the causal variable to the outcome variable.
    It does not contain any colliders between the causal variable and the outcome variable.
    It is not affected by the outcome variable.
    Please determine the frontdoor set for the following causal relationship: [Causal Relationship of Interest]

    To clarify, the DAG is represented by the following list of edges: [List of Edges]

    Note: If no frontdoor set exists for the given causal relationship, please indicate that as well."""

    iv_prompt = '''You are a helpful assistant with expertise in causal inference. In the field of causal inference, instrumental variables play a crucial role in estimating causal effects when facing unobserved confounding. Your task is to identify instrumental variables for a specific causal relationship in a directed acyclic graph (DAG), where the DAG is represented as a list of edges, where each edge denotes a causal relationship between two variables. 

    Instrumental variables are defined as variables that satisfy the following conditions:

    1. They cause the treatment variable (exposure) of interest.
    2. They asre not caused by the outcome variable.
    3. They only affect the outcome variable indirectly through their effect on the treatment variable, they have no direct effect on the outcome variable.

    Please identify the instrumental variables for the following causal relationship: [Causal Relationship of Interest]

    To assist you, here is the DAG represented by the list of edges: [List of Edges]

    Note: If no instrumental variables exist for the given causal relationship, please indicate that as well.'''