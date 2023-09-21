from enum import Enum


class RelationshipStrategy(Enum):
    Parent = "parent"
    Child = "child"
    Confounder = "confounder"
    IV = "iv"
    Mediator = "mediator"
    Pairwise = "pairwise"


class ModelType(Enum):
    Completion = "completion"
    Chat = "chat"
