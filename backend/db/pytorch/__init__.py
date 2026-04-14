"""
PyTorch node registry -- the single entry point for looking up any node by
category and name.

Loads all sub-registries (activations, operators, modules) on import
and exposes ``get_node(node_type, node_name)`` as a unified dispatcher.

Also reads ``utils.txt`` into ``utils_code`` so the code generator can emit
a self-contained ``utils.py`` in the output project.
"""

from typing import Literal, overload

from schemas import ActivationNode, ModuleNode, OperatorNode

from .activations import get_activation, register_activation
from .net import get_module, register_module
from .operators import get_operator, register_operator

# Pre-load the utils source that will be written verbatim into generated projects.
with open("db/pytorch/utils.txt", "r") as f:
    utils_code = f.read()


@overload
def get_node(node_type: Literal["activations"], node_name: str) -> ActivationNode: ...
@overload
def get_node(node_type: Literal["operators"], node_name: str) -> OperatorNode: ...
@overload
def get_node(node_type: Literal["modules"], node_name: str) -> ModuleNode: ...


def get_node(
    node_type: Literal["activations", "operators", "modules"],
    node_name: str,
):
    """Dispatch to the appropriate sub-registry based on *node_type* and return the matching node definition."""
    if node_type == "activations":
        return get_activation(node_name)
    elif node_type == "operators":
        return get_operator(node_name)
    elif node_type == "modules":
        return get_module(node_name)
    else:
        raise ValueError(f"Invalid node type: {node_type}")


__all__ = [
    "register_activation",
    "register_operator",
    "register_module",
    "get_operator",
    "get_module",
    "get_node",
    "utils_code",
]
