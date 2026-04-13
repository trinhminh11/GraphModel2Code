from typing import Literal, overload

from schemas import ActivationNode, ModuleNode, NodeBase, OperatorNode

from .activations import get_activation, register_activation
from .custom import get_custom, register_custom_node
from .net import get_module, register_module
from .operators import get_operator, register_operator

with open("db/pytorch/utils.txt", "r") as f:
    utils_code = f.read()


@overload
def get_node(node_type: Literal["activations"], node_name: str) -> ActivationNode: ...
@overload
def get_node(node_type: Literal["operators"], node_name: str) -> OperatorNode: ...
@overload
def get_node(node_type: Literal["modules"], node_name: str) -> ModuleNode: ...
@overload
def get_node(node_type: Literal["customs"], node_name: str) -> NodeBase: ...


def get_node(
    node_type: Literal["activations", "operators", "modules", "customs"],
    node_name: str,
):
    if node_type == "activations":
        return get_activation(node_name)
    elif node_type == "operators":
        return get_operator(node_name)
    elif node_type == "modules":
        return get_module(node_name)
    elif node_type == "customs":
        return get_custom(node_name)


__all__ = [
    "register_activation",
    "register_operator",
    "register_module",
    "register_custom_node",
    "get_operator",
    "get_module",
    "get_custom",
    "get_node",
    "utils_code",
]
