from typing import Literal, overload

from schemas import ActivationNode, NetworkNode, NodeBase, OperatorNode

from .activations import activations_dict, register_activation
from .custom import custom_dict, register_custom_node
from .net import networks_dict, register_network
from .operators import operators_dict, register_operator

with open("db/pytorch/utils.txt", "r") as f:
    utils_code = f.read()


@overload
def get_node(node_type: Literal["activations"], node_name: str) -> ActivationNode: ...
@overload
def get_node(node_type: Literal["operators"], node_name: str) -> OperatorNode: ...
@overload
def get_node(node_type: Literal["networks"], node_name: str) -> NetworkNode: ...
@overload
def get_node(node_type: Literal["customs"], node_name: str) -> NodeBase: ...


def get_node(
    node_type: Literal["activations", "operators", "networks", "customs"],
    node_name: str,
):
    if node_type == "activations":
        return activations_dict[node_name]
    elif node_type == "operators":
        return operators_dict[node_name]
    elif node_type == "networks":
        return networks_dict[node_name]
    elif node_type == "customs":
        return custom_dict[node_name]


__all__ = [
    "register_activation",
    "register_operator",
    "register_network",
    "register_custom_node",
    "get_node",
    "utils_code",
]
