from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal

class NodeProperties(BaseModel):
    node_id: str
    kwargs: dict[str, Any] = Field(default_factory=dict, description="The kwargs of the node, to use the reference of the kwargs (e.g: def __init__(self, inp_dim=128) and later: self.net = MLP(input_dim=inp_dim)), use the format #ref/key")


class Nodes(BaseModel):
    modules: dict[str, list[NodeProperties]] = Field(default_factory=dict, description="This node is a pre-defined neural network architecture from the system")
    activations: dict[str, list[NodeProperties]] = Field(default_factory=dict, description="This node is a pre-defined activation function from the system")
    operators: dict[str, list[NodeProperties]] = Field(default_factory=dict, description="This node is a pre-defined operator from the system")

    customs: dict[str, list[NodeProperties]] = Field(default_factory=dict, description="This node is a custom code that can be anything, defined by the system")


    def to_shallow_dict(self) -> dict[Literal["modules", "activations", "operators", "customs"], dict[str, list[NodeProperties]]]:
        return {
            "modules": self.modules,
            "activations": self.activations,
            "operators": self.operators,
            "customs": self.customs,
        }


class PrevProperties(BaseModel):
    node_name: str
    input_gate: str
    input_receive: str | None

    @model_validator(mode="after")
    def refactor_input_gate(self) -> str:
        if self.input_gate.startswith("__default__"):
            self.input_gate = f"{self.node_name}_output" + self.input_gate[len('__default__'):]
        return self

class Edge(BaseModel):
    prev_nodes: tuple[PrevProperties, ...]
    node_name: str
    output_gates: tuple[str, ...]

    @model_validator(mode="after")
    def refactor_output_gates(self) -> str:
        output_gates_list = []
        for output_gate in self.output_gates:
            if output_gate.startswith("__default__"):
                output_gates_list.append(f"{self.node_name}_output" + output_gate[len('__default__'):])
            else:
                output_gates_list.append(output_gate)
        self.output_gates = tuple(output_gates_list)
        return self



class Graph(BaseModel):
    name: str
    class_name: str
    kwargs: dict[str, tuple[str, Any]] = Field(default_factory=dict, description="The kwargs of the graph, the tuple is the type of the kwargs and the default value of the kwargs, if the default value is `__required__`, the kwargs is required")
    nodes: Nodes
    inputs: dict[str, tuple[str, Any]] = Field(default_factory=dict, description="The inputs of the graph, the tuple is the type of the input and the default value of the input, if the default value is `__required__`, the input is required")
    edges: list[Edge]
    dependencies: set[tuple[str, ...]] = Field(default_factory=set, description="The dependencies of the graph, the tuple is the dependencies of the library e.g ('torch', 'nn') -> from torch import nn")
