from pydantic import BaseModel, Field, model_validator
from typing import Any


class Nodes(BaseModel):
    networks: dict[str, dict[str, dict]] = Field(default_factory=dict, description="The networks of the node, key is the network name, value is a dict of key -> node_id, value -> node_kwargs")
    activations: dict[str, dict[str, dict]] = Field(default_factory=dict, description="The activations of the node, key is the activation name, value is a dict of key -> node_id, value -> node_kwargs")
    operators: dict[str, dict[str, dict]] = Field(default_factory=dict, description="The operators of the node, key is the operator name, value is a dict of key -> node_id, value -> node_kwargs")
    customs: dict[str, dict[str, dict]] = Field(default_factory=dict, description="The customs of the node, key is the custom name, value is a dict of key -> node_id, value -> node_kwargs")

    def to_shallow_dict(self):
        return {
            "networks": self.networks,
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
    kwargs: dict[str, Any]
    nodes: Nodes
    inputs: dict[str, tuple[str, str | None]]
    edges: list[Edge]
    dependencies: set[tuple[str, ...]]
