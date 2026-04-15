"""
Pydantic models for the graph structure that describes a neural network.

A Graph is the top-level container holding:
  - categorized node instances (modules, activations, operators)
  - edges wiring nodes together via input/output gates
  - constructor kwargs and forward-pass inputs for the generated nn.Module
  - third-party dependency declarations
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal

class NodeProperties(BaseModel):
    """Properties of a single node instance placed in the graph."""

    node_id: str = Field(
        ...,
        description="Unique identifier for this node instance within the graph, used as the variable name in generated code (e.g. 'self.{node_id} = ...')",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor arguments passed when instantiating this node. Values may use '#ref/key' syntax to reference graph-level kwargs (e.g. '#ref/inp_dim' -> inp_dim)",
    )


class Nodes(BaseModel):
    """Container grouping all node instances by category."""

    modules: dict[str, list[NodeProperties]] = Field(
        default_factory=dict,
        description="Pre-defined neural network module nodes (e.g. MLP, GatedNet). Key is the node name in the registry, value is a list of instances",
    )
    activations: dict[str, list[NodeProperties]] = Field(
        default_factory=dict,
        description="Activation function nodes (e.g. relu, gelu). Key is the activation name in the registry, value is a list of instances",
    )
    operators: dict[str, list[NodeProperties]] = Field(
        default_factory=dict,
        description="Mathematical operator nodes (e.g. add, matmul). Key is the operator name in the registry, value is a list of instances",
    )

    torch_modules: dict[str, list[NodeProperties]] = Field(
        default_factory=dict,
        description="PyTorch library module nodes (e.g. nn.Linear, nn.Flatten). Key is the module name in the registry, value is a list of instances",
    )


    subgraphs: dict[str, list[NodeProperties]] = Field(
        default_factory=dict,
        description="Nodes that are subgraphs. Key is the subgraph name, value is the subgraph",
    )


    def to_shallow_dict(self) -> dict[Literal["modules", "activations", "operators", "torch_modules", "subgraphs"], dict[str, list[NodeProperties]]]:
        """Return all four node categories as a plain dict keyed by category name."""
        return {
            "modules": self.modules,
            "activations": self.activations,
            "operators": self.operators,
            "torch_modules": self.torch_modules,
            "subgraphs": self.subgraphs,
        }


class PrevProperties(BaseModel):
    """Describes one incoming connection (wire) from a predecessor node into the current edge's destination."""

    node_id: str = Field(
        ...,
        description="ID of the source (predecessor) node in the graph",
    )
    input_gate: int = Field(
        ...,
        description="Output gate index on the source node that provides the tensor",
    )
    input_receive: str | None = Field(
        default=None,
        description="ID of the input slot on the destination node that receives this tensor, or None if positional",
    )



class Edge(BaseModel):
    """A directed edge connecting one or more predecessor nodes to a destination node."""

    prev_nodes: tuple[PrevProperties, ...] = Field(
        ...,
        description="Tuple of incoming connections, each specifying a source node and its output gate",
    )
    node_id: str = Field(
        ...,
        description="ID of the destination node that receives the incoming tensors",
    )
    output_gates: tuple[str, ...] = Field(
        ...,
        description="IDs of the output gates on the destination node. '__default__' prefix is auto-rewritten to '{node_id}_output'",
    )

    @model_validator(mode="after")
    def refactor_output_gates(self) -> str:
        """Rewrite ``__default__`` gate prefix to ``{node_id}_output`` for readable variable names."""
        output_gates_list = []
        for output_gate in self.output_gates:
            if output_gate.startswith("__default__"):
                output_gates_list.append(f"{self.node_id}_output" + output_gate[len('__default__'):])
            else:
                output_gates_list.append(output_gate)
        self.output_gates = tuple(output_gates_list)
        return self



class Graph(BaseModel):
    """Top-level model representing a complete neural network graph.

    Parsed from JSON, this drives the code generator to produce a full
    ``nn.Module`` subclass with ``__init__`` and ``forward`` methods.
    """

    name: str = Field(
        ...,
        description="Logical name of the graph, used for display and file naming",
    )
    class_name: str = Field(
        ...,
        description="Python class name for the generated nn.Module (e.g. 'TestModel', 'SelfAttention')",
    )
    description: str = Field(
        default="No description provided",
        description="Description of the graph, used for display and subgraph documentation",
    )

    kwargs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description="Constructor kwargs for the generated class. Each value is (type_str, default, description): type_str is the Python type as a string, default is the default value (use '__required__' for mandatory args), and description explains the argument",
    )
    nodes: Nodes = Field(
        ...,
        description="All node instances in the graph, grouped by category (modules, activations, operators)",
    )
    inputs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description="Forward-pass inputs for the generated class. Each value is (type_str, default, description): type_str is the Python type as a string, default is the default value (use '__required__' for mandatory inputs), and description explains the input",
    )
    edges: list[Edge] = Field(
        ...,
        description="List of edges defining the dataflow connections between nodes",
    )
    dependencies: set[tuple[str, ...]] = Field(
        default_factory=set,
        description="Third-party import tuples required by the generated module, e.g. ('torch', 'nn') -> from torch import nn",
    )

    subgraphs: dict[str, Graph] = Field(
        default_factory=dict,
        description="Subgraphs in the graph. Key is the subgraph name, value is the subgraph. Rule: Subgraph must not contains another subgraphs, all subgraphs must be at the root",
    )


    def get_module_nodes(self) -> list[str]:
        current_nodes = set(self.nodes.modules.keys())
        for subgraph in self.subgraphs.values():
            current_nodes.update(subgraph.get_module_nodes())
        return current_nodes
