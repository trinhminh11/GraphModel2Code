"""
Pydantic models for node definitions in the graph-to-code system.

Defines the base schema (NodeBase) and its specializations:
  - ModuleNode:     custom nn.Module with inline code template and dependencies
  - LibNode:        a library-provided class (e.g. nn.Flatten) with no custom code
  - ActivationNode: an activation function resolved at runtime via get_activation()
  - OperatorNode:   a binary/unary operator resolved at runtime via get_operator_function()
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
import ast
from typing import Any
from .base import __REQUIRED__


def validate_literal(type_: str, default: Any):
    """Check that *default* is a valid member of *type_* when *type_* is a ``Literal[...]`` string.

    Returns False if *default* is not among the allowed literal values,
    True otherwise (including when *type_* is not a Literal).
    """
    type_ = type_.strip()
    if type_.startswith("Literal["):
        literal_list = list(ast.literal_eval(type_.replace("Literal", "")))
        if default not in literal_list:
            return False
    return True

class NodeBase(BaseModel):
    """Base schema shared by every node type in the system.

    Carries display metadata, dependency declarations, constructor kwargs,
    forward-pass kwargs, and the number of output tensors.
    """

    display_name: str = Field(
        ...,
        description="Human-readable label shown in the UI for this node",
    )
    name: str = Field(
        ...,
        description="Internal identifier used to look up this node in the registry",
    )

    description: str = Field(
        default=...,
        description="Brief explanation of the node's purpose, shown in the UI and embedded in generated docstrings",
    )
    system_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set,
        description="Standard-library imports required by this node. Each tuple maps to an import statement, e.g. ('os',) -> import os, ('typing', 'Callable') -> from typing import Callable",
    )
    third_party_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set,
        description="Third-party package imports required by this node. Each tuple maps to an import statement, e.g. ('torch', 'nn') -> from torch import nn",
    )
    local_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set,
        description="Project-local imports required by this node. Each tuple maps to an import statement, e.g. ('utils', 'get_activation') -> from utils import get_activation",
    )
    kwargs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description=(
            "Constructor arguments for this node. "
            "Each value is a 3-tuple (type_str, default, description): "
            "type_str is the Python type as a string, "
            "default is the default value (use __REQUIRED__ for mandatory args), "
            "and description explains the argument"
        ),
    )
    forward_kwargs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description=(
            "Forward-pass arguments for this node. "
            "Each value is a 3-tuple (type_str, default, description): "
            "type_str is the Python type as a string, "
            "default is the default value (use __REQUIRED__ for mandatory args), "
            "and description explains the argument"
        ),
    )
    n_outputs: int = Field(
        default=1,
        description="Number of output tensors this node produces (>1 for fan-out nodes like Dup)",
    )


    @field_validator("kwargs")
    @classmethod
    def validate_literal_kwargs(cls, kwargs: dict[str, tuple[str, Any, str]]):
        """Ensure that every kwarg whose type is ``Literal[...]`` has a default that belongs to that literal set."""
        for key, (type_, default, _) in kwargs.items():
            if not validate_literal(type_, default):
                raise ValueError(f"The default value {default} is not in the literal list {type_} for the key {key}")
        return kwargs

    @property
    def n_required_inputs(self) -> int:
        return sum(1 for _, (_, default, _) in self.forward_kwargs.items() if default == __REQUIRED__)

    @property
    def n_inputs(self) -> int:
        return len(self.forward_kwargs)

    def get_dependencies(self):
        """Return a dict of copied dependency sets, keyed by 'system_lib', 'third_party_lib', 'local_lib'."""
        return {
            "system_lib": self.system_dependencies.copy(),
            "third_party_lib": self.third_party_dependencies.copy(),
            "local_lib": self.local_dependencies.copy(),
        }

    def get_var_code(self) -> str:
        raise NotImplementedError("get_var_code is not implemented for NodeBase")

    def get_creation_code(self) -> str:
        raise NotImplementedError("get_creation_code is not implemented for NodeBase")

class ModuleNode(NodeBase):
    """A node backed by a custom ``nn.Module`` whose source is stored in ``code``.

    ``code`` is a string template that uses ``{class_name}``, ``{description}``,
    and any keys from ``node_dependencies`` as format placeholders.

    ``node_dependencies`` lists other ModuleNodes whose generated classes must
    be emitted before this one (transitive code dependencies).
    """

    class_name: str = Field(
        ...,
        description="Python class name used in the generated code (e.g. 'MLP', 'GatedNet')",
    )
    code: str = Field(
        ...,
        description="Python source template for the nn.Module class. Format placeholders: {class_name}, {description}, and dependency class names",
    )
    node_dependencies: dict[str, ModuleNode] = Field(
        default_factory=dict,
        description="Map of placeholder name -> ModuleNode for nodes whose classes must be generated before this one",
    )

    code_file: tuple[str, ...] = Field(
        default=...,
        description="the tuple that guide the code generator to generate the code for this module (e.g (net, common) -> net/common.py)",
    )


    def get_creation_code(self) -> str:
        """Return the full class source for this module, including any dependency classes prepended."""
        code_str = ""
        # for name, node in self.node_dependencies.items():
        #     current_code = node.get_creation_code()
        #     if current_code is not None:
        #         code_str += current_code
        #         code_str += "\n"

        code_str += self.code.format(
            class_name=self.class_name,
            description=self.description,
            **{
                node_name: node.class_name
                for node_name, node in self.node_dependencies.items()
            },
        )

        return code_str

    def get_var_code(self, include_default_value: bool = False, **kwargs) -> str:
        """Build an instantiation expression like ``ClassName(arg=value, ...)``.

        Resolves ``#ref/key`` strings to bare variable references and
        validates Literal constraints on supplied values.
        """

        var_key: dict[str, tuple[type, Any]] = {}

        for key, (type_, default, _) in self.kwargs.items():
            if key in kwargs:
                if not validate_literal(type_, kwargs[key]):
                    raise ValueError(f"The value {kwargs[key]} is not in the literal list {type_} for the key {key} of the node {self.class_name}")

                var_key[key] = (type_, kwargs[key])
            elif default == __REQUIRED__:
                raise ValueError(f"The argument {key} is required for the node {self.class_name}")
            elif include_default_value:
                var_key[key] = (type_, default)

        kwargs_lst = []

        for key, (type_, value) in var_key.items():
            if isinstance(value, str) and value.startswith("#ref/"):
                ref_key = value[len("#ref/"):]
                kwargs_lst.append(f"{key}={ref_key}")
            else:
                kwargs_lst.append(f"{key}={value!r}")

        return f"{self.class_name}({", ".join(kwargs_lst)})"

    def get_dependencies(self, code_root: tuple[str, ...] = None):
        """Merge this node's dependencies with all transitive ``node_dependencies``."""
        dependencies = super().get_dependencies()
        dependencies["code_dependencies"] = set()
        code_root = code_root if code_root is not None else tuple()

        for node_name, node in self.node_dependencies.items():
            dependencies["code_dependencies"] = set([(*code_root, *node.code_file, node.class_name), ])
            # current_node_dependencies = node.get_dependencies()
            # dependencies["code_dependencies"].update(current_node_dependencies["code_dependencies"])
            # dependencies["system_lib"].update(current_node_dependencies["system_lib"])
            # dependencies["third_party_lib"].update(current_node_dependencies["third_party_lib"])
            # dependencies["local_lib"].update(current_node_dependencies["local_lib"])
        return dependencies


class LibNode(NodeBase):
    """A node representing a library-provided class (e.g. ``nn.Flatten``).

    Unlike ModuleNode, LibNode has no inline ``code`` -- it is simply imported
    from the library specified in the dependency sets.
    """
    class_name: str = Field(
        ...,
        description="Fully-qualified or short name of the library class (e.g. 'nn.Flatten')",
    )


class ActivationNode(NodeBase):
    """A node representing an activation function resolved at runtime via ``get_activation(name)``."""

    def model_post_init(self, context: Any, /) -> None:
        """Auto-add the ``get_activation`` local dependency after model initialization."""
        super().model_post_init(context)
        self.local_dependencies.add(("utils", "get_activation"))

    def get_var_code(self) -> str:
        """Return the runtime lookup expression for this activation."""
        return f"get_activation('{self.name}')"

class OperatorNode(NodeBase):
    """A node representing a mathematical operator resolved at runtime via ``get_operator_function(symbol)``."""

    operator_symbol: str = Field(
        ...,
        description="The mathematical symbol for the operator (e.g. '+', '-', '*', '/', '@', 'T')",
    )

    def model_post_init(self, context: Any, /) -> None:
        """Auto-add the ``get_operator_function`` local dependency after model initialization."""
        super().model_post_init(context)
        self.local_dependencies.add(("utils", "get_operator_function"))

    def get_var_code(self) -> str:
        """Return the runtime lookup expression for this operator."""
        return f"get_operator_function('{self.operator_symbol}')"

def main():
    test = ActivationNode(name="torch").get_var_code()

    print(test)


if __name__ == "__main__":
    main()
