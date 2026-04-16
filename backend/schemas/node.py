"""
Pydantic models for node definitions in the graph-to-code system.

Defines the base schema (NodeBase) and its specializations:
  - ModuleNode:     custom nn.Module with inline code template and dependencies
  - LibNode:        a library-provided class (e.g. nn.Flatten) with no custom code
  - ActivationNode: an activation function resolved at runtime via get_activation()
  - OperatorNode:   a binary/unary operator resolved at runtime via get_operator_function()
"""

from __future__ import annotations

import ast
from string import Formatter
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .base import __REQUIRED__
from .enum import Tags


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
    model_config = ConfigDict(frozen=True)

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
    dependencies: set[tuple[str, ...]] = Field(
        default_factory=set,
        description="Dependencies required by this node. Each tuple maps to an import statement, e.g. ('torch', 'nn') -> from torch import nn",
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

    outputs: tuple[tuple[str, str], ...] = Field(
        default_factory=lambda: (("Tensor", "The output tensor"),),
        description="Outputs of this node. Each value is a 2-tuple (type_str, description): type_str is the Python type as a string, and description explains the output. Default is (Tensor, The output tensor), you have to change this if you have multiple outputs or different types of outputs",
    )

    tags: set[Tags] = Field(
        default_factory=set,
        description="tags of the node. This is used to group nodes together for UI display purposes",
    )

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    @field_validator("kwargs")
    @classmethod
    def validate_literal_kwargs(cls, kwargs: dict[str, tuple[str, Any, str]]):
        """Ensure that every kwarg whose type is ``Literal[...]`` has a default that belongs to that literal set."""
        for key, (type_, default, _) in kwargs.items():
            if not validate_literal(type_, default):
                raise ValueError(
                    f"The default value {default} is not in the literal list {type_} for the key {key}"
                )
        return kwargs

    @property
    def n_required_inputs(self) -> int:
        return sum(
            1
            for _, (_, default, _) in self.forward_kwargs.items()
            if default == __REQUIRED__
        )

    @property
    def n_inputs(self) -> int:
        return len(self.forward_kwargs)

    def get_dependencies(
        self,
    ) -> set[tuple[str, ...]]:
        """Return a set of copied dependencies."""
        return self.dependencies.copy()

    def get_assign_code(self) -> str:
        raise NotImplementedError("get_assign_code is not implemented for NodeBase")

    def get_creation_code(self) -> str:
        raise NotImplementedError("get_creation_code is not implemented for NodeBase")


class ClassNode(NodeBase):
    class_name: str = Field(
        ...,
        description="Fully-qualified or short name of the library class (e.g. 'nn.Flatten')",
    )

    is_abstract: bool = Field(
        default=False,
        description="Whether this node is an abstract class (i.e. not instantiable) and should not be shown in the UI selection list",
    )

    def get_assign_code(self, include_default_value: bool = False, **kwargs) -> str:
        """Build an instantiation expression like ``ClassName(arg=value, ...)``.

        Resolves ``#ref/key`` strings to bare variable references and
        validates Literal constraints on supplied values.
        """

        var_key: dict[str, tuple[type, Any]] = {}

        for key, (type_, default, _) in self.kwargs.items():
            if key in kwargs:
                if not validate_literal(type_, kwargs[key]):
                    raise ValueError(
                        f"The value {kwargs[key]} is not in the literal list {type_} for the key {key} of the node {self.class_name}"
                    )

                var_key[key] = (type_, kwargs[key])
            elif default == __REQUIRED__:
                raise ValueError(
                    f"The argument {key} is required for the node {self.class_name}"
                )
            elif include_default_value:
                var_key[key] = (type_, default)

        kwargs_lst = []

        for key, (type_, value) in var_key.items():
            if isinstance(value, str) and value.startswith("#ref/"):
                ref_key = value[len("#ref/") :]
                kwargs_lst.append(f"{key}={ref_key}")
            else:
                kwargs_lst.append(f"{key}={value!r}")

        return f"{self.class_name}({', '.join(kwargs_lst)})"


class FunctionNode(NodeBase):
    """A node representing a function that can use right away without defining a variable for it"""

    function_name: str = Field(
        ...,
        description="The name of the function to use",
    )


class CodeNode(NodeBase):
    identifier: str = Field(
        ...,
        description="The identifier of the code node",
    )
    code: str = Field(
        ...,
        description="Python source template for the nn.Module class. Format placeholders: {identifier}, {description}, and dependency class names",
    )
    node_dependencies: dict[str, CodeNode] = Field(
        default_factory=dict,
        description="Map of placeholder name -> ModuleNode for nodes whose classes must be generated before this one",
    )

    code_file: tuple[str, ...] = Field(
        default=...,
        description="the tuple that guide the code generator to generate the code for this module (e.g (net, common) -> net/common.py)",
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, code: str) -> str:
        identifier_check = False
        description_check = False
        for _, field, *_ in Formatter().parse(code):
            if field == "identifier":
                identifier_check = True
            elif field == "description":
                description_check = True
        if not identifier_check:
            raise ValueError(
                f"The code {code} is not a valid code template, it must contain the identifier field"
            )
        if not description_check:
            raise ValueError(
                f"The code {code} is not a valid code template, it must contain the description field"
            )
        return code

    def get_creation_code(self) -> str:
        """Return the full class source for this module, including any dependency classes prepended."""
        code_str = ""
        # for name, node in self.node_dependencies.items():
        #     current_code = node.get_creation_code()
        #     if current_code is not None:
        #         code_str += current_code
        #         code_str += "\n"

        code_str += self.code.format(
            identifier=self.identifier,
            description=self.description,
            **{
                node_name: node.identifier
                for node_name, node in self.node_dependencies.items()
            },
        )

        return code_str

    def get_dependencies(self, *code_root: str) -> dict[Literal["dependencies", "code_dependencies"], set[tuple[str, ...]]]:
        """Merge this node's dependencies with all transitive ``node_dependencies``."""
        dependencies: dict[str, set[tuple[str, ...]]] = {
            "dependencies": super().get_dependencies(),
            "code_dependencies": set(),
        }

        for _, node in self.node_dependencies.items():
            dependencies["code_dependencies"].add((*code_root, *self.code_file, node.identifier))

        return dependencies


class FunctionCodeNode(FunctionNode, CodeNode):
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: set[Tags]) -> set[Tags]:
        """Auto-add the function tags after model initialization.
        """
        tags.add(Tags.FUNCTION)
        return tags

    @model_validator(mode="before")
    @classmethod
    def validate_identifier(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["identifier"] = data["function_name"]
        return data


class ModuleNode(ClassNode, CodeNode):
    """A node backed by a custom ``nn.Module`` whose source is stored in ``code``.

    ``code`` is a string template that uses ``{class_name}``, ``{description}``,
    and any keys from ``node_dependencies`` as format placeholders.

    ``node_dependencies`` lists other ModuleNodes whose generated classes must
    be emitted before this one (transitive code dependencies).
    """

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: set[Tags]) -> set[Tags]:
        """Auto-add the module tags after model initialization.
        """
        tags.add(Tags.MODULE)
        return tags

    @model_validator(mode="before")
    @classmethod
    def validate_identifier(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["identifier"] = data["class_name"]
        return data


class LibNode(ClassNode):
    """A node representing a library-provided class (e.g. ``nn.Flatten``).

    Unlike ModuleNode, LibNode has no inline ``code`` -- it is simply imported
    from the library specified in the dependency sets.
    """

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: set[Tags]) -> set[Tags]:
        """Auto-add the lib tags after model initialization.
        """
        tags.add(Tags.LIB)
        return tags


class ActivationNode(NodeBase):
    """A node representing an activation function resolved at runtime via ``get_activation(name)``."""

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: set[Tags]) -> set[Tags]:
        """Auto-add the activation tags after model initialization.
        """
        tags.add(Tags.ACTIVATION)
        return tags

    def model_post_init(self, context: Any, /) -> None:
        """Auto-add the ``get_activation`` local dependency after model initialization."""
        super().model_post_init(context)
        self.dependencies.add(("utils", "get_activation"))

    def get_assign_code(self) -> str:
        """Return the runtime lookup expression for this activation."""
        return f"get_activation('{self.name}')"


class OperatorNode(NodeBase):
    """A node representing a mathematical operator resolved at runtime via ``get_operator_function(symbol)``."""

    operator_symbol: str = Field(
        ...,
        description="The mathematical symbol for the operator (e.g. '+', '-', '*', '/', '@', 'T')",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: set[Tags]) -> set[Tags]:
        """Auto-add the operator tags after model initialization.
        """
        tags.add(Tags.OPERATOR)
        return tags

    def model_post_init(self, context: Any, /) -> None:
        """Auto-add the ``get_operator_function`` local dependency after model initialization."""
        super().model_post_init(context)
        self.dependencies.add(("utils", "get_operator_function"))

    def get_assign_code(self) -> str:
        """Return the runtime lookup expression for this operator."""
        return f"get_operator_function('{self.operator_symbol}')"


# class FunctionLibNode(LibNode):
#     """A node representing a function that is a library function"""

#     pass


def main():
    test = ActivationNode(name="torch").get_assign_code()

    print(test)


if __name__ == "__main__":
    main()
