from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
import ast
from typing import Any
from .base import __REQUIRED__


def validate_literal(type_: str, default: Any):
    type_ = type_.strip()
    if type_.startswith("Literal["):
        literal_list = list(ast.literal_eval(type_.replace("Literal", "")))
        if default not in literal_list:
            return False
    return True

class NodeBase(BaseModel):
    display_name: str = Field(..., description="The name of the node to display in the UI")
    name: str

    description: str = Field(default=..., description="The description of the node")
    system_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set, description="The libraries that the node depends on, the tuple is the dependencies of the library e.g ('os') -> import os"
    )
    third_party_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set, description="The libraries that the node depends on, the tuple is the dependencies of the library e.g ('torch', 'nn') -> from torch import nn"
    )
    local_dependencies: set[tuple[str, ...]] = Field(
        default_factory=set, description="The libraries that the node depends on, the tuple is the dependencies of the library e.g ('utils', 'get_activation') -> from utils import get_activation"
    )
    kwargs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the class, the tuple[str, Any, str]the first term is the type of argument in string, the second term is the default value of the argument, if the second term is __REQUIRED__ it's mean that this is the required input, the last term is the description of the argument",
    )
    forward_kwargs: dict[str, tuple[str, Any, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the forward method, the tuple[str, Any, str]the first term is the type of argument in string, the second term is the default value of the argument, if the second term is __REQUIRED__ it's mean that this is the required input, the last term is the description of the argument",
    )
    n_outputs: int = Field(default=1, description="The number of outputs of the node")


    @field_validator("kwargs")
    @classmethod
    def validate_literal_kwargs(cls, kwargs: dict[str, tuple[str, Any, str]]):
        for key, (type_, default, _) in kwargs.items():
            if not validate_literal(type_, default):
                raise ValueError(f"The default value {default} is not in the literal list {type_} for the key {key}")
        return kwargs

    def get_dependencies(self):
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
    """
    The ModuleNode is a node the represents a module that are created by the code specified in the properties.code.

    class_name: use to defined the class name of the node, properties.code.format(class_name=class_name)
    node_dependencies: that mean if the code inside this Node need the creation of another Node, you can use the node_dependencies to define the dependencies.
    """

    class_name: str = Field(..., description="The name of the class of the node")
    code: str = Field(
        ...,
        description="The code of the node, if empty, the node is a function import from the library specified in lib_dependencies with the name specified in class_name",
    )
    node_dependencies: dict[str, ModuleNode] = Field(
        default_factory=dict, description="The nodes that the node depends on"
    )


    def get_creation_code(self) -> str:
        """
        return the code that is used to create the class
        """
        code_str = ""
        for name, node in self.node_dependencies.items():
            current_code = node.get_creation_code()
            if current_code is not None:
                code_str += current_code
                code_str += "\n"

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
        """
        return the code that is used to create the variable of the class like self.a = A(b=4), this function will generate A(b=4)
        """

        var_key: dict[str, tuple[type, Any]] = {}

        for key, (type_, default, _) in self.kwargs.items():
            if key in kwargs:               # if the key is in the kwargs, use the value from the kwargs
                if not validate_literal(type_, kwargs[key]):
                    raise ValueError(f"The value {kwargs[key]} is not in the literal list {type_} for the key {key} of the node {self.class_name}")

                var_key[key] = (type_, kwargs[key])
            elif default == __REQUIRED__:   # if the key is required and not in the kwargs, raise an error
                raise ValueError(f"The argument {key} is required for the node {self.class_name}")
            elif include_default_value:        # if the key is not in the kwargs and is not required, and include_default_value is True, use the default value
                var_key[key] = (type_, default)

        kwargs_lst = []

        for key, (type_, value) in var_key.items():
            if isinstance(value, str) and value.startswith("#ref/"):
                ref_key = value[len("#ref/"):]
                kwargs_lst.append(f"{key}={ref_key}")
            else:
                kwargs_lst.append(f"{key}={value!r}")

        return f"{self.class_name}({", ".join(kwargs_lst)})"

    def get_dependencies(self) -> set[str]:
        dependencies = super().get_dependencies()
        for node_name, node in self.node_dependencies.items():
            current_node_dependencies = node.get_dependencies()
            dependencies["system_lib"].update(current_node_dependencies["system_lib"])
            dependencies["third_party_lib"].update(current_node_dependencies["third_party_lib"])
            dependencies["local_lib"].update(current_node_dependencies["local_lib"])
        return dependencies


class LibNode(NodeBase):
    """LibNode is a node that represents a library function that is imported from the library specified in lib_dependencies with the name specified in class_name"""
    class_name: str = Field(..., description="The name of the class of the library (e.g: nn.Flatten)")


class ActivationNode(NodeBase):
    """ActivationNode is a node that represents an activation function that is imported from the library specified in lib_dependencies with the name specified in name"""
    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a ActivationNode"""
        super().model_post_init(context)
        self.local_dependencies.add(("utils", "get_activation"))

    def get_var_code(self) -> str:
        """
        return the code that is used to create the activation function
        """
        return f"get_activation('{self.name}')"

class OperatorNode(NodeBase):
    """OperatorNode is a node that represents an operator that is imported from the library specified in lib_dependencies with the name specified in name"""
    operator_symbol: str = Field(..., description="The symbol of the operator")
    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a ActivationNode"""
        super().model_post_init(context)
        self.local_dependencies.add(("utils", "get_operator_function"))

    def get_var_code(self) -> str:
        """
        return the code that is used to create the operator function
        """
        return f"get_operator_function('{self.operator_symbol}')"

def main():
    test = ActivationNode(name="torch").get_var_code()

    print(test)


if __name__ == "__main__":
    main()
