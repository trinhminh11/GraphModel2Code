from pydantic import BaseModel, Field
from typing import Any, TypeVar
from .base import __REQUIRED__

T = TypeVar("T")


class NodeBase(BaseModel):
    display_name: str = Field(..., description="The name of the node to display in the UI")
    name: str

    description: str = Field(default=..., description="The description of the node")
    lib_dependencies: set[str] = Field(
        default_factory=set, description="The libraries that the node depends on"
    )
    kwargs: dict[str, tuple[str, str, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the class, the tuple[str, str, str]the first term is the type of argument in string, the second term is the default value of the argument in string, if the second term is __REQUIRED__ it's mean that this is the required input, the last term is the description of the argument",
    )
    forward_kwargs: dict[str, tuple[str, str, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the forward method, the tuple[str, str, str]the first term is the type of argument in string, the second term is the default value of the argument in string, if the second term is __REQUIRED__ it's mean that this is the required input, the last term is the description of the argument",
    )
    n_outputs: int = Field(default=1, description="The number of outputs of the node")

    def kwargs_str(self, **kwargs) -> str:
        kwargs_lst = []
        current_kwargs = self.kwargs.copy()

        for key, value in kwargs.items():
            current_kwargs[key] = (current_kwargs[key][0], value)

        for name, tuple_ in current_kwargs.items():
            if tuple_[1] == __REQUIRED__:
                kwargs_lst.append(f"{name}")
            else:
                value = tuple_[1]
                if tuple_[0] == "str":
                    value = f"'{tuple_[1]}'"
                kwargs_lst.append(f"{name}={value}")

        return ", ".join(kwargs_lst)

    def get_dependencies(self) -> set[str]:
        current_dependencies = self.lib_dependencies.copy()
        return current_dependencies

    def get_var_code(self) -> str:
        raise NotImplementedError("get_var_code is not implemented for NodeBase")

    def get_creation_code(self) -> str:
        raise NotImplementedError("get_creation_code is not implemented for NodeBase")

    def get_required_init_inputs(self) -> set[str]:
        ret = set()
        for name, tuple_ in self.kwargs.items():
            if tuple_[1] == __REQUIRED__:
                ret.add(name)
        return ret

    def get_all_init_inputs(self) -> set[str]:
        ret = set()
        for name, tuple_ in self.kwargs.items():
            ret.add(name)
        return ret

    def get_required_inputs(self) -> set[str]:
        ret = set()
        for name, tuple_ in self.forward_kwargs.items():
            if tuple_[1] == __REQUIRED__:
                ret.add(name)
        return ret

    def get_all_inputs(self) -> set[str]:
        ret = set()
        for name, tuple_ in self.forward_kwargs.items():
            ret.add(name)
        return ret

class NetworkNode(NodeBase):
    """
    The NetworkNode is a node the represents a network that are created by the code specified in the properties.code.

    class_name: use to defined the class name of the node, properties.code.format(class_name=class_name)
    node_dependencies: that mean if the code inside this Node need the creation of another Node, you can use the node_dependencies to define the dependencies.
    """

    class_name: str = Field(..., description="The name of the class of the node")
    code: str = Field(
        ...,
        description="The code of the node, if empty, the node is a function import from the library specified in lib_dependencies with the name specified in class_name",
    )
    node_dependencies: dict[str, NetworkNode] = Field(
        default_factory=dict, description="The nodes that the node depends on"
    )

    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a NetworkNode"""
        super().model_post_init(context)
        if self.code is None:
            raise ValueError("Code is required for a NetworkNode")

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
            **{
                node_name: node.class_name
                for node_name, node in self.node_dependencies.items()
            },
        )

        return code_str

    def get_var_code(self, **kwargs) -> str:
        """
        return the code that is used to create the variable of the class like self.a = A(b=4), this function will generate A(b=4)
        """

        for key in self.get_required_init_inputs():
            if key not in kwargs:
                raise ValueError(f"The argument {key} is required for the node {self.class_name}")

        for key in kwargs:
            if key not in self.get_all_init_inputs():
                raise ValueError(f"The argument {key} is not a valid input for the node {self.class_name}")


        return f"{self.class_name}({self.kwargs_str(**kwargs)})"

    def get_dependencies(self) -> set[str]:
        current_dependencies = super().get_dependencies()
        for node_name, node in self.node_dependencies.items():
            current_dependencies.update(node.get_dependencies())
        return current_dependencies


class LibNode(NodeBase):
    """LibNode is a node that represents a library function that is imported from the library specified in lib_dependencies with the name specified in class_name"""
    pass


class ActivationNode(NodeBase):
    """ActivationNode is a node that represents an activation function that is imported from the library specified in lib_dependencies with the name specified in name"""
    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a ActivationNode"""
        super().model_post_init(context)
        self.lib_dependencies.add("from utils import get_activation")

    def get_var_code(self) -> str:
        """
        return the code that is used to create the activation function
        """
        if len(self.kwargs) > 0:
            return f"get_activation('{self.name}', {self.kwargs_str()})"
        else:
            return f"get_activation('{self.name}')"

class OperatorNode(NodeBase):
    """OperatorNode is a node that represents an operator that is imported from the library specified in lib_dependencies with the name specified in name"""
    operator_symbol: str = Field(..., description="The symbol of the operator")
    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a ActivationNode"""
        super().model_post_init(context)
        self.lib_dependencies.add("from utils import get_operator_function")

    def get_var_code(self) -> str:
        """
        return the code that is used to create the operator function
        """
        if len(self.kwargs) > 0:
            return f"get_operator_function('{self.operator_symbol}', {self.kwargs_str()})"
        else:
            return f"get_operator_function('{self.operator_symbol}')"

def main():
    test = ActivationNode(name="torch").get_var_code()

    print(test)


if __name__ == "__main__":
    main()
