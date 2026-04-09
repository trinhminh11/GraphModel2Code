from pydantic import BaseModel, Field
from typing import Any, TypeVar

T = TypeVar("T")


class NodeProperties(BaseModel):
    code: str = Field(
        ...,
        description="The code of the node, if empty, the node is a function import from the library specified in lib_dependencies with the name specified in class_name",
    )
    description: str = Field(default="", description="The description of the node")
    lib_dependencies: set[str] = Field(
        default_factory=set, description="The libraries that the node depends on"
    )
    kwargs: dict[str, tuple[str, str | None, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the class, the tuple[str, str|None, str]the first term is the type of argument in string, the second term is the default value of the argument in string, None mean that the argument is required, the last term is the description of the argument",
    )
    forward_kwargs: dict[str, tuple[str, str | None, str]] = Field(
        default_factory=dict,
        description="The kwargs to use the forward method, the tuple[str, str|None, str]the first term is the type of argument in string, the second term is the default value of the argument in string, None mean that the argument is required, the last term is the description of the argument",
    )
    n_outputs: int = Field(default=1, description="The number of outputs of the node")


class NodeBase(BaseModel):
    name: str
    properties: NodeProperties = Field(
        default=..., description="The properties of the node"
    )

    def get_dependencies(self) -> set[str]:
        current_dependencies = self.properties.lib_dependencies.copy()
        return current_dependencies

    def get_var_code(self) -> str:
        raise NotImplementedError("get_var_code is not implemented for NodeBase")

    def get_creation_code(self) -> str:
        raise NotImplementedError("get_creation_code is not implemented for NodeBase")

    def get_required_inputs(self) -> set[str]:
        ret = set()
        for name, tuple_ in self.properties.forward_kwargs.items():
            if tuple_[1] is None:
                ret.add(name)
        return ret


class ClassNode(NodeBase):
    """
    The ClassNode is a node the represents a class that are created by the code specified in the properties.code.

    class_name: use to defined the class name of the node, properties.code.format(class_name=class_name)
    node_dependencies: that mean if the code inside this Node need the creation of another Node, you can use the node_dependencies to define the dependencies.
    """

    class_name: str = Field(..., description="The name of the class of the node")
    node_dependencies: dict[str, ClassNode] = Field(
        default_factory=dict, description="The nodes that the node depends on"
    )

    def model_post_init(self, context: Any, /) -> None:
        """Ensure that the code is specified for a ClassNode"""
        super().model_post_init(context)
        if self.properties.code is None:
            raise ValueError("Code is required for a ClassNode")

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

        code_str += self.properties.code.format(
            class_name=self.class_name,
            **{
                node_name: node.class_name
                for node_name, node in self.node_dependencies.items()
            },
        )

        return code_str

    def get_var_code(self) -> str:
        """
        return the code that is used to create the variable of the class like self.a = A(b=4), this function will generate A(b=4)
        """
        kwargs_str = ", ".join(
            [f"{name}={tuple_[1]}" for name, tuple_ in self.properties.kwargs.items()]
        )
        return f"{self.class_name}({kwargs_str})"

    def get_dependencies(self) -> set[str]:
        current_dependencies = super().get_dependencies()
        for node_name, node in self.node_dependencies.items():
            current_dependencies.update(node.get_dependencies())
        return current_dependencies


class LibNode(NodeBase):
    """LibNode is a node that represents a library function that is imported from the library specified in lib_dependencies with the name specified in class_name"""
    def get_var_code(self) -> str:
        """
        return the code that is used to create the variable of the class like self.a = A(b=4), this function will generate A(b=4)
        """
        kwargs_str = ", ".join(
            [f"{name}={tuple_[1]}" for name, tuple_ in self.properties.kwargs.items()]
        )
        return f"{self.properties.code}({kwargs_str})"


def main():
    test = LibNode(name="torch").get_var_code()

    print(test)


if __name__ == "__main__":
    main()
