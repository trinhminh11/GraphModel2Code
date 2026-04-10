from schemas import OperatorNode, __REQUIRED__

operators_dict: dict[str, OperatorNode] = {}


def register_operator(
    node: OperatorNode,
):
    operators_dict[node.name] = node


register_operator(
    OperatorNode(
        display_name="Addition",
        name="add",
        operator_symbol="+",
        description="Addition operator",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Subtraction",
        name="sub",
        operator_symbol="-",
        description="Subtraction operator",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Multiplication",
        name="mul",
        operator_symbol="*",
        description="Multiplication operator",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Division",
        name="div",
        operator_symbol="/",
        description="Division operator",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Matrix Multiplication",
        name="matmul",
        operator_symbol="@",
        description="Matrix multiplication operator",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor"),
        },
    )
)
