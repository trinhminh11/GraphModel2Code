"""
Registry of mathematical operator nodes for PyTorch code generation.

Each operator is registered as an OperatorNode and looked up by name
at code-generation time via ``get_operator_function(symbol)``.
"""

from schemas import OperatorNode, __REQUIRED__

operators_dict: dict[str, OperatorNode] = {}


def register_operator(
    node: OperatorNode,
):
    """Add an OperatorNode to the global registry, keyed by its name."""
    operators_dict[node.name] = node

def get_operator(
    name: str,
):
    """Retrieve a registered OperatorNode by name. Raises KeyError if not found."""
    return operators_dict[name]

register_operator(
    OperatorNode(
        display_name="Addition",
        name="add",
        operator_symbol="+",
        description="Element-wise addition operator. Computes X + Y, broadcasting shapes as needed",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor (left operand)"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor (right operand)"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Subtraction",
        name="sub",
        operator_symbol="-",
        description="Element-wise subtraction operator. Computes X - Y, broadcasting shapes as needed",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor (left operand)"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor (right operand)"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Multiplication",
        name="mul",
        operator_symbol="*",
        description="Element-wise multiplication operator. Computes X * Y (Hadamard product), broadcasting shapes as needed",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor (left operand)"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor (right operand)"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Division",
        name="div",
        operator_symbol="/",
        description="Element-wise division operator. Computes X / Y (true division), broadcasting shapes as needed",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor (numerator)"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor (denominator)"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Matrix Multiplication",
        name="matmul",
        operator_symbol="@",
        description="Matrix multiplication operator. Computes X @ Y using batch-aware matrix multiplication (torch.matmul semantics)",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The first input tensor (left matrix)"),
            "Y": ("Tensor", __REQUIRED__, "The second input tensor (right matrix)"),
        },
    )
)

register_operator(
    OperatorNode(
        display_name="Transpose",
        name="transpose",
        operator_symbol="T",
        description="Transpose operator. Returns X.T, swapping the last two dimensions of the input tensor",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor to transpose"),
        },
    )
)
