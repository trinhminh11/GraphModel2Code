"""
Pre-defined neural network module registrations (MLP, Gated, GatedNet).

Each module is a ModuleNode with an inline code template, dependency
declarations, and kwargs metadata. The code templates use ``{class_name}``,
``{description}``, and dependency placeholders that are filled in during
code generation.
"""

from schemas import __REQUIRED__, ModuleNode, Tags

from .state import register_module, get_module

code_dict: dict[str, str] = {
    "mlp": '''
class {identifier}(nn.Module):
    """
    {description}
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: None | list[int] = None,
        bias: bool = True,
        activation_fn: str | Callable[[Tensor], Tensor] = "relu",
    ):
        """
        Args:
            input_dim: The input dimension
            output_dim: The output dimension, None means the output dimension is the same as the input dimension
            hidden_dims: The hidden dimensions, None means no hidden layers
            bias: Whether to use bias, default is True
            activation_fn: The activation function, default is "relu"
        """
        super().__init__()

        output_dim = output_dim if output_dim is not None else input_dim

        if hidden_dims is None:
            hidden_dims = []

        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(get_activation(activation_fn))

        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
''',
    "gated": '''
class {identifier}(nn.Module):
    """
    {description}
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        bias=False,
        gate_act_fn: str | Callable[[Tensor], Tensor] = "sigmoid",
        gate_operator_fn: str | Callable[[Tensor, Tensor], Tensor] = "*",
    ):
        """
        Args:
            input_dim: The input dimension
            output_dim: The output dimension, None means the output dimension is the same as the input dimension
            bias: Whether to use bias, default is False
            gate_act_fn: The activation function of the gate, default is "sigmoid"
            gate_operator_fn: The operator function of the gate, default is "*"
        """
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim

        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            get_activation(gate_act_fn)
        )

        self.operator_fn = get_operator_function(gate_operator_fn)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        # X: (..., Ex)
        # Y: (..., Ey)
        return self.operator_fn(Y, self.gate(X))  # Y o gate(X)   (..., Exy)
''',
    "gated_net": '''
class {identifier}(nn.Module):
    """
    {description}
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = False,
        gate_act_fn: str | Callable[[Tensor], Tensor] = "silu",
        gate_operator_fn: str | Callable[[Tensor, Tensor], Tensor] = "*",
    ):
        """
        Args:
            input_dim: The input dimension
            hidden_dim: The hidden dimension
            output_dim: The output dimension, None means the output dimension is the same as the input dimension
            bias: Whether to use bias, default is False
            gate_act_fn: The activation function of the gate, default is "silu"
            gate_operator_fn: The operator function of the gate, default is "*"
        """

        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim

        self.up = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.gate = {gated}(
            input_dim,
            hidden_dim,
            bias=bias,
            gate_act_fn=gate_act_fn,
            gate_operator_fn=gate_operator_fn,
        )

    def forward(self, X: Tensor) -> Tensor:
        Y = self.up(X)  # Y = fc(X)
        gate_output = self.gate(X, Y)  # Y' = Y o act(X @ W (+B) )
        return self.down(gate_output)  # out = fc(Y')
''',
}


register_module(
    ModuleNode(
        display_name="MLP",
        name="mlp",
        description="Multi-Layer Perceptron (MLP). A fully-connected feed-forward network with configurable hidden layers and activation functions. Applies alternating Linear and activation layers",
        class_name="MLP",
        code=code_dict["mlp"],
        system_dependencies={
            ("typing", "Callable"),
        },
        third_party_dependencies={
            ("torch", "nn"),
            ("torch", "Tensor"),
        },
        local_dependencies={
            ("utils", "get_activation"),
        },
        kwargs={
            "input_dim": ("int", __REQUIRED__, "The input dimension"),
            "output_dim": (
                "int",
                None,
                "The output dimension, None means the output dimension is the same as the input dimension",
            ),
            "hidden_dims": (
                "list[int]",
                None,
                "list of hidden dimensions, None means no hidden layers",
            ),
            "bias": ("bool", "True", "Whether to use bias"),
            "activation_fn": (
                "str",
                "relu",
                "The activation function",
            ),
        },
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor"),
        },
        code_file=("common", ),
        tags={Tags.COMMON, }
    )
)

register_module(
    ModuleNode(
        display_name="Gated",
        name="gated",
        description="Gated Operator. Computes Y o gate(X) where 'o' is a configurable binary operator and gate is a Linear+activation that maps X to the same shape as Y. Enables multiplicative gating patterns common in modern architectures",
        class_name="Gated",
        code=code_dict["gated"],
        system_dependencies={
            ("typing", "Callable"),
        },
        third_party_dependencies={
            ("torch", "nn"),
            ("torch", "Tensor"),
        },
        local_dependencies={
            ("utils", "get_activation"),
            ("utils", "get_operator_function"),
        },
        kwargs={
            "input_dim": ("int", __REQUIRED__, "The input dimension"),
            "output_dim": (
                "int",
                None,
                "The output dimension, None means the output dimension is the same as the input dimension",
            ),
            "bias": (
                "bool",
                False,
                "Whether to use bias, gate function rarely use bias",
            ),
            "gate_act_fn": (
                "str",
                "sigmoid",
                "The activation function of the gate",
            ),
            "gate_operator_fn": (
                "str",
                "*",
                "The operator function of the gate `o`",
            ),
        },
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor"),
            "Y": ("Tensor", __REQUIRED__, "The output tensor"),
        },
        code_file=("common", ),
        tags={Tags.COMMON, }
    )
)

register_module(
    ModuleNode(
        display_name="Gated Network",
        name="gated_net",
        description="Gated Network. Computes down(up(X) o gate(X)) where up/down are Linear projections and gate is a Gated operator. Combines projection with multiplicative gating, used in architectures like LLaMA's feed-forward blocks",
        class_name="GatedNet",
        code=code_dict["gated_net"],
        system_dependencies={
            ("typing", "Callable"),
        },
        third_party_dependencies={
            ("torch", "nn"),
            ("torch", "Tensor"),
        },
        node_dependencies={"gated": get_module("gated")},
        kwargs={
            "input_dim": ("int", __REQUIRED__, "The input dimension"),
            "hidden_dim": ("int", __REQUIRED__, "The hidden dimension"),
            "output_dim": (
                "int",
                None,
                "The output dimension, None means the output dimension is the same as the input dimension",
            ),
            "bias": (
                "bool",
                False,
                "Whether to use bias, gate function rarely use bias",
            ),
            "gate_act_fn": (
                "str",
                "silu",
                "The activation function of the gate",
            ),
            "gate_operator_fn": (
                "str",
                "*",
                "The operator function of the gate `o`",
            ),
        },
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor"),
        },
        code_file=("common", ),
        tags={Tags.COMMON, }
    )
)
