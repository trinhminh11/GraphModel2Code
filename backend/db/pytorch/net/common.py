from schemas import NetworkNode, __REQUIRED__

networks_dict: dict[str, NetworkNode] = {}


code_dict: dict[str, str] = {
    "mlp": '''
class {class_name}(nn.Module):
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
class {class_name}(nn.Module):
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
class {class_name}(nn.Module):
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


def register_network(
    node: NetworkNode,
):
    networks_dict[node.name] = node


register_network(
    NetworkNode(
        display_name="MLP",
        name="mlp",
        description="Multi-Layer Perceptron",
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
    )
)

register_network(
    NetworkNode(
        display_name="Gated",
        name="gated",
        description="Gated Operator (perform the operation `Y o gate(X)`) where gate is a function that takes X and returns a tensor of the same shape as Y",
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
    )
)

register_network(
    NetworkNode(
        display_name="Gated Network",
        name="gated_net",
        description="Gated Network (perform the operation `FC( FC(X) ) o gate(X)`)",
        class_name="GatedNet",
        code=code_dict["gated_net"],
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
        node_dependencies={"gated": networks_dict["gated"]},
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
    )
)
