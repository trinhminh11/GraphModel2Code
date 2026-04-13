# This file is used to register all the torch.nn modules

from schemas import __REQUIRED__, LibNode

torchlib_dict: dict[str, LibNode] = {}


def register_torchlib(lib_node: LibNode) -> None:
    torchlib_dict[lib_node.name] = lib_node


# ============================================= Pre-defined torch libraries =============================================
# ============================================= Learnable Network Modules =============================================
# Linear layers
register_torchlib(
    LibNode(
        name="linear",
        class_name="nn.Linear",
        display_name="Fully Connected/Linear/Dense Layer",
        description="Fully Connected/Linear/Dense Layer",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "in_features": (
                "int",
                __REQUIRED__,
                "The number of input features",
            ),
            "out_features": (
                "int",
                __REQUIRED__,
                "The number of output features",
            ),
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
        },
    )
)
# Convolutional layers
register_torchlib(
    LibNode(
        name="conv2d",
        class_name="nn.Conv2d",
        display_name="2D Convolution Layer",
        description="2D Convolution Layer",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "in_channels": (
                "int",
                __REQUIRED__,
                "The number of input features",
            ),
            "out_channels": (
                "int",
                __REQUIRED__,
                "The number of output channels",
            ),
            "kernel_size": (
                "int | tuple[int, int]",
                __REQUIRED__,
                "The size of the kernel",
            ),
            "stride": (
                "int | tuple[int, int]",
                1,
                "The stride of the convolution",
            ),
            "padding": (
                "int | tuple[int, int]",
                0,
                "The padding of the convolution",
            ),
            "dilation": (
                "int | tuple[int, int]",
                1,
                "The dilation of the convolution",
            ),
            "groups": (
                "int",
                1,
                "The number of groups in the convolution",
            ),
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "padding_mode": (
                "Literal['zeros', 'reflect', 'replicate', 'circular']",
                "zeros",
                "The padding mode of the convolution, default is 'zeros'",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
        },
    )
)

# RNN layers
# RNN base (for RNN, LSTM, GRU)
register_torchlib(
    LibNode(
        name="rnnbase",
        class_name="nn.RNNBase",
        display_name="RNN base layer",
        description="RNN base layer",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "mode": (
                "Literal['RNN', 'LSTM', 'GRU']",
                __REQUIRED__,
                "The type of RNN layer, can be 'RNN', 'LSTM', or 'GRU'",
            ),
            "input_size": (
                "int",
                __REQUIRED__,
                "The number of expected features in the input",
            ),
            "hidden_size": (
                "int",
                __REQUIRED__,
                "The number of features in the hidden state",
            ),
            "num_layers": (
                "int",
                1,
                "The number of recurrent layers (default is 1)",
            ),
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "batch_first": (
                "bool",
                False,
                "Whether the input and output tensors are provided as (batch, seq, feature) (default is False)",
            ),  
            "dropout": (
                "float",
                0.0,
                "The dropout probability for the dropout layer on the outputs of each RNN layer except the last layer (default is 0.0)",
            ),
            "bidirectional": (
                "bool",
                False,
                "Whether to use a bidirectional RNN (default is False)",
            ),
            "proj_size": (
                "int",
                0,
                "The number of features in the projected output (default is 0, meaning no projection)",
            ),
        },
    )
)

# RNN (a multi-layer Elman RNN)
register_torchlib(
    LibNode(
        name="rnn",
        class_name="nn.RNN",
        display_name="RNN layer",
        description="RNN layer",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "input_size": (
                "int",
                __REQUIRED__,
                "The number of expected features in the input",
            ),
            "hidden_size": (
                "int",
                __REQUIRED__,
                "The number of features in the hidden state",
            ),
            "num_layers": (
                "int",
                1,
                "The number of recurrent layers (default is 1)",
            ),
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "batch_first": (
                "bool",
                False,
                "Whether the input and output tensors are provided as (batch, seq, feature) (default is False)",
            ),  
            "dropout": (
                "float",
                0.0,
                "The dropout probability for the dropout layer on the outputs of each RNN layer except the last layer (default is 0.0)",
            ),
            "bidirectional": (
                "bool",
                False,
                "Whether to use a bidirectional RNN (default is False)",
            ),
            "proj_size": (
                "int",
                0,
                "The number of features in the projected output (default is 0, meaning no projection)",
            ),
            "nonlinearity": (
                "Literal['tanh', 'relu']",
                "tanh",
                "The non-linearity to use. Can be either 'tanh' or 'relu' (default is 'tanh')",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
            "hx": (
                "Tensor | tuple[Tensor, Tensor]",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). For RNN and GRU, hx should be a single tensor of shape (num_layers * num_directions, batch, hidden_size). For LSTM, hx should be a tuple of two tensors (h_0, c_0), each of shape (num_layers * num_directions, batch, hidden_size)",
            ),
        }
    )
)

# TODO: GRU

# TODO: LSTM

# ============================================= Non-Learnable Modules =============================================
# Like Norm, Maxpool, etc

register_torchlib(
    LibNode(
        name="layernorm",
        class_name="nn.LayerNorm",
        display_name="Layer Normalization",
        description="Layer Normalization",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "normalized_shape": (
                "int | list[int]",
                __REQUIRED__,
                "The dimensions to normalize over",
            ),
            "eps": (
                "float",
                1e-5,
                "The epsilon value to avoid division by zero",
            ),
            "elementwise_affine": (
                "bool",
                True,
                "Whether to use elementwise affine (default is True)",
            ),
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
        },
    )
)

# ============================================= Tensor Manipulation Modules =============================================
register_torchlib(
    LibNode(
        name="flatten",
        class_name="nn.Flatten",
        display_name="Flatten",
        description="Flatten the tensor",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "start_dim": (
                "int",
                1,
                "The dimension to start flattening from (default is 1)",
            ),
            "end_dim": (
                "int",
                -1,
                "The dimension to end flattening at (default is -1)",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
        },
    )
)
