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
            "input": ("Tensor | PackedSequence", __REQUIRED__, "The input tensor"),
            "hx": (
                "Tensor",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). For RNN and GRU, hx should be a single tensor of shape (num_layers * num_directions, batch, hidden_size). For LSTM, hx should be a tuple of two tensors (h_0, c_0), each of shape (num_layers * num_directions, batch, hidden_size)",
            ),
        }
    )
)

# GRU
register_torchlib(
    LibNode(
        name="gru",
        class_name="nn.GRU",
        display_name="GRU layer",
        description="GRU layer",
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
        },
        forward_kwargs={
            "input": ("Tensor | PackedSequence", __REQUIRED__, "The input tensor"),
            "hx": (
                "Tensor",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). Should be of shape (num_layers * num_directions, batch, hidden_size)",
            ),
        }
    )
)

# LSTM
register_torchlib(
    LibNode(
        name="lstm",
        class_name="nn.LSTM",
        display_name="LSTM layer",
        description="LSTM layer",
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
                "The number of features in the projected output (default is 0, meaning no projection). proj_size should be less than or equal to hidden_size",
            ),
        },
        forward_kwargs={
            "input": ("Tensor | PackedSequence", __REQUIRED__, "The input tensor"),
            "hx": (
                "tuple[Tensor, Tensor]",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). Should be a tuple of two tensors (h_0, c_0), each of shape (num_layers * num_directions, batch, hidden_size)",
            ),
        }
    )
)

# RNNCellBase
register_torchlib(
    LibNode(
        name="rnncellbase",
        class_name="nn.RNNCellBase",
        display_name="RNNCell base layer",
        description="RNNCell base layer",
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
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "num_chunks": (
                "int",
                0,
                "The number of chunks to divide the gates into (default is 4, meaning the gates will be divided into 4 chunks, which is the case for LSTM. For GRU, num_chunks should be 3, and for RNN, num_chunks should be 1)",
            )
        }
    )
)

# RNNCell
register_torchlib(
    LibNode(
        name="rnncell",
        class_name="nn.RNNCell",
        display_name="RNNCell layer",
        description="RNNCell layer",
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
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",    
            ),
            "num_chunks": (
                "int",
                1,
                "The number of chunks to divide the gates into (default is 1, meaning no chunking, which is the case for RNN. For GRU, num_chunks should be 3, and for LSTM, num_chunks should be 4)",
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
                "Tensor",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). Should be of shape (batch, hidden_size)",
            ),
        }
    )
)

# LSTMCell
register_torchlib(
    LibNode(
        name="lstmcell",
        class_name="nn.LSTMCell",
        display_name="LSTMCell layer",
        description="LSTMCell layer",
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
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "num_chunks": (
                "int",
                4,
                "The number of chunks to divide the gates into (default is 4, meaning the gates will be divided into 4 chunks, which is the case for LSTM. For GRU, num_chunks should be 3, and for RNN, num_chunks should be 1)",
            )
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
            "hx": (
                "tuple[Tensor, Tensor]",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). Should be a tuple of two tensors (h_0, c_0), each of shape (batch, hidden_size)",
            ),
        }
    )
)

# GRUCell
register_torchlib(
    LibNode(
        name="grucell",
        class_name="nn.GRUCell",
        display_name="GRUCell layer",
        description="GRUCell layer",
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
            "bias": (
                "bool",
                True,
                "Whether to use bias (default is True)",
            ),
            "num_chunks": (
                "int",
                3,
                "The number of chunks to divide the gates into (default is 3, meaning the gates will be divided into 3 chunks, which is the case for GRU. For RNN, num_chunks should be 1, and for LSTM, num_chunks should be 4)",
            ),
        },
        forward_kwargs={
            "input": ("Tensor", __REQUIRED__, "The input tensor"),
            "hx": (
                "Tensor",
                None,
                "The initial hidden state (default is None, meaning zero initial hidden state). Should be of shape (batch, hidden_size)",
            ),
        }
    )
)

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

# Transformer layers
# Transformer basic layerWhether the src sequence is a causal sequence, which will result in an auto-regressive mask being applied to the src sequence (default is False)
register_torchlib(
    LibNode(
        name="transformer",
        class_name="nn.Transformer",
        display_name="Transformer layer",
        description="Transformer layer",
        third_party_dependencies={
            ("torch", "nn"),
        },
        kwargs={
            "d_model": (
                "int",
                512,
                "The number of expected features in the input (default is 512)",
            ),
            "nhead": (
                "int",
                8,
                "The number of heads in the multiheadattention models (default is 8)",
            ),
            "num_encoder_layers": (
                "int",
                6,
                "The number of sub-encoder-layers in the encoder (default is 6)",
            ),
            "num_decoder_layers": (
                "int",
                6,
                "The number of sub-decoder-layers in the decoder (default is 6)",
            ),
            "dim_feedforward": (
                "int",
                2048,
                "The dimension of the feedforward network model (default is 2048)",
            ),
            "dropout": (
                "float",
                0.1,
                "The dropout value (default is 0.1)",
            ),
            "activation": (
                "Literal['relu', 'gelu'] | Callable[[Tensor], Tensor]",
                "relu",
                """The activation function of encoder/decoder intermediate layer, 
                can be a string ("relu" or "gelu") or a unary callable. Default: relu"""
            ),
            "custom_encoder": (
                "any",
                None,
                "A custom encoder module to use instead of the default TransformerEncoder. Default is None, meaning to use the default TransformerEncoder."
            ),
            "custom_decoder": (
                "any",
                None,
                "A custom decoder module to use instead of the default TransformerDecoder. Default is None, meaning to use the default TransformerDecoder."
            ),
            "layer_norm_eps": (
                "float",
                1e-5,
                "The eps value in layer normalization components (default is 1e-5)",
             ),
             "batch_first": (
                "bool",
                True,
                "Whether the input and output tensors are provided as (batch, seq, feature) (default is False)"
             ),
             "norm_first": (
                "bool",
                False,
                "Whether to perform layer normalization before other operations (default is False, meaning to perform layer normalization after other operations)",
             ),
             "bias": (
                "bool",
                True,
                "Whether to use bias in the linear layers (default is True)",
             ),
        },
        forward_kwargs={
            "src": ("Tensor", __REQUIRED__, "The source sequence tensor"),
            "tgt": ("Tensor", __REQUIRED__, "The target sequence tensor"),
            "src_mask": (
                "Tensor",
                None,
                "The additive mask for the src sequence (default is None)",
            ),
            "tgt_mask": (
                "Tensor",
                None,
                "The additive mask for the tgt sequence (default is None)",
            ),
            "memory_mask": (
                "Tensor",
                None,
                "The additive mask for the encoder output (default is None)",
            ),
            "src_key_padding_mask": (
                "Tensor",
                None,
                "The ByteTensor mask for src keys per batch (default is None)",
            ),
            "tgt_key_padding_mask": (
                "Tensor",
                None,
                "The ByteTensor mask for tgt keys per batch (default is None)",
            ),
            "src_is_causal": (
                "bool",
                None,
                "If specified, applies a causal mask as ``src_mask``. Default: ``None``; try to detect a causal mask.",
            ),
            "tgt_is_causal": (
                "bool",
                None,
                "If specified, applies a causal mask as ``tgt_mask``. Default: ``None``; try to detect a causal mask.",
            ),
            "memory_key_padding_mask": (
                "Tensor",
                None,
                "The ByteTensor mask for memory keys per batch (default is None)",
            ),
        },
    )
)

# TODO: TransformerEncoderLayer


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

