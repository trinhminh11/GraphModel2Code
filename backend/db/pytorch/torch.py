# This file is used to register all the torch.nn modules

from schemas import __REQUIRED__, LibNode

torchlib_dict: dict[str, LibNode] = {}


def register_torchlib(lib_node: LibNode) -> None:
    torchlib_dict[lib_node.name] = lib_node


# ============================================= Pre-defined torch libraries =============================================
# ============================================= Learnable Network Modules =============================================
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
