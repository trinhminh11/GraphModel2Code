from schemas import ActivationNode, __REQUIRED__

activations_dict: dict[str, ActivationNode] = {}


def register_activation(
    node: ActivationNode,
):
    activations_dict[node.name] = node


# ============================================= Pre-defined activations =============================================
register_activation(
    ActivationNode(
        display_name="GELU",
        name="gelu",
        description="GELU activation function",
        kwargs={
            "use_gelu_python": (
                "bool",
                False,
                "Whether to use the Python implementation of the GELU activation function",
            )
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Clipped GELU",
        name="clipped_gelu",
        description="Clipped GELU activation function",
        kwargs={
            "min": (
                "float",
                -10,
                "The minimum value of the clipped GELU activation function",
            ),
            "max": (
                "float",
                10,
                "The maximum value of the clipped GELU activation function",
            ),
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Fast GELU",
        name="gelu_fast",
        description="Fast GELU activation function, slower than QuickGELU but more accurate",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="New GELU",
        name="gelu_new",
        description="New GELU activation function, same as OpenAI GPT's GELU",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="PyTorch GELU Tanh",
        name="gelu_pytorch_tanh",
        description="GELU activation function using PyTorch tanh approximation",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Accurate GELU",
        name="gelu_accurate",
        description="Accurate GELU activation function, faster than default and more accurate than QuickGELU",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Laplace",
        name="laplace",
        description="Laplace activation function, introduced in MEGA as an attention activation",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor"),
            "mu": (
                "float",
                0.707107,
                "The mean of the Laplace distribution (default is 0.707107)",
            ),
            "sigma": (
                "float",
                0.282095,
                "The standard deviation of the Laplace distribution (default is 0.282095)",
            ),
        },
    ),
)


register_activation(
    ActivationNode(
        display_name="Linear / Identity",
        name="linear",
        description="Linear activation function, forwarding input directly to output",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Mish",
        name="mish",
        description="Mish activation function, see Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681)",
        kwargs={
            "use_mish_python": (
                "bool",
                False,
                "Whether to use the Python implementation of the Mish activation function",
            )
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Quick GELU",
        name="quick_gelu",
        description="Quick GELU activation function, fast but somewhat inaccurate",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU^2",
        name="relu2",
        description="ReLU^2 activation function, introduced in https://huggingface.co/papers/2109.08668v2",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


# ============================================= Activations that are imported from libraries =============================================

register_activation(
    ActivationNode(
        display_name="LeakyReLU",
        name="leaky_relu",
        description="Leaky ReLU activation function",
        kwargs={
            "negative_slope": (
                "float",
                0.01,
                "The negative slope of the Leaky ReLU activation function",
            ),
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place",
            ),
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU",
        name="relu",
        description="ReLU activation function",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU6",
        name="relu6",
        description="ReLU6 activation function",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Sigmoid",
        name="sigmoid",
        description="Sigmoid activation function",
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="SiLU",
        name="silu",
        description="SiLU activation function (the same as Swish)",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Swish",
        name="swish",
        description="Swish activation function (the same as SiLU)",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Tanh",
        name="tanh",
        description="Tanh activation function",
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="PReLU",
        name="prelu",
        description="PReLU activation function",
        kwargs={
            "num_parameters": (
                "int",
                1,
                "The number of parameters for the PReLU activation function",
            ),
            "init": (
                "float",
                0.25,
                "The initial value for the PReLU activation function",
            ),
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Softmax",
        name="softmax",
        description="Softmax activation function",
        kwargs={"dim": ("int", None, "The dimension to apply the softmax function")},
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)
