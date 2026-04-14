"""
Registry of activation function nodes for PyTorch code generation.

Each activation is registered as an ActivationNode and looked up by name
at code-generation time via ``get_activation(name)``.
"""

from schemas import ActivationNode, __REQUIRED__

activations_dict: dict[str, ActivationNode] = {}


def register_activation(
    node: ActivationNode,
):
    """Add an ActivationNode to the global registry, keyed by its name."""
    activations_dict[node.name] = node

def get_activation(
    name: str,
):
    """Retrieve a registered ActivationNode by name. Raises KeyError if not found."""
    return activations_dict[name]

# ============================================= Pre-defined activations =============================================
register_activation(
    ActivationNode(
        display_name="GELU",
        name="gelu",
        description="Gaussian Error Linear Unit (GELU) activation function. Computes x * Phi(x) where Phi is the cumulative distribution function of the standard normal distribution. See https://huggingface.co/papers/1606.08415",
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
        description="Clipped GELU activation function. Applies GELU then clamps the output to [min, max]. Useful for quantization as it maps the negative tail to a bounded range. See https://huggingface.co/papers/2004.09602",
        kwargs={
            "min": (
                "float",
                -10,
                "Lower bound of the clipping range",
            ),
            "max": (
                "float",
                10,
                "Upper bound of the clipping range",
            ),
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Fast GELU",
        name="gelu_fast",
        description="Fast GELU activation function. A tanh-based approximation that is slower than QuickGELU but more numerically accurate. See https://github.com/hendrycks/GELUs",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="New GELU",
        name="gelu_new",
        description="New GELU activation function. Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). Identical to the implementation in OpenAI's GPT. See https://huggingface.co/papers/1606.08415",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="PyTorch GELU Tanh",
        name="gelu_pytorch_tanh",
        description="GELU activation function using PyTorch's native C-level tanh approximation. Numerically equivalent to NewGELU/FastGELU but significantly faster. See https://huggingface.co/papers/1606.08415",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Accurate GELU",
        name="gelu_accurate",
        description="Accurate GELU activation function. Faster than the default erf-based GELU and more numerically precise than QuickGELU. Introduced alongside the MEGA (Moving Average Equipped Gated Attention) architecture",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Laplace",
        name="laplace",
        description="Laplace activation function. Applies an element-wise activation based on the Laplace distribution CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2)))). Introduced in MEGA as an attention activation with bounded range and gradient for stability. See https://huggingface.co/papers/2209.10655",
        forward_kwargs={
            "X": ("Tensor", __REQUIRED__, "The input tensor"),
            "mu": (
                "float",
                0.707107,
                "The mean of the Laplace distribution (default is 1/sqrt(2) ~ 0.707107)",
            ),
            "sigma": (
                "float",
                0.282095,
                "The standard deviation of the Laplace distribution (default is 1/sqrt(2*pi) ~ 0.282095)",
            ),
        },
    ),
)


register_activation(
    ActivationNode(
        display_name="Linear / Identity",
        name="linear",
        description="Linear (identity) activation function. Returns the input unchanged: f(x) = x. Used when no non-linearity is desired",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Mish",
        name="mish",
        description="Mish activation function. Computes x * tanh(softplus(x)), a self-regularized non-monotonic activation. See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra, 2019, https://huggingface.co/papers/1908.08681)",
        kwargs={
            "use_mish_python": (
                "bool",
                False,
                "Whether to use the Python implementation instead of the native C kernel",
            )
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


register_activation(
    ActivationNode(
        display_name="Quick GELU",
        name="quick_gelu",
        description="Quick GELU activation function. Approximates GELU as x * sigmoid(1.702 * x). Very fast but less numerically accurate than other GELU variants. See https://github.com/hendrycks/GELUs",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU^2",
        name="relu2",
        description="Squared ReLU activation function. Computes relu(x)^2, providing a smooth sparse activation. Introduced in Primer: Searching for Efficient Transformers. See https://huggingface.co/papers/2109.08668v2",
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)


# ============================================= Activations that are imported from libraries =============================================

register_activation(
    ActivationNode(
        display_name="LeakyReLU",
        name="leaky_relu",
        description="Leaky ReLU activation function. Computes max(negative_slope * x, x) element-wise, allowing a small gradient for negative inputs to avoid dead neurons",
        kwargs={
            "negative_slope": (
                "float",
                0.01,
                "Slope for negative input values (controls the gradient for x < 0)",
            ),
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place to save memory",
            ),
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU",
        name="relu",
        description="Rectified Linear Unit (ReLU) activation function. Computes max(0, x) element-wise, zeroing out all negative values",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place to save memory",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="ReLU6",
        name="relu6",
        description="ReLU6 activation function. Computes min(max(0, x), 6), capping the output at 6. Commonly used in mobile and quantized architectures for numerical stability",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place to save memory",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Sigmoid",
        name="sigmoid",
        description="Sigmoid activation function. Computes 1 / (1 + exp(-x)) element-wise, mapping inputs to the range (0, 1). Often used for gating mechanisms and binary classification outputs",
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="SiLU",
        name="silu",
        description="Sigmoid Linear Unit (SiLU) activation function. Computes x * sigmoid(x), also known as Swish. Provides smooth, non-monotonic activation with self-gating properties",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place to save memory",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Swish",
        name="swish",
        description="Swish activation function. Computes x * sigmoid(x), identical to SiLU. Provides smooth, non-monotonic activation with self-gating properties",
        kwargs={
            "inplace": (
                "bool",
                False,
                "Whether to perform the operation in place to save memory",
            )
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Tanh",
        name="tanh",
        description="Hyperbolic tangent (Tanh) activation function. Computes tanh(x) element-wise, mapping inputs to the range (-1, 1). Zero-centered output makes it useful in recurrent architectures",
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="PReLU",
        name="prelu",
        description="Parametric ReLU (PReLU) activation function. Computes max(0, x) + a * min(0, x) where 'a' is a learnable parameter, allowing the network to adapt the negative-slope coefficient during training",
        kwargs={
            "num_parameters": (
                "int",
                1,
                "Number of learnable 'a' parameters (1 for a shared slope, or equal to the number of channels for per-channel slopes)",
            ),
            "init": (
                "float",
                0.25,
                "Initial value for the learnable parameter 'a'",
            ),
        },
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)

register_activation(
    ActivationNode(
        display_name="Softmax",
        name="softmax",
        description="Softmax activation function. Computes exp(x_i) / sum(exp(x_j)) along a given dimension, normalizing the input into a probability distribution that sums to 1",
        kwargs={"dim": ("int", None, "The dimension along which to compute the softmax normalization")},
        forward_kwargs={"input": ("Tensor", __REQUIRED__, "The input tensor")},
    ),
)
