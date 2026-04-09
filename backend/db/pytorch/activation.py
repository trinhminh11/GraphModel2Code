from schemas.base import (
    ClassNode,
    LibNode,
    NodeBase,
    NodeProperties,
)

activations_dict: dict[str, NodeBase] = {}


def register_activation(
    node: NodeBase,
):
    activations_dict[node.name] = node


# ============================================= Pre-defined activations =============================================
gelu = '''class {class_name}(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, X: Tensor) -> Tensor:
        return X * 0.5 * (1.0 + torch.erf(X / math.sqrt(2.0)))

    def forward(self, X: Tensor) -> Tensor:
        return self.act(X)
'''

register_activation(
    ClassNode(
        name="gelu",
        class_name="GELUActivation",
        properties=NodeProperties(
            code=gelu,
            description="GELU activation function",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
                "import math",
            ],
            kwargs={
                "use_gelu_python": (
                    "bool",
                    "False",
                    "Whether to use the Python implementation of the GELU activation function",
                )
            },
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

clipped_gelu = '''class {class_name}(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://huggingface.co/papers/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://huggingface.co/papers/1606.08415
    """

    def __init__(self, min: float = -10, max: float = 10):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, X: Tensor) -> Tensor:
        return torch.clip({gelu}(True)(X), self.min, self.max)
'''

register_activation(
    ClassNode(
        class_name="ClippedGELUActivation",
        name="clipped_gelu",
        properties=NodeProperties(
            code=clipped_gelu,
            description="Clipped GELU activation function",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
            ],
            kwargs={
                "min": (
                    "float",
                    "-10",
                    "The minimum value of the clipped GELU activation function",
                ),
                "max": (
                    "float",
                    "10",
                    "The maximum value of the clipped GELU activation function",
                ),
            },
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
        node_dependencies={"gelu": activations_dict["gelu"]},
    ),
)

gelu_fast = '''class {class_name}(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, X: Tensor) -> Tensor:
        return (
            0.5
            * X
            * (
                1.0
                + torch.tanh(X * 0.7978845608 * (1.0 + 0.044715 * X * X))
            )
        )
'''

register_activation(
    ClassNode(
        name="gelu_fast",
        class_name="FastGELUActivation",
        properties=NodeProperties(
            code=gelu_fast,
            description="Fast GELU activation function, slower than QuickGELU but more accurate",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
            ],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

gelu_new = '''class {class_name}(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def forward(self, X: Tensor) -> Tensor:
        return (
            0.5
            * X
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (X + 0.044715 * torch.pow(X, 3.0))
                )
            )
        )
'''

register_activation(
    ClassNode(
        name="gelu_new",
        class_name="NewGELUActivation",
        properties=NodeProperties(
            code=gelu_new,
            description="New GELU activation function, same as OpenAI GPT's GELU",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
                "import math",
            ],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

gelu_pytorch_tanh = '''class {class_name}(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def forward(self, X: Tensor) -> Tensor:
        return nn.functional.gelu(X, approximate="tanh")
'''

register_activation(
    ClassNode(
        name="gelu_pytorch_tanh",
        class_name="PytorchGELUTanh",
        properties=NodeProperties(
            code=gelu_pytorch_tanh,
            description="GELU activation function using PyTorch tanh approximation",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
            ],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

gelu_accurate = '''class {class_name}(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, X: Tensor) -> Tensor:
        return (
            0.5
            * X
            * (
                1
                + torch.tanh(
                    self.precomputed_constant * (X + 0.044715 * torch.pow(X, 3))
                )
            )
        )
'''

register_activation(
    ClassNode(
        name="gelu_accurate",
        class_name="AccurateGELUActivation",
        properties=NodeProperties(
            code=gelu_accurate,
            description="Accurate GELU activation function, faster than default and more accurate than QuickGELU",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
                "import math",
            ],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

laplace = '''class {class_name}(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://huggingface.co/papers/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, X: Tensor, mu: float = 0.707107, sigma: float = 0.282095) -> Tensor:
        X = (X - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(X))
'''

register_activation(
    ClassNode(
        name="laplace",
        class_name="LaplaceActivation",
        properties=NodeProperties(
            code=laplace,
            description="Laplace activation function, introduced in MEGA as an attention activation",
            lib_dependencies=[
                "import torch",
                "import torch.nn as nn",
                "from torch import Tensor",
                "import math",
            ],
            forward_kwargs={
                "X": ("Tensor", None, "The input tensor"),
                "mu": ("float", "0.707107", "The mean of the Laplace distribution"),
                "sigma": (
                    "float",
                    "0.282095",
                    "The standard deviation of the Laplace distribution",
                ),
            },
        ),
    ),
)

linear = '''class {class_name}(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, X: Tensor) -> Tensor:
        return X
'''

register_activation(
    ClassNode(
        name="linear",
        class_name="LinearActivation",
        properties=NodeProperties(
            code=linear,
            description="Linear activation function, forwarding input directly to output",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

mish = '''class {class_name}(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        self.act = nn.functional.mish

    def forward(self, X: Tensor) -> Tensor:
        return self.act(X)
'''

register_activation(
    ClassNode(
        name="mish",
        class_name="MishActivation",
        properties=NodeProperties(
            code=mish,
            description="Mish activation function, see Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681)",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

quick_gelu = '''class {class_name}(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, X: Tensor) -> Tensor:
        return X * torch.sigmoid(1.702 * X)
'''

register_activation(
    ClassNode(
        name="quick_gelu",
        class_name="QuickGELUActivation",
        properties=NodeProperties(
            code=quick_gelu,
            description="Quick GELU activation function, fast but somewhat inaccurate",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)

relu2 = '''class {class_name}(nn.Module):
    """
    Applies the relu^2 activation introduced in https://huggingface.co/papers/2109.08668v2
    """

    def forward(self, X: Tensor) -> Tensor:
        relu_applied = nn.functional.relu(X)
        squared = torch.square(relu_applied)
        return squared
'''

register_activation(
    ClassNode(
        name="relu2",
        class_name="ReLUSquaredActivation",
        properties=NodeProperties(
            code=relu2,
            description="ReLU^2 activation function, introduced in https://huggingface.co/papers/2109.08668v2",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
        ),
    ),
)


# ============================================= Activations that are imported from libraries =============================================

register_activation(
    LibNode(
        name="leaky_relu",
        properties=NodeProperties(
            code="nn.LeakyReLU",
            description="Leaky ReLU activation function",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "negative_slope": (
                    "float",
                    "0.01",
                    "The negative slope of the Leaky ReLU activation function",
                ),
                "inplace": (
                    "bool",
                    "False",
                    "Whether to perform the operation in place",
                ),
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="relu",
        properties=NodeProperties(
            code="nn.ReLU",
            description="ReLU activation function",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "inplace": (
                    "bool",
                    "False",
                    "Whether to perform the operation in place",
                )
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="relu6",
        properties=NodeProperties(
            code="nn.ReLU6",
            description="ReLU6 activation function",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "inplace": (
                    "bool",
                    "False",
                    "Whether to perform the operation in place",
                )
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="sigmoid",
        properties=NodeProperties(
            code="nn.Sigmoid",
            description="Sigmoid activation function",
            lib_dependencies=["import torch.nn as nn"],
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="silu",
        properties=NodeProperties(
            code="nn.SiLU",
            description="SiLU activation function (the same as Swish)",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "inplace": (
                    "bool",
                    "False",
                    "Whether to perform the operation in place",
                )
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="swish",
        properties=NodeProperties(
            code="nn.SiLU",
            description="Swish activation function (the same as SiLU)",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "inplace": (
                    "bool",
                    "False",
                    "Whether to perform the operation in place",
                )
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="tanh",
        properties=NodeProperties(
            code="nn.Tanh",
            description="Tanh activation function",
            lib_dependencies=["import torch.nn as nn"],
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="prelu",
        properties=NodeProperties(
            code="nn.PReLU",
            description="PReLU activation function",
            lib_dependencies=["import torch.nn as nn"],
            kwargs={
                "num_parameters": (
                    "int",
                    "1",
                    "The number of parameters for the PReLU activation function",
                ),
                "init": (
                    "float",
                    "0.25",
                    "The initial value for the PReLU activation function",
                ),
            },
            forward_kwargs={"input": ("Tensor", None, "The input tensor")},
        ),
    ),
)

register_activation(
    LibNode(
        name="softmax",
        properties=NodeProperties(
            code="nn.Softmax",
            description="Softmax activation function",
            lib_dependencies=["import torch.nn as nn"],
        ),
        kwargs={"dim": ("int", "None", "The dimension to apply the softmax function")},
        forward_kwargs={"input": ("Tensor", None, "The input tensor")},
    ),
)


# ============================================= Testing Activations =============================================
concat_act = '''
class {class_name}(nn.Module):
    def forward(self, X: Tensor, Y: Tensor, dim: int = 1) -> Tensor:
        return torch.cat([X, Y], dim=dim)
'''

register_activation(
    ClassNode(
        name="concat",
        class_name="ConcatActivation",
        properties=NodeProperties(
            code=concat_act,
            description="Concat activation function",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={
                "X": ("Tensor", None, "The first input tensor"),
                "Y": ("Tensor", None, "The second input tensor"),
                "dim": ("int", "1", "The dimension to concatenate the tensors"),
            },
        ),
    ),
)

dup_act = '''
class {class_name}(nn.Module):
    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        return X, X
'''

register_activation(
    ClassNode(
        name="dup",
        class_name="DupActivation",
        properties=NodeProperties(
            code=dup_act,
            description="Dup activation function",
            lib_dependencies=["import torch.nn as nn", "from torch import Tensor"],
            forward_kwargs={"X": ("Tensor", None, "The input tensor")},
            n_outputs=2,
        ),
    ),
)

