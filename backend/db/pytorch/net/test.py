"""
Registry of custom (user-defined) nodes for PyTorch code generation.

Custom nodes are arbitrary nn.Module subclasses that don't fit the standard
module/activation/operator categories. They are registered as ModuleNode
instances and looked up by name via ``get_custom(name)``.
"""

from schemas import ModuleNode, __REQUIRED__
from .state import register_module




register_module(
    ModuleNode(
        display_name="Dup",
        name="dup",
        description="Duplicate (fan-out) node. Takes a single input tensor and returns two identical copies of it, enabling the same tensor to feed into multiple downstream nodes in the graph",
        class_name="Dup",
        code="""
class {class_name}(nn.Module):
    def forward(self, X: Tensor) -> Tensor:
        return X, X
""",
        third_party_dependencies={
            ("torch", "nn"),
            ("torch", "Tensor"),
        },
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor to duplicate")},
        n_outputs=2,
        code_file=("test", ),
    )
)
