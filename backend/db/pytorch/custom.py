"""
Registry of custom (user-defined) nodes for PyTorch code generation.

Custom nodes are arbitrary nn.Module subclasses that don't fit the standard
module/activation/operator categories. They are registered as ModuleNode
instances and looked up by name via ``get_custom(name)``.
"""

from schemas import ModuleNode, NodeBase, __REQUIRED__


custom_dict: dict[str, NodeBase] = {}


def register_custom_node(
    node: NodeBase,
):
    """Add a custom NodeBase (or subclass) to the global registry, keyed by its name."""
    custom_dict[node.name] = node

def get_custom(
    name: str,
):
    """Retrieve a registered custom node by name. Raises KeyError if not found."""
    return custom_dict[name]


register_custom_node(
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
    )
)
