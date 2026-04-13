from schemas import ModuleNode, NodeBase, __REQUIRED__


custom_dict: dict[str, NodeBase] = {}


def register_custom_node(
    node: NodeBase,
):
    custom_dict[node.name] = node

def get_custom(
    name: str,
):
    return custom_dict[name]


register_custom_node(
    ModuleNode(
        display_name="a Custom Duplicate Node to test a node that output 2 Tensor",
        name="dup",
        description="a Custom Duplicate Node to test a node that output 2 Tensor",
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
        forward_kwargs={"X": ("Tensor", __REQUIRED__, "The input tensor")},
        n_outputs=2,
    )
)
