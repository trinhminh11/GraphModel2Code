from schemas import NetworkNode, NodeBase, __REQUIRED__


custom_dict: dict[str, NodeBase] = {}


def register_custom_node(
    node: NodeBase,
):
    custom_dict[node.name] = node


register_custom_node(
    NetworkNode(
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
