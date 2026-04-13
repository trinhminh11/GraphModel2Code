from schemas import ModuleNode

modules_dict: dict[str, ModuleNode] = {}

def register_module(
    node: ModuleNode,
):
    modules_dict[node.name] = node


def get_module(
    name: str,
):
    return modules_dict[name]
