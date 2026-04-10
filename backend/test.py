from collections import deque
from typing import Literal
from utils import get_dependencies_str

from db.pytorch import get_node, utils_code

"""
                ---------
                | Input |
                ---------
                    | X-> X
                    v
                ---------
                |  MLP  |
                ---------
                    | -> X
                    v
        <--[1]  ---------
        --------|  Dup  |
        |       ---------
        |           | [0]-->
        |           |
        |           |-------------|  -> Y
        |           v  -> X       |
        |    ---------------      |
        |    |  Gated Net  |      |
        |    ---------------      |
        |           | -> X        |
        |           v             |
        |       ---------         |
        |       |  add  | <-------|
        |       ---------
        |           | -> X
        |           v
        |       ---------
        |       |  MLP  |
        |       ---------
        |           | -> input
        |           v
        |       ---------
        |       | Tanh  |
        |       ---------
        |           | -> [0]
        |->[1]      v
        |       ---------
        |------>| Output |
                ---------
"""

data = {
    "name": "testing",
    "class_name": "Testing",
    "kwargs": {},
    "nodes": {
        "networks": {
            "mlp": {
                "mlp_first": {
                    "input_dim": 64,
                    "output_dim": 128,
                    "hidden_dims": [256, 256],
                    "bias": True,
                    "activation_fn": "relu",
                },
                "mlp_last": {
                    "input_dim": 128,
                    "output_dim": 10,
                    "hidden_dims": None,
                    "bias": True,
                    "activation_fn": "relu",
                }
            },
            "gated_net": {
                "gated_net_1": {
                    "input_dim": 128,
                    "hidden_dim": 128,
                    "output_dim": 128,
                    "gate_act_fn": "silu",
                    "gate_operator_fn": "*",
                }
            },
        },
        "activations": {
            "tanh": {"tanh": {}},
        },
        "operators": {"add": {"add": {}}},
        "customs": {
            "dup": {"dup": {}},
        },
    },
    "inputs": {
        "X": ("Tensor", None),
    },
    "edges": [
        ("inputs", "mlp_first", "X", "X", ("__default__")),
        ("mlp_first", "dup", "__default__", "X", ("__default__0", "__default__1")),
        ("dup", "gated_net_1", "__default__0", "X", ("__default__")),
        ("gated_net_1", "add", "__default__", "X", ("__default__")),
        ("dup", "add", "__default__0", "Y", ("__default__")),
        ("add", "mlp_last", "__default__", "X", ("__default__")),
        ("mlp_last", "tanh", "__default__", "input", ("__default__")),
        ("tanh", "outputs", "__default__", None, None),
        ("dup", "outputs", "__default__1", None, None),
    ],
    "dependencies": {"from torch import Tensor", "import torch.nn as nn"},
}


class GraphNode:
    def __init__(self, node_type, node_id, node_name, level=0):
        self.node_id = node_id
        self.node_type = node_type
        self.node_name = node_name
        self.level = level
        self.prev: list[tuple[GraphNode, str, str]] = []

    def add_prev(
        self,
        prev_node: "GraphNode",
        input_gate: str,
        input_receive: str,
        output_gate: str,
    ):
        self.prev.append((prev_node, input_gate, input_receive, output_gate))

    def __repr__(self):
        return f"GraphNode(node_name={self.node_name}, level={self.level}, prev={[(prev_tuple[0].node_name, *prev_tuple[1:]) for prev_tuple in self.prev]})"


class ExecuteGraph:
    def __init__(self, data: dict):
        self.nodes: dict[str, GraphNode] = {}

        self.nodes["inputs"] = GraphNode("inputs", "inputs", "inputs")
        for node_type, nodes_base in data["nodes"].items():
            for node_name, nodes in nodes_base.items():
                for node_id in nodes:
                    self.nodes[node_id] = GraphNode(node_type, node_id, node_name)
        self.nodes["outputs"] = GraphNode("outputs", "outputs", "outputs")

        for edge in data["edges"]:
            prev_node = self.nodes[edge[0]]
            child_node = self.nodes[edge[1]]
            input_gate = edge[2]
            if edge[2].startswith("__default__"):
                input_gate = (
                    f"{prev_node.node_id}_output" + edge[2][len("__default__") :]
                )
            child_node.add_prev(prev_node, input_gate, edge[3], edge[4])

        self.calc_level()

    def calc_level(self):
        """
        Calc level of the graph backward, from outputs to inputs
        outputs level is 0

        """

        queue = deque([self.nodes["outputs"]])

        while queue:
            node = queue.popleft()
            for prev_tuple in node.prev:
                prev_node = prev_tuple[0]
                prev_node.level = max(prev_node.level, node.level + 1)

                # only traverse again if the level is increased
                if prev_node.level == node.level + 1:
                    queue.append(prev_node)

    def return_by_level(self):
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.level, reverse=True)
        return sorted_nodes[1:]  # don't need inputs node


def json2code(data: dict) -> str:
    class_name = data["class_name"]
    kwargs_str = ", ".join(
        [
            f"{name}: {tuple_[0]} = {tuple_[1]}"
            for name, tuple_ in data["kwargs"].items()
        ]
    )

    dependencies_str = "\n".join(data["dependencies"])

    if len(data["nodes"]["activations"]) > 0:
        dependencies_str += "\nfrom utils import get_activation"
    if len(data["nodes"]["operators"]) > 0:
        dependencies_str += "\nfrom utils import get_operator_function"


    graph = ExecuteGraph(data)

    main_code = f"""{dependencies_str}

class {class_name}(nn.Module):
    def __init__(self, {kwargs_str}):
        super().__init__()
"""

    for node_type, nodes_base in data["nodes"].items():
        main_code += f"\n        # {node_type}\n"
        for node_name, nodes in nodes_base.items():
            for node_id, node_kwargs in nodes.items():
                node_type: Literal["activations", "operators", "networks", "customs"]
                node_name: str
                node = get_node(node_type, node_name)
                main_code += (
                    f"        self.{node_id} = {node.get_var_code(**node_kwargs)}\n"
                )

    main_code += "\n"

    input_kwargs = []

    for name, tuple_ in data["inputs"].items():
        if name.startswith("__default__"):
            name = "input" + name[len("__default__") :]

        if tuple_[1] is None:
            input_kwargs.append(f"{name}: {tuple_[0]}")
        else:
            input_kwargs.append(f"{name}: {tuple_[0]} = {tuple_[1]}")

    initial_inputs_str = ", ".join(input_kwargs)
    main_code += f"    def forward(self, {initial_inputs_str}):\n"

    for node in graph.return_by_level():
        if node.node_name == "outputs":
            return_str = ""
            for prev_tuple in node.prev:
                input_gate = prev_tuple[1]
                return_str += f"{input_gate}, "
            return_str = return_str[:-2]
            main_code += f"        return {return_str}\n"

            break

        input_str = ""
        for prev_tuple in node.prev:
            prev_node, input_gate, input_receive, output_gate = prev_tuple
            if prev_node.node_name != "inputs":
                input_str += f"{input_receive}={input_gate}, "
            else:
                input_str += f"{input_receive}={input_gate}, "
        input_str = input_str[:-2]

        n_outputs = 1
        if node.node_name != "inputs" and node.node_name != "outputs":
            n_outputs = get_node(node.node_type, node.node_name).n_outputs

        if n_outputs == 1:
            main_code += (
                f"        {node.node_id}_output = self.{node.node_id}({input_str})\n"
            )
        else:
            main_code += f"        {', '.join([f'{node.node_id}_output{i}' for i in range(n_outputs)])} = self.{node.node_id}({input_str})\n"

    networks_code = ""
    main_depends_networks = set()
    networks_code_dependencies = set()
    for node_name in data["nodes"]["networks"]:
        node = get_node("networks", node_name)
        networks_code_dependencies.update(node.get_dependencies())
        networks_code += node.get_creation_code()
        main_depends_networks.add(node.class_name)

    networks_code = get_dependencies_str(networks_code_dependencies) + "\n" + networks_code

    if len(main_depends_networks) > 0:
        import_from_networks = "from networks import "
        for network_class_name in main_depends_networks:
            import_from_networks += f"{network_class_name}, "
        import_from_networks = import_from_networks[:-2]
        main_code = import_from_networks + "\n" + main_code

    customs_code = ""
    main_depends_customs = set()
    customs_code_dependencies = set()
    for node_name in data["nodes"]["customs"]:
        node = get_node("customs", node_name)
        customs_code += node.get_creation_code()
        main_depends_customs.add(node.class_name)
        customs_code_dependencies.update(node.get_dependencies())
    customs_code = get_dependencies_str(customs_code_dependencies) + "\n" + customs_code
    if len(main_depends_customs) > 0:
        import_from_customs = "from customs import "
        for custom_class_name in main_depends_customs:
            import_from_customs += f"{custom_class_name}, "
        import_from_customs = import_from_customs[:-2]
        main_code = import_from_customs + "\n" + main_code

    return {
        "main.py": main_code,
        "utils.py": utils_code,
        "networks.py": networks_code,
        "customs.py": customs_code,
    }


def main():
    import os
    os.makedirs("temp", exist_ok=True)
    codes = json2code(data)

    for code_file, code in codes.items():
        with open(f"temp/{code_file}", "w") as f:
            if isinstance(code, str):
                f.write(code)


if __name__ == "__main__":
    main()
