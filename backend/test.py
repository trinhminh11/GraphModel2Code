from collections import deque
from typing import Literal

from db.pytorch import get_node, utils_code
from utils.import_utils import get_dependencies_str

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


class GraphNode:
    def __init__(self, node_type, node_id, node_name, level=0):
        self.node_id = node_id
        self.node_type = node_type
        self.node_name = node_name
        self.level = level
        self.prev: list[tuple[GraphNode, str, str]] = []
        self.output_gates = None

    def add_prev(
        self,
        prev_node: "GraphNode",
        input_gate: str,
        input_receive: str,
    ):
        self.prev.append((prev_node, input_gate, input_receive))

    def update_output_gate(self, output_gates: tuple[str, ...]):
        self.output_gates = tuple(output_gates)

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

            child_node.add_prev(prev_node, input_gate, edge[3])

            output_gates: tuple[str, ...] = edge[4]

            output_gates_list: list[str] = []

            if len(edge[4]) == 1:
                if output_gates[0].startswith("__default__"):
                    output_gates_list.append(f"{child_node.node_id}_output")
                else:
                    output_gates_list.append(output_gates[0])
            else:
                for output_gate in output_gates:
                    if output_gate.startswith("__default__"):
                        output_gates_list.append(
                            f"{child_node.node_id}_output"
                            + output_gate[len("__default__") :]
                        )
                    else:
                        output_gates_list.append(output_gate)

            child_node.update_output_gate(tuple(output_gates_list))

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


def add_code(node_type: Literal["customs", "networks"], data: dict) -> str:
    code = ""
    main_depends_code = set()
    code_system_dependencies = set()
    code_third_party_dependencies = set()
    code_local_dependencies = set()

    print(node_type)
    print(data["nodes"][node_type])

    for node_name in data["nodes"][node_type]:
        node = get_node(node_type, node_name)
        node_dependencies = node.get_dependencies()

        code_system_dependencies.update(node_dependencies["system_lib"])
        code_third_party_dependencies.update(node_dependencies["third_party_lib"])
        code_local_dependencies.update(node_dependencies["local_lib"])
        code += node.get_creation_code()
        main_depends_code.add(node.class_name)

    code = f"""# System libraries
{get_dependencies_str(code_system_dependencies)}
# Third party libraries
{get_dependencies_str(code_third_party_dependencies)}
# Local imports
{get_dependencies_str(code_local_dependencies)}
{code}
"""

    additional_to_main = ""
    if len(main_depends_code) > 0:
        import_from_code = f"from {node_type} import "
        for node_class_name in main_depends_code:
            import_from_code += f"{node_class_name}, "
        import_from_code = import_from_code[:-2]
        additional_to_main = import_from_code

    return code, additional_to_main


def json2code(data: dict) -> str:
    class_name = data["class_name"]
    kwargs_str = ", ".join(
        [
            f"{name}: {tuple_[0]} = {tuple_[1]}"
            for name, tuple_ in data["kwargs"].items()
        ]
    )

    dependencies_str = get_dependencies_str(data["dependencies"])

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
            prev_node, input_gate, input_receive = prev_tuple
            if prev_node.node_name != "inputs":
                input_str += f"{input_receive}={input_gate}, "
            else:
                input_str += f"{input_receive}={input_gate}, "
        input_str = input_str[:-2]

        output_gates = node.output_gates

        n_outputs = len(output_gates)
        if n_outputs == 1:
            main_code += (
                f"        {output_gates[0]} = self.{node.node_id}({input_str})\n"
            )
        else:
            main_code += f"        {', '.join([f'{output_gates[i]}' for i in range(n_outputs)])} = self.{node.node_id}({input_str})\n"

    networks_code, add_to_main = add_code("networks", data)
    main_code = add_to_main + "\n" + main_code

    customs_code, add_to_main = add_code("customs", data)
    main_code = add_to_main + "\n" + main_code

    return {
        "main.py": main_code,
        "utils.py": utils_code,
        "networks.py": networks_code,
        "customs.py": customs_code,
    }


def main():
    import json
    import os

    os.makedirs("temp", exist_ok=True)

    with open("test_attn.json", "r") as f:
        data = json.load(f)

    codes = json2code(data)

    for code_file, code in codes.items():
        with open(f"temp/{code_file}", "w") as f:
            if isinstance(code, str):
                f.write(code)


if __name__ == "__main__":
    main()
