from collections import deque
from re import M
from db import activations_dict, register_activation

class GraphNode:
    def __init__(self, name, level=0):
        self.name = name
        self.level = level
        self.prev: list[tuple[GraphNode, str, str]] = []

    def add_prev(self, prev_node: 'GraphNode', input_gate: str, input_receive: str):
        self.prev.append((prev_node, input_gate, input_receive))


    def __repr__(self):
        return f"GraphNode(name={self.name}, level={self.level}, prev={[(prev_tuple[0].name, prev_tuple[1], prev_tuple[2]) for prev_tuple in self.prev]})"


class ExecuteGraph:
    def __init__(self, data: dict):
        self.nodes: dict[str, GraphNode] = {}
        self.nodes["inputs"] = GraphNode("inputs")
        self.nodes.update({node_name: GraphNode(node_name) for node_name in data["nodes"]})
        self.nodes["outputs"] = GraphNode("outputs")

        for edge in data["edges"]:
            prev_node = self.nodes[edge[0]]
            child_node = self.nodes[edge[1]]
            input_gate = edge[2]
            if edge[2].startswith("__default__"):
                input_gate = f"{prev_node.name}_output" + edge[2][len("__default__"):]
            child_node.add_prev(prev_node, input_gate, edge[3])

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
        return sorted_nodes[1:] # don't need inputs node


data = {
    "name": "testing",
    "class_name": "Testing",
    "kwargs": {},
    "nodes": {
        "gelu_0": "gelu",
        "tanh": "tanh",
        "dup": "dup",
        "concat": "concat",
        "gelu_1": "gelu",
    },
    "edges": [
        ("inputs", "gelu_0", "X", "X"),
        ("inputs", "gelu_1", "Y", "X"),
        ("gelu_0", "tanh", "__default__", "input"),
        ("tanh", "dup", "__default__", "X"),
        ("dup", "concat", "__default__0", "X"),
        ("gelu_0", "concat", "__default__", "Y"),
        ("dup", "outputs", "__default__1", "__default__0"),
        ("concat", "outputs", "__default__", "__default__1"),
        ("gelu_1", "outputs", "__default__", "__default__2"),
    ],
    "inputs": {
        "X": ("Tensor", None),
        "Y": (None, None)
    },
    "dependencies": {"from torch import Tensor"}
}

def json2code(data: dict) -> str:
    class_name = data["class_name"]
    kwargs_str = ", ".join([f"{name}: {tuple_[0]} = {tuple_[1]}" for name, tuple_ in data["kwargs"].items()])

    dependencies_str = "\n".join(data["dependencies"])

    code_template = f"""
{dependencies_str}

class {class_name}(nn.Module):
    def __init__(self, {kwargs_str}):
        super().__init__()
"""
    for node_name, node_type in data["nodes"].items():
        node = activations_dict[node_type]
        code_template += f"        self.{node_name} = {node.get_var_code()}\n"

    code_template += "\n"

    initial_inputs_str = ", ".join([f"{name}: {tuple_[0]} = {tuple_[1]}" if tuple_[1] is not None else f"{name}: {tuple_[0]}" for name, tuple_ in data["inputs"].items()])
    code_template += f"    def forward(self, {initial_inputs_str}):\n"

    graph = ExecuteGraph(data)

    for node in graph.return_by_level():
        if node.name == "outputs":
            return_str = ""
            for prev_tuple in node.prev:
                input_gate = prev_tuple[1]
                return_str += f"{input_gate}, "
            return_str = return_str[:-2]
            code_template += f"        return {return_str}\n"

            break

        input_str = ""
        for prev_tuple in node.prev:
            prev_node, input_gate, input_receive = prev_tuple
            if prev_node.name != "inputs":
                input_str += f"{input_receive}={input_gate}, "
            else:
                input_str += f"{input_receive}={input_gate}, "
        input_str = input_str[:-2]

        n_outputs = 1
        if node.name != "inputs" and node.name != "outputs":
            n_outputs = activations_dict[data["nodes"][node.name]].properties.n_outputs

        if n_outputs == 1:
            code_template += f"        {node.name}_output = self.{node.name}({input_str})\n"
        else:
            code_template += f"        {", ".join([f'{node.name}_output{i}' for i in range(n_outputs)])} = self.{node.name}({input_str})\n"

    print(code_template)


def main():
    json2code(data)

if __name__ == "__main__":
    main()

