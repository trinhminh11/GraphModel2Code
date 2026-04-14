"""
Graph-to-PyTorch code generator.

Reads a JSON graph definition (validated as a ``Graph`` schema), builds a DAG
of ``GraphNode`` objects, determines execution order via BFS leveling, and
emits a complete set of Python source files (main.py, utils.py, modules.py,
customs.py) that define a runnable ``nn.Module``.

Example graph topologies are illustrated in the ASCII diagrams below.
"""

from collections import deque
from typing import Literal

from db.pytorch import get_node, utils_code
from utils.import_utils import get_dependencies_str
from schemas.graph import Graph
from schemas import __REQUIRED__, __ANY__
from services import logger

raise_flow_check = True

"""test.json
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

"""test_attn.json
                ---------
                | Input |
                ---------
                    |
            --------|--------
         -----    -----    -----
         |MLP|    |MLP|    |MLP|
         -----    -----    -----
            |       |       |
            Q       K       V
            |       |       |
            |       v       |
            |   |--------|  |
            --->| Matmul |  |
                |--------|  |
                    |       |
                    v       |
                |--------|  |
                |Softmax |  |
                |--------|  |
                    |       |
                    v       |
                |--------|  |
                | Matmul |<--
                |--------|
                    |
                    v
                ----------
                | Output |
                ----------
"""


class GraphNode:
    """A single vertex in the execution DAG.

    Tracks the node's identity (type, id, name), its topological level
    (computed via backward BFS from outputs), its predecessor connections,
    and the names of its output gates.
    """

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
        """Register *prev_node* as a predecessor, recording which gate feeds which input slot."""
        self.prev.append((prev_node, input_gate, input_receive))

    def update_output_gate(self, output_gates: tuple[str, ...]):
        """Set the names of the output gates produced by this node."""
        self.output_gates = tuple(output_gates)

    def __repr__(self):
        return f"GraphNode(node_type={self.node_type}, node_id={self.node_id}, node_name={self.node_name}, level={self.level}, prev={[(prev_tuple[0].node_name, *prev_tuple[1:]) for prev_tuple in self.prev]})"


class ExecuteGraph:
    """Builds an executable DAG from a ``Graph`` schema.

    Creates synthetic ``inputs`` / ``outputs`` sentinel nodes, maps every
    real graph node by its id, wires edges, and computes topological levels
    via backward BFS so that nodes can be visited in correct execution order.
    """

    def __init__(self, data: Graph):
        """Parse *data* edges into ``GraphNode`` objects and compute levels."""
        self.nodes: dict[str, GraphNode] = {}

        self.nodes["inputs"] = GraphNode("inputs", "inputs", "inputs")
        self.nodes["inputs"].output_gates = tuple(data.inputs.keys())
        for node_type, nodes_base in data.nodes.to_shallow_dict().items():
            for node_name, nodes in nodes_base.items():
                for node in nodes:
                    self.nodes[node.node_id] = GraphNode(
                        node_type, node.node_id, node_name
                    )
        self.nodes["outputs"] = GraphNode("outputs", "outputs", "outputs")

        for edge in data.edges:
            current_node = self.nodes[edge.node_name]

            for prev_node in edge.prev_nodes:
                current_node.add_prev(
                    self.nodes[prev_node.node_name],
                    prev_node.input_gate,
                    prev_node.input_receive,
                )

            current_node.update_output_gate(edge.output_gates)

        self.check_validity()
        self.calc_level()

    def check_validity(self):
        for node in self.nodes.values():
            if node.node_type in ["inputs", "outputs"]:
                continue

            if (
                len(node.prev) < get_node(node.node_type, node.node_name).n_required_inputs
                or len(node.prev) > get_node(node.node_type, node.node_name).n_inputs
            ):
                raise ValueError(
                    f"Node {node.node_id} has {len(node.prev)} inputs but needs to be between {get_node(node.node_type, node.node_name).n_required_inputs} (required) and {get_node(node.node_type, node.node_name).n_inputs} (total)"
                )

        visited = set()
        in_stack = set()

        def _dfs(node: GraphNode):
            visited.add(node.node_id)
            in_stack.add(node.node_id)

            for prev_node, *_ in node.prev:
                if prev_node.node_id not in visited:
                    if _dfs(prev_node):
                        return True
                elif prev_node.node_id in in_stack:
                    return True

            in_stack.remove(node.node_id)
            return False


        # if all node are connected to the outputs -> one dfs from outputs should visit all nodes
        if _dfs(self.nodes["outputs"]):
            raise ValueError("Cycle detected in the graph")

        for node in self.nodes.values():
            # if a node is not visited, it means it's not connected to the outputs
            if node.node_id not in visited:
                if node.node_type == "inputs":
                    msg = "the inputs is not connected to the outputs"
                else:
                    msg = f"the output of node {node.node_id} is not connected to the outputs"

                if raise_flow_check:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

                # if a node is not visited, we need to dfs from it to check if there is a cycle
                if _dfs(node):
                    raise ValueError("Cycle detected in the graph")

    def calc_level(self):
        """Assign topological levels via backward BFS starting from outputs (level 0).

        Each predecessor's level is set to at least one more than its
        successor's, ensuring that higher-level nodes are executed first.
        """

        queue = deque([self.nodes["outputs"]])

        while queue:
            node = queue.popleft()
            for prev_tuple in node.prev:
                prev_node = prev_tuple[0]
                input_gate = prev_tuple[1]

                if input_gate not in prev_node.output_gates:
                    if raise_flow_check:
                        raise ValueError(
                            f"Input gate {input_gate} does not match output gate {prev_node.output_gates}"
                        )
                    else:
                        logger.warning(
                            f"Input gate {input_gate} does not match output gate {prev_node.output_gates}"
                        )

                prev_node.level = max(prev_node.level, node.level + 1)

                if prev_node.level == node.level + 1:
                    queue.append(prev_node)

    def return_by_level(self):
        """Return all nodes (except the synthetic ``inputs`` node) sorted by descending level.

        This gives the correct execution order: highest-level (earliest)
        nodes first, down to the ``outputs`` node last.
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.level, reverse=True)
        return sorted_nodes[1:]


class CodeGenerator:
    """Generates PyTorch ``nn.Module`` source files from a ``Graph`` definition.

    Decomposes the code generation pipeline into discrete steps:
    building the constructor signature, import statements, ``__init__`` body,
    ``forward`` signature and body, and auxiliary module/custom files.
    """

    def __init__(self, data: Graph):
        """Initialize the generator with a validated Graph."""
        self.data = data

    def generate(self) -> dict[str, str]:
        """Orchestrate full code generation and return a dict of filename -> source code.

        Returns a dict with keys ``'main.py'``, ``'utils.py'``,
        ``'modules.py'``, and ``'customs.py'``.
        """
        kwargs_str = self._build_init_kwargs()
        dependencies_str = self._build_dependencies()
        init_body = self._build_init_body()
        forward_sig = self._build_forward_signature()
        forward_body = self._build_forward_body()

        main_code = f"""{dependencies_str}

class {self.data.class_name}(nn.Module):
    def __init__(self, {kwargs_str}):
        super().__init__()
{init_body}
    def forward(self, {forward_sig}):
{forward_body}"""

        modules_code, modules_import = self._build_auxiliary_file("modules")
        main_code = modules_import + "\n" + main_code

        customs_code, customs_import = self._build_auxiliary_file("customs")
        main_code = customs_import + "\n" + main_code

        return {
            "main.py": main_code,
            "utils.py": utils_code,
            "modules.py": modules_code,
            "customs.py": customs_code,
        }

    def _build_init_kwargs(self) -> str:
        """Build the ``__init__`` parameter signature string from ``data.kwargs``.

        Separates required arguments (no default) from optional ones and
        joins them into a comma-separated string suitable for insertion
        into a ``def __init__(self, ...)`` line.
        """
        required_kwargs_lst = []
        optional_kwargs_lst = []

        for name, (type_, default) in self.data.kwargs.items():
            if type_ == __ANY__:
                item = f"{name}"
            else:
                item = f"{name}: {type_}"

            if default == __REQUIRED__:
                required_kwargs_lst.append(item)
            else:
                optional_kwargs_lst.append(f"{item} = {default}")

        return ", ".join(required_kwargs_lst + optional_kwargs_lst)

    def _build_dependencies(self) -> str:
        """Build the top-level import block for the generated main module.

        Includes third-party imports from ``data.dependencies`` plus
        conditional ``utils`` imports when the graph uses activations
        or operators.
        """
        dependencies_str = get_dependencies_str(self.data.dependencies)

        if len(self.data.nodes.activations) > 0:
            dependencies_str += "\nfrom utils import get_activation"
        if len(self.data.nodes.operators) > 0:
            dependencies_str += "\nfrom utils import get_operator_function"

        return dependencies_str

    def _build_init_body(self) -> str:
        """Build the submodule instantiation lines inside ``__init__``.

        Iterates over all node categories and generates
        ``self.<node_id> = <NodeClass>(...)`` assignment lines.
        """
        body = ""
        for node_type, nodes_base in self.data.nodes.to_shallow_dict().items():
            body += f"\n        # {node_type}\n"
            for node_name, nodes in nodes_base.items():
                for node in nodes:
                    body += f"        self.{node.node_id} = {get_node(node_type, node_name).get_var_code(**node.kwargs)}\n"
        body += "\n"
        return body

    def _build_forward_signature(self) -> str:
        """Build the ``forward()`` method parameter signature from ``data.inputs``.

        Handles ``__default__`` name rewriting, ``__ANY__`` type elision,
        and ``__REQUIRED__`` vs optional defaults.
        """
        input_kwargs = []

        for name, (type_, default) in self.data.inputs.items():
            if name.startswith("__default__"):
                name = "input" + name[len("__default__") :]

            if type_ == __ANY__:
                inp_str = f"{name}"
            else:
                inp_str = f"{name}: {type_}"

            if default == __REQUIRED__:
                input_kwargs.append(inp_str)
            else:
                input_kwargs.append(f"{inp_str} = {default}")

        return ", ".join(input_kwargs)

    def _build_forward_body(self) -> str:
        """Build the ``forward()`` method body by traversing nodes in execution order.

        Uses ``ExecuteGraph`` to determine the correct ordering, then
        emits assignment lines for each node and a ``return`` statement
        for the outputs node.
        """
        body = ""
        graph = ExecuteGraph(self.data)

        for node in graph.return_by_level():
            if node.node_name == "outputs":
                return_parts = [prev_tuple[1] for prev_tuple in node.prev]
                return_str = ", ".join(return_parts)
                body += f"        return {return_str}\n"
                break

            input_parts = []
            for _, input_gate, input_receive in node.prev:
                input_parts.append(f"{input_receive}={input_gate}")
            input_str = ", ".join(input_parts)

            output_gates = node.output_gates
            n_outputs = len(output_gates)
            if n_outputs == 1:
                body += (
                    f"        {output_gates[0]} = self.{node.node_id}({input_str})\n"
                )
            else:
                lhs = ", ".join(output_gates[i] for i in range(n_outputs))
                body += f"        {lhs} = self.{node.node_id}({input_str})\n"

        return body

    def _build_auxiliary_file(
        self, node_type: Literal["customs", "modules"]
    ) -> tuple[str, str]:
        """Generate a supporting source file (modules.py or customs.py) and its import line for main.py.

        Aggregates all node class definitions and their dependencies for
        the given *node_type*, formats the file with import blocks, and
        builds a ``from <node_type> import ...`` line for the main module.

        Returns:
            A tuple of (file_source_code, import_line_for_main).
        """
        code = ""
        main_depends_code = set()
        code_system_dependencies = set()
        code_third_party_dependencies = set()
        code_local_dependencies = set()

        for node_name in self.data.nodes.model_dump()[node_type]:
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

        import_line_for_main = ""
        if len(main_depends_code) > 0:
            import_from_code = f"from {node_type} import "
            for node_class_name in main_depends_code:
                import_from_code += f"{node_class_name}, "
            import_from_code = import_from_code[:-2]
            import_line_for_main = import_from_code

        return code, import_line_for_main


def main():
    """CLI entry point: load a test graph JSON, generate code, and write to temp/."""
    import json
    import os

    os.makedirs("temp", exist_ok=True)

    with open("test.json", "r") as f:
        data = json.load(f)

    generator = CodeGenerator(Graph(**data))
    codes = generator.generate()

    for code_file, code in codes.items():
        with open(f"temp/{code_file}", "w") as f:
            if isinstance(code, str):
                f.write(code)


if __name__ == "__main__":
    main()
