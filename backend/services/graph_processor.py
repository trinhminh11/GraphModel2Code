from __future__ import annotations

from typing import Literal

from db.pytorch import get_node, utils_code
from schemas import __ANY__, __REQUIRED__, Graph, ModuleNode
from utils.import_utils import get_dependencies_str

from .log import logger

raise_flow_check = True


class FileNode:
    def __init__(self):
        self.classes: list[ModuleNode] = []

        self.system_dependencies: set[tuple[str, ...]] = set()
        self.third_party_dependencies: set[tuple[str, ...]] = set()
        self.local_dependencies: set[tuple[str, ...]] = set()

        self.code_str = ""

    def add_code_class(self, class_node: ModuleNode):
        self.code_str = class_node.get_creation_code() + "\n" + self.code_str

    @property
    def file_str(self) -> str:
        return f"""# System libraries
{get_dependencies_str(self.system_dependencies)}
# Third party libraries
{get_dependencies_str(self.third_party_dependencies)}
# Local imports
{get_dependencies_str(self.local_dependencies)}

# Code
{self.code_str}
"""

    def add_system_dependencies(self, dependencies: set[str]):
        self.system_dependencies.update(dependencies)

    def add_third_party_dependencies(self, dependencies: set[str]):
        self.third_party_dependencies.update(dependencies)

    def add_local_dependencies(self, dependencies: set[str]):
        self.local_dependencies.update(dependencies)


type FileTree = dict[str, FileNode | FileTree]
type FileTreeDict = dict[str, FileTree]


class GraphNode:
    """A single vertex in the execution DAG.

    Tracks the node's identity (type, id, name), its topological level
    (computed via backward BFS from outputs), its predecessor connections,
    and the names of its output gates.
    """

    def __init__(
        self,
        node_type: Literal[
            "inputs", "outputs", "modules", "activations", "operators", "torch_modules"
        ],
        node_id: str,
        node_name: str,
    ):
        self.node_type: Literal[
            "inputs", "outputs", "modules", "activations", "operators", "torch_modules"
        ] = node_type
        self.node_id = node_id
        self.node_name = node_name
        self.level: int = -1
        self.prev: list[tuple[GraphNode, str, str]] = []
        self.output_gates: tuple[str, ...] = None

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

        self.check_and_calc_level()

    def check_and_calc_level(self):
        """Validate the graph, assign topological levels, and drop unreachable nodes.

        Walks backward from ``outputs`` with DFS. For each node, checks that
        the number of incoming edges matches the node's declared input bounds,
        that wire names reference real output gates, and that there is no cycle.
        Levels increase toward ``inputs`` so ``return_by_level`` can emit
        ``forward`` in the correct order. Any vertex still at ``level == -1``
        after the walk does not reach ``outputs``; those are removed (or
        reported per ``raise_flow_check``).
        """
        # Per-node DFS state for cycle detection: 0 = unseen, 1 = on stack (visiting), 2 = finished.
        state = {}

        def _dfs(node: GraphNode, current_level: int):
            # Real graph nodes must have predecessor count within [n_required_inputs, n_inputs].
            if node.node_type not in ("inputs", "outputs"):
                n_required_inputs = get_node(
                    node.node_type, node.node_name
                ).n_required_inputs
                n_inputs = get_node(node.node_type, node.node_name).n_inputs
                if len(node.prev) < n_required_inputs or len(node.prev) > n_inputs:
                    raise ValueError(
                        f"Node {node.node_id} has {len(node.prev)} inputs but needs to be between {n_required_inputs} (required) and {n_inputs} (total)"
                    )
                n_outputs = get_node(node.node_type, node.node_name).n_outputs
                if len(node.output_gates) != n_outputs:
                    raise ValueError(
                        f"Node {node.node_id} has {len(node.output_gates)} outputs but needs to be equal to {n_outputs}"
                    )

            s = state.get(node.node_id, 0)

            # Re-entering a node that is still on the recursion stack means a back-edge -> cycle.
            if s == 1:
                raise ValueError("Cycle detected in the graph")
            # Furthest distance from outputs along any path to this node (used for codegen order).
            self.nodes[node.node_id].level = max(
                self.nodes[node.node_id].level, current_level
            )
            # Already fully processed this node on another path; skip revisiting predecessors.
            if s == 2:
                return

            state[node.node_id] = 1

            for prev_node, input_gate, _ in node.prev:
                # Each edge must read a tensor from a gate that the predecessor actually exposes.
                if input_gate not in prev_node.output_gates:
                    if raise_flow_check:
                        raise ValueError(
                            f"Input gate {input_gate} does not match output gate {prev_node.output_gates}"
                        )
                    else:
                        logger.warning(
                            f"Input gate {input_gate} does not match output gate {prev_node.output_gates}"
                        )

                _dfs(prev_node, current_level + 1)

            state[node.node_id] = 2

        _dfs(self.nodes["outputs"], 0)

        # Orphans: never reached from outputs, so level stayed at initial -1.
        rm_node_ids = []
        for node in self.nodes.values():
            if node.level == -1:
                if node.node_type == "inputs":
                    msg = "the inputs is not connected to the outputs"
                else:
                    msg = f"the output of node {node.node_id} is not connected to the outputs"

                if raise_flow_check:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

                rm_node_ids.append(node.node_id)

        # Drop disconnected nodes so codegen only sees the subgraph feeding outputs.
        for node_id in rm_node_ids:
            del self.nodes[node_id]

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
    ``forward`` signature and body, and auxiliary module files.
    """

    def generate(self, data: Graph) -> dict[str, FileTreeDict]:
        """Orchestrate full code generation and return a dict of filename -> source code.

        Returns a dict with keys ``'main.py'``, ``'utils.py'``,
        ``'modules.py'``.
        """
        kwargs_str = self._build_init_kwargs(data)
        dependencies_str = self._build_dependencies(data)
        init_body = self._build_init_body(data)
        forward_sig = self._build_forward_signature(data)
        forward_body = self._build_forward_body(data)

        main_code = f"""{dependencies_str}

class {data.class_name}(nn.Module):
    def __init__(self, {kwargs_str}):
        super().__init__()
{init_body}
    def forward(self, {forward_sig}):
{forward_body}"""

        modules_tree, modules_import = self._build_modules_folder(data)
        main_code = modules_import + "\n" + main_code

        return {
            "main": main_code,
            "modules": modules_tree,
            "utils": utils_code,
        }

    def _build_init_kwargs(self, data: Graph) -> str:
        """Build the ``__init__`` parameter signature string from ``data.kwargs``.

        Separates required arguments (no default) from optional ones and
        joins them into a comma-separated string suitable for insertion
        into a ``def __init__(self, ...)`` line.
        """
        required_kwargs_lst = []
        optional_kwargs_lst = []

        for name, (type_, default) in data.kwargs.items():
            if type_ == __ANY__:
                item = f"{name}"
            else:
                item = f"{name}: {type_}"

            if default == __REQUIRED__:
                required_kwargs_lst.append(item)
            else:
                optional_kwargs_lst.append(f"{item} = {default}")

        return ", ".join(required_kwargs_lst + optional_kwargs_lst)

    def _build_dependencies(self, data: Graph) -> str:
        """Build the top-level import block for the generated main module.

        Includes third-party imports from ``data.dependencies`` plus
        conditional ``utils`` imports when the graph uses activations
        or operators.
        """
        dependencies_str = get_dependencies_str(data.dependencies)

        if len(data.nodes.activations) > 0:
            dependencies_str += "\nfrom utils import get_activation"
        if len(data.nodes.operators) > 0:
            dependencies_str += "\nfrom utils import get_operator_function"

        return dependencies_str

    def _build_init_body(self, data: Graph) -> str:
        """Build the submodule instantiation lines inside ``__init__``.

        Iterates over all node categories and generates
        ``self.<node_id> = <NodeClass>(...)`` assignment lines.
        """
        body = ""
        for node_type, nodes_base in data.nodes.to_shallow_dict().items():
            body += f"\n        # {node_type}\n"
            for node_name, nodes in nodes_base.items():
                for node in nodes:
                    body += f"        self.{node.node_id} = {get_node(node_type, node_name).get_assign_code(**node.kwargs)}\n"
        body += "\n"
        return body

    def _build_forward_signature(self, data: Graph) -> str:
        """Build the ``forward()`` method parameter signature from ``data.inputs``.

        Handles ``__default__`` name rewriting, ``__ANY__`` type elision,
        and ``__REQUIRED__`` vs optional defaults.
        """
        input_kwargs = []

        for name, (type_, default) in data.inputs.items():
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

    def _build_forward_body(self, data: Graph) -> str:
        """Build the ``forward()`` method body by traversing nodes in execution order.

        Uses ``ExecuteGraph`` to determine the correct ordering, then
        emits assignment lines for each node and a ``return`` statement
        for the outputs node.
        """
        body = ""
        graph = ExecuteGraph(data)

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
                # if the node is a fan-out node, we need to assign the output gates to the node
                lhs = ", ".join(output_gates[i] for i in range(n_outputs))
                body += f"        {lhs} = self.{node.node_id}({input_str})\n"

        return body

    def _build_modules_folder(self, data: Graph) -> tuple[FileTreeDict, str]:
        """Materialize custom module sources under ``<root>/modules/`` and wire ``__init__.py``.

        Walks every registered ``modules`` node from *data*, builds a nested tree from
        each ``ModuleNode.code_file`` (directories from all but the last segment; the
        last segment is the file stem without ``.py``), and merges all classes that share
        a path into one ``.py`` file.

        For each file, collects imports from ``ModuleNode.get_dependencies(code_root=("modules",))``:
        ``system_lib`` and ``third_party_lib`` become top-of-file imports; ``local_lib`` and
        ``code_dependencies`` become local imports (``FileNode.write`` drops self-imports
        when the dependency path matches the file being written).

        Side effects:
            - Creates/writes ``<root>/modules/**.py`` for each leaf in the tree.
            - Writes ``<root>/modules/__init__.py`` with relative imports (``add_dot=True``)
              and ``__all__`` listing every generated class name.

        Args:
            data: Graph definition whose ``nodes.modules`` keys drive which registry
                ``ModuleNode`` templates are expanded.
            root: Output directory; the ``modules`` package is created at ``root/modules``.

        Returns:
            A single import line to prepend to generated ``main.py``, e.g.
            ``from modules import Foo, Bar``, or an empty string if no module classes
            are referenced from the main graph.
        """
        file_tree: FileTree = {}
        file_tree_dict: FileTreeDict = {}
        main_depends_code: set[str] = set()

        __init__import_code: set[tuple[str, ...]] = set()

        def _recursive_file_tree(node: ModuleNode, main_used=True):
            current_file_tree: FileTree = file_tree
            current_file_tree_dict: FileTreeDict = file_tree_dict
            current_node_dependency = (*node.code_file, node.identifier)
            __init__import_code.add(current_node_dependency)

            if main_used:
                main_depends_code.add(node.class_name)

            for folder_name in node.code_file[:-1]:
                if folder_name not in current_file_tree:
                    current_file_tree[folder_name] = {}
                    file_tree_dict[folder_name] = {}
                current_file_tree = current_file_tree[folder_name]
                current_file_tree_dict = current_file_tree_dict[folder_name]

            file_name = node.code_file[-1].strip()

            if file_name not in current_file_tree:
                current_file_tree[file_name] = FileNode()
            else:
                if not isinstance(current_file_tree[file_name], FileNode):
                    raise ValueError(
                        f"cannot create file {file_name} because it already exists as a folder"
                    )

            dependencies = node.get_dependencies(code_root=("modules",))

            current_file_tree[file_name].add_code_class(node)
            current_file_tree[file_name].add_system_dependencies(
                dependencies["system_lib"]
            )
            current_file_tree[file_name].add_third_party_dependencies(
                dependencies["third_party_lib"]
            )
            current_file_tree[file_name].add_local_dependencies(
                dependencies["local_lib"]
            )


            current_file_tree[file_name].add_local_dependencies(
                (code_dependency for code_dependency in dependencies["code_dependencies"] if code_dependency[1:-1] != node.code_file)
            )

            for _, node_module in node.node_dependencies.items():
                _recursive_file_tree(node_module, main_used=False)

            current_file_tree_dict[file_name] = current_file_tree[file_name].file_str

        for node_name in data.nodes.modules.keys():
            node = get_node("modules", node_name)
            _recursive_file_tree(node, main_used=True)

        __init__file_str = f"""{get_dependencies_str(__init__import_code, add_dot=True)}
__all__ = [
{",\n".join([f'    "{import_class}"' for *_, import_class in __init__import_code])}
]
        """

        file_tree_dict["__init__"] = __init__file_str

        import_line_for_main = ""
        if len(main_depends_code) > 0:
            import_from_code = "from modules import "
            for node_class_name in main_depends_code:
                import_from_code += f"{node_class_name}, "
            import_line_for_main = import_from_code[:-2]

        return file_tree_dict, import_line_for_main
