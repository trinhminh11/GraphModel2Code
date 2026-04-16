"""
Graph-to-PyTorch code generator.

Reads a JSON graph definition (validated as a ``Graph`` schema), builds a DAG
of ``GraphNode`` objects, determines execution order via BFS leveling, and
emits a complete set of Python source files (main.py, utils.py, modules.py,
customs.py) that define a runnable ``nn.Module``.

Example graph topologies are illustrated in the ASCII diagrams below.
"""

from services import CodeGenerator
from schemas import Graph
from services.graph_processor import FileTree, FileNode
from pathlib import Path

raise_flow_check = True

"""test.json
                ---------
                | Input |
                ---------
                    | X-> input
                    v
                ---------
               |nn.Linear|
                ---------
                    | -> X
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

def recursive_write_file(file_tree: FileTree, path: Path):
    for file_or_folder in file_tree.keys():
        if isinstance(file_tree[file_or_folder], FileNode):
            path.mkdir(parents=True, exist_ok=True)
            with open((path / file_or_folder).with_suffix(".py"), "w") as f:
                f.write(file_tree[file_or_folder].file_str)
        else:
            recursive_write_file(file_tree[file_or_folder], path / file_or_folder)


def main():
    """CLI entry point: load a test graph JSON, generate code, and write to temp/."""
    import json
    import os
    import shutil

    shutil.rmtree("temp", ignore_errors=True)

    os.makedirs("temp", exist_ok=True)

    with open("test/test_attn.json", "r") as f:
        data = json.load(f)

    generator = CodeGenerator()
    file_tree = generator.generate(Graph(**data))

    recursive_write_file(file_tree, Path("temp"))



if __name__ == "__main__":
    main()
