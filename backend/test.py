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





def main():
    """CLI entry point: load a test graph JSON, generate code, and write to temp/."""
    import json
    import os

    os.makedirs("temp", exist_ok=True)

    with open("test.json", "r") as f:
        data = json.load(f)

    generator = CodeGenerator()
    generator.generate(Graph(**data), "temp")

if __name__ == "__main__":
    main()
