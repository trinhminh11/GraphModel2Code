# FromGraphModel2Code

## What Is This?

A **draw.io-like web application for Deep Learning** — you visually design neural network architectures by dragging, connecting, and configuring nodes on a canvas, and the tool generates **workable PyTorch code** from your graph.

Think of it as a visual IDE for neural networks: sketch the architecture, tweak the parameters, hit export, get real code.

## The Core Loop

```
┌──────────┐       JSON         ┌──────────┐       JSON         ┌──────────┐
│ Frontend │  ───────────────▶  │ Backend  │  ───────────────▶  │ PyTorch  │
│ (Canvas) │  ◀───────────────  │  (API)   │                    │  Code    │
└──────────┘       JSON         └──────────┘                    └──────────┘
```

**JSON is the single source of truth.** Everything revolves around it:

1. **JSON → UI**: The frontend reads a JSON graph definition and renders it as a visual diagram (nodes, edges, positions, colors, sizes).
2. **UI → JSON**: When the user edits the graph on the canvas (moves nodes, connects edges, changes parameters), the frontend serializes the state back to JSON.
3. **JSON → Code**: The backend takes the graph JSON and translates it into a real, runnable PyTorch `nn.Module` — complete with `__init__`, `forward`, and all necessary imports.

## Frontend

A React-based canvas (using React Flow) where users can:

- **Drag & drop** layers from a sidebar palette onto the canvas
- **Connect** nodes by drawing edges between ports
- **Configure** each node's parameters via a properties panel (kwargs, forward args, etc.)
- **Style** nodes with custom colors, borders, sizes — visual clarity matters

The frontend JSON contains everything needed for rendering:

```json
{
  "position": { "x": 100, "y": 200 },
  "width": 180,
  "height": 60,
  "color": "#4A90D9",
  "border": "solid",
  "label": "GeLU"
}
```

## Backend

A FastAPI server that:

- **Stores a database of layer/node definitions** — each node knows its class name, constructor kwargs, forward signature, dependencies, and optionally its full source code.
- **Serves the layer catalog** to the frontend so the palette always reflects available components.
- **Generates PyTorch code** from a graph JSON, resolving the execution order, wiring inputs/outputs, collecting imports, and producing a valid `nn.Module`.

### Node Types

There are two kinds of backend nodes:

- **LibNode** — wraps an existing PyTorch class (e.g., `nn.ReLU`, `nn.Linear`). No custom code needed; just import and instantiate.
- **ClassNode** — defines a custom `nn.Module` with inline source code (e.g., a GELU variant, a gated MLP). Can depend on other ClassNodes.

### Layer Database (what exists and what's planned)

| Category | Status | Examples |
|---|---|---|
| Activations | Done | ReLU, GELU (many variants), Sigmoid, Tanh, SiLU, Mish, Softmax, PReLU, ... |
| Basic Layers | Planned | Linear, Conv1d/2d/3d, Dropout, BatchNorm, LayerNorm, Embedding, ... |
| Recurrent | Planned | RNN, LSTM, GRU |
| Composite Blocks | Planned | MLP, GatedMLP, Residual Block |
| Attention | Planned | MultiHeadAttention, Scaled Dot-Product Attention |
| Pooling | Planned | MaxPool, AvgPool, AdaptiveAvgPool |
| Utilities | Planned | Concat, Split, Reshape, Permute |

### Code Generation Example

Given this graph JSON:

```json
{
  "name": "my_model",
  "class_name": "MyModel",
  "kwargs": {},
  "nodes": {
    "gelu_0": "gelu",
    "tanh": "tanh",
    "dup": "dup",
    "concat": "concat"
  },
  "edges": [
    ["inputs", "gelu_0", "X", "X"],
    ["gelu_0", "tanh", "__default__", "input"],
    ["tanh", "dup", "__default__", "X"],
    ["dup", "concat", "__default__0", "X"],
    ["gelu_0", "concat", "__default__", "Y"],
    ["dup", "outputs", "__default__1", "__default__0"],
    ["concat", "outputs", "__default__", "__default__1"]
  ],
  "inputs": { "X": ["Tensor", null] },
  "dependencies": ["from torch import Tensor"]
}
```

The backend produces:

```python
import torch
import torch.nn as nn
from torch import Tensor

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu_0 = GELUActivation(use_gelu_python=False)
        self.tanh = nn.Tanh()
        self.dup = DupActivation()
        self.concat = ConcatActivation()

    def forward(self, X: Tensor):
        gelu_0_output = self.gelu_0(X=X)
        tanh_output = self.tanh(input=gelu_0_output)
        dup_output0, dup_output1 = self.dup(X=tanh_output)
        concat_output = self.concat(X=dup_output0, Y=gelu_0_output)
        return dup_output1, concat_output
```

The graph is topologically sorted (outputs → inputs), execution order is resolved, multi-output nodes are handled, and all required class definitions and imports are collected automatically.



## Project Status

Early development. The backend activation database and the JSON-to-code pipeline are functional. Next steps are expanding the layer database, building out the REST API, and wiring the frontend canvas to the backend.
