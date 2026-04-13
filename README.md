# FromGraphModel2Code

## Vision

A **drawio-like web application** for visually designing neural network architectures as directed graphs, with the ability to:

- **Export to JSON** — a serializable graph representation of the model
- **Import from JSON** — reload and edit previously designed architectures
- **Export to runnable PyTorch code** — the graph compiles into real, executable Python/PyTorch code

The primary goal is to turn a **visual graph into workable code** that a user can run to instantiate and train their model. Currently targeting PyTorch only, with potential to expand to other frameworks (JAX, TensorFlow) in the future.

---

## Core Concepts

### The Graph

A model is represented as a **directed acyclic graph (DAG)** where:

- **Nodes** are neural network components (layers, activations, operators, custom modules)
- **Edges** define data flow between nodes, carrying tensor(s) from one node's output gates to another node's input gates
- **Input/Output nodes** are special boundary nodes representing the model's `forward()` signature

### Node Types

| Type | Description | Examples |
|------|-------------|----------|
| **Module** | Custom NN modules defined by code templates with `__init__` + `forward`. Can have sub-module dependencies. | MLP, Gated, GatedNet, Attention blocks |
| **LibNode** | Thin wrappers around built-in PyTorch modules (`nn.*`). No custom code — just kwargs for instantiation. | nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Flatten |
| **Activation** | Activation functions, both custom implementations and PyTorch builtins. Resolved at runtime via `get_activation()`. | GELU (7 variants), ReLU, SiLU, Mish, Tanh, Softmax, etc. (20+ total) |
| **Operator** | Stateless math operations between tensors. Resolved via `get_operator_function()`. | +, -, *, /, @ (matmul), T (transpose) |
| **Custom** | System-defined modules that don't fit the other categories. |
| **User** | User-defined modules in the marketplace or export from json. |

### **CURRENT** Key Features of the Node System

- **`kwargs`** — constructor arguments with type, default value, and description. `__required__` sentinel marks mandatory args.
- **`forward_kwargs`** — forward method arguments with the same structure.
- **`#ref/` references** — a node's kwargs can reference the parent graph's kwargs (e.g., `"input_dim": "#ref/inp_dim"` binds to the graph-level `inp_dim` parameter).
- **`__default__` output gates** — auto-named outputs (e.g., `"__default__"` becomes `"{node_id}_output"`).
- **Multi-output nodes** — nodes can produce multiple output tensors (e.g., the `Dup` node outputs 2 copies).
- **`node_dependencies`** — modules can depend on other modules (e.g., `GatedNet` depends on `Gated`), and the code generator recursively resolves and emits all dependent class definitions.
- **`Literal[...]` validation** — kwargs with `Literal` types are validated at schema level to ensure defaults are within the allowed set.

---

## Architecture

### Backend (Python/FastAPI) — *in progress*

```
backend/
├── schemas/            # Pydantic data models
│   ├── base.py         # Sentinels: __REQUIRED__, __ANY__
│   ├── node.py         # NodeBase, ModuleNode, LibNode, ActivationNode, OperatorNode
│   └── graph.py        # Graph, Edge, Nodes, NodeProperties, PrevProperties
├── db/                 # Module database (registry pattern)
│   └── pytorch/
│       ├── activations.py   # 20+ registered activation nodes
│       ├── operators.py     # 6 registered operators
│       ├── custom.py        # User-defined custom nodes
│       ├── torch.py         # PyTorch nn.* library nodes
│       ├── utils.py         # Runtime utils (get_activation, get_operator_function, ACT2CLS, OP2CLS)
│       ├── utils.txt        # Bundled utils source shipped with generated code
│       └── net/
│           ├── common.py    # MLP, Gated, GatedNet with code templates
│           ├── state.py     # Module registry (register_module / get_module)
│           └── attentions/  # (planned) attention mechanism modules
├── utils/
│   └── import_utils.py      # DependencyTree — generates clean Python import statements
├── core/
│   ├── config.py       # (empty — planned)
│   ├── constants.py    # DEFAULT_TAB_SIZE
│   └── security.py     # (empty — planned)
├── api/
│   ├── deps.py         # (empty — planned)
│   └── beta/api.py     # FastAPI router stub
├── test.py             # Core code generator: json2code(), ExecuteGraph, GraphNode
├── test.json           # Test: MLP → Dup → GatedNet + skip → Add → MLP → Tanh
└── test_attn.json      # Test: Self-attention (Q/K/V projections → matmul → softmax → matmul)
```

### Frontend (planned)

- Drawio-like canvas for drag-and-drop node placement
- Node palette / sidebar with searchable module database
- Edge drawing with input/output gate connections
- Properties panel for editing node kwargs
- JSON import/export
- Code generation & preview

### Communication

- Backend and frontend communicate via **JSON over HTTP** (REST API)
- The JSON schema for the API is not finalized yet

---

## Code Generation Pipeline

The `json2code()` function performs these steps:

1. **Parse** the `Graph` JSON into Pydantic models
2. **Build the DAG** — create `GraphNode` objects, link edges
3. **Topological sort** — BFS backward from outputs to assign levels (execution order)
4. **Generate `__init__`** — iterate over all nodes, call `get_var_code()` to produce instantiation lines (`self.x = Module(...)`)
5. **Generate `forward`** — iterate nodes by level (deepest first), wire inputs/outputs using gate names as variable names
6. **Resolve dependencies** — collect all system/third-party/local imports via `DependencyTree`
7. **Emit code files**:
   - `main.py` — the model class with `__init__` and `forward`
   - `utils.py` — bundled activation/operator lookup functions
   - `modules.py` — custom module class definitions (MLP, Gated, etc.)
   - `customs.py` — user-defined custom module class definitions

---

## Module Database

The database uses a **registry pattern**: each module type has a `register_*` function that adds a node definition (with metadata) to an in-memory dictionary. Currently pre-registered:

### Activations (20+)
GELU (original, clipped, fast, new, pytorch_tanh, accurate), Laplace, Linear/Identity, Mish, QuickGELU, ReLU, ReLU6, ReLU², LeakyReLU, PReLU, Sigmoid, SiLU/Swish, Tanh, Softmax

### Operators (6)
Addition (+), Subtraction (-), Multiplication (*), Division (/), Matrix Multiplication (@), Transpose (T)

### Modules (3)
MLP, Gated, GatedNet (with node_dependencies between them)

### Library Nodes (4)
nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Flatten

### Custom (1)
Dup (multi-output test node)

---

## What's Done

- [x] Pydantic schema for nodes, edges, and graphs
- [x] Registry pattern for module database
- [x] 20+ activation functions with both metadata and runtime implementations
- [x] 6 operator nodes
- [x] 3 custom module templates (MLP, Gated, GatedNet) with code generation
- [x] 4 PyTorch library node wrappers
- [x] DAG builder with BFS topological sort
- [x] `json2code()` — full code generation from graph JSON to 4 Python files
- [x] `#ref/` system for binding node kwargs to graph-level parameters
- [x] `__default__` output gate auto-naming
- [x] Multi-output node support
- [x] DependencyTree for generating clean, deduplicated import statements
- [x] Literal type validation on kwargs
- [x] Two working test cases (skip-connection network + self-attention)

## What's Not Done / TODO

- [ ] FastAPI endpoints (currently just an empty router)
- [ ] JSON schema finalization for frontend-backend communication
- [ ] Frontend (drawio-like canvas, node palette, properties panel)
- [ ] More library nodes (nn.Conv1d, nn.BatchNorm, nn.Dropout, nn.Embedding, nn.LSTM, nn.GRU, nn.MaxPool2d, nn.AvgPool2d, etc.)
- [ ] More custom modules (attention mechanisms — `attentions/` directory is empty)
- [ ] User-defined custom module support via the frontend
- [ ] Code preview / validation before export
- [ ] Graph validation (cycle detection, type checking, dimension compatibility)
- [ ] Error handling and user-friendly error messages
- [ ] Sub-graph support (composing a group of nodes into a reusable module)
- [ ] LibNode code generation (`get_var_code` and `get_creation_code` not implemented on `LibNode`)
- [ ] Database persistence (currently everything is in-memory at import time)
- [ ] Multi-framework support (JAX, TensorFlow — future goal)
- [ ] Testing suite

---

## Open Design Decisions

1. **JSON API schema** — the exact shape of request/response payloads between frontend and backend
2. **Frontend framework** — React Flow, Vue, Svelte, or raw Canvas for the graph editor
3. **Custom module UX** — code editor, composable sub-graphs, or both?
4. **Execution scope** — generate code only, or also support in-browser inference / shape validation?
5. **User accounts & sharing** — single-user local tool vs. hosted app with collaboration
6. **Deployment model** — local-only, cloud-hosted, or desktop app
7. **LibNode vs ModuleNode boundary** — should LibNodes eventually merge into ModuleNodes, or remain separate for simplicity?
8. **Graph-level features** — training loops, loss functions, optimizers, data loading as part of the graph?
