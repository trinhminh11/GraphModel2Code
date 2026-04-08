# Artifacts

Shared collaboration space between the human engineer and AI agent.

## Folder Structure

| Folder | Owner | Purpose |
|--------|-------|---------|
| `schema/` | Human | JSON schema drafts, specs, and design notes for the graph data model |
| `ui-spec/` | AI Agent | UI component specs written before each task for human review |
| `layer-registry/` | Human | Layer definition files that the UI palette consumes |
| `examples/` | Both | Sample graph JSON files for testing and demos |
| `decisions/` | Both | Architecture Decision Records (ADRs) for key choices |

## Workflow

1. **Before each task**, the AI agent writes a spec file in `ui-spec/<task-id>-spec.md`
2. The human reviews the spec and approves or requests changes
3. The AI agent implements the feature on a feature branch
4. After testing, the AI agent asks the human to review a PR

For schema design: drop drafts into `schema/`, the AI agent reads them and adapts the UI accordingly.
