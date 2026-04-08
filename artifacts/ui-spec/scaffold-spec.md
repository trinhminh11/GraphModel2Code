# Task: scaffold

## What this task produces

### Frontend (`frontend/`)
- Vite + React 18 + TypeScript project
- Dependencies: `@xyflow/react`, `zustand`, `tailwindcss`, `shadcn/ui` primitives, `vitest`, `@testing-library/react`
- Folder structure with empty placeholder files for all component directories
- Vitest config + test setup file
- Basic `App.tsx` shell with placeholder layout (sidebar | canvas | properties)

### Backend (`backend/`)
- Python project managed by `uv`
- `pyproject.toml` with FastAPI, uvicorn, jinja2 as dependencies
- Basic folder structure: `app/`, `app/api/`, `app/codegen/`, `app/models/`
- Placeholder `main.py` with a health check endpoint

### Files created/modified
- `frontend/` — full Vite scaffold + deps
- `backend/` — uv-managed Python project
- `artifacts/ui-spec/scaffold-spec.md` (this file)

## Decisions
- Using Tailwind CSS v4 (latest, CSS-first config)
- shadcn/ui components added incrementally as needed (not bulk-installed)
- Vitest with jsdom for component testing
- uv for Python dependency management
