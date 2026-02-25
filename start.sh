#!/usr/bin/env bash
# start.sh — launch the PrivaSee frontend and backend together for local development.
#
# Prerequisites:
#   - backend/.env exists and is populated (copy from backend/.env.template)
#   - Python virtualenv created at backend/.venv with requirements installed
#   - Node modules installed in frontend/ (npm install)
#
# Usage:
#   ./start.sh
#
# Press Ctrl+C to stop both processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/backend/.env" ]; then
  echo "ERROR: backend/.env not found."
  echo "Copy backend/.env.template to backend/.env and fill in your credentials."
  exit 1
fi

if [ ! -d "$SCRIPT_DIR/backend/.venv" ]; then
  echo "ERROR: backend/.venv not found."
  echo "Run: cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
  echo "ERROR: frontend/node_modules not found."
  echo "Run: cd frontend && npm install"
  exit 1
fi

# ---------------------------------------------------------------------------
# Start backend
# ---------------------------------------------------------------------------
echo "Starting FastAPI backend on http://localhost:8000 ..."
(
  cd "$SCRIPT_DIR/backend"
  source .venv/bin/activate
  uvicorn app.main:app --reload --port 8000
) &
BACKEND_PID=$!

# ---------------------------------------------------------------------------
# Start frontend
# ---------------------------------------------------------------------------
echo "Starting React dev server on http://localhost:5173 ..."
(
  cd "$SCRIPT_DIR/frontend"
  npm run dev
) &
FRONTEND_PID=$!

# ---------------------------------------------------------------------------
# Wait and handle Ctrl+C
# ---------------------------------------------------------------------------
trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "PrivaSee is running:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."

wait $BACKEND_PID $FRONTEND_PID
