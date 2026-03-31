Review what changed in this session and update user-facing documentation files as needed.

## Step 1 — Understand what changed

Run `git diff HEAD` and `git diff --cached` to see all changes. Focus on: new files, deleted files, renamed files, new endpoints, changed env vars, changed folder structure, new deployment steps.

## Step 2 — Decide which files need updating

| File | Update when |
|---|---|
| `README.md` | Repo structure changes, new features visible to users, quick-start steps change |
| `backend/tests/README.md` | Test structure changes, new test levels, new fixtures, new make targets, known limitations change |
| `databricks/README.md` | New model files, changed pipelines, new/removed env vars, changed MLflow input/output schemas, new deployment steps |
| `docs/architecture.md` | Component relationships or data flow changes |
| `docs/setup.md` | Prerequisites or local dev setup steps change |
| `docs/deployment.md` | CI/CD pipeline or deploy process changes |

Only update files that are actually affected — don't touch files for cosmetic reasons.

## Step 3 — Update each affected file

Keep the existing style and structure. Be specific and accurate — these are read by developers setting up and operating the system. Do not add placeholder text or speculative content.
