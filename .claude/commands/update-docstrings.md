Review changed files in this session and update inline docstrings to reflect current behaviour.

## Step 1 — Find changed files

Run `git diff HEAD --name-only` and `git diff --cached --name-only` to get the list of modified files. Focus on `.py` files only.

## Step 2 — For each changed Python file

Read the file and check every function or class that was added or modified. For each:

- If it has a docstring: verify it still accurately describes the current behaviour, parameters, and return value. Update if stale.
- If it has no docstring and the logic is non-obvious: add a concise docstring.
- If it has no docstring and the logic is self-evident from the name and signature: leave it — don't add noise.

## Guidelines

- Describe **what** and **why**, not **how** (the code shows how)
- Keep docstrings short — one line is fine for simple functions
- For complex functions, document: what it does, key parameters, return value, any non-obvious side effects or exceptions
- Do not add type annotations, comments, or docstrings to code you didn't change
- Match the existing docstring style in the file
