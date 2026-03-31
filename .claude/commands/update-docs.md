Review what changed in this session and update the three project documentation files according to their defined boundaries.

## Step 1 — Understand what changed

Run `git diff HEAD` and `git diff --cached` to see all staged and unstaged changes. Also check `git log --oneline -5` for recent commits if helpful.

## Step 2 — Update CLAUDE.md (if needed)

`CLAUDE.md` owns: stable architecture decisions, conventions, constraints, test/deploy instructions.

Update it if any of the following changed:
- New or modified architectural patterns (e.g. new service, new data format)
- New conventions or rules developers should follow
- New known constraints or platform limitations
- Changes to how tests are run or deployed

Do NOT add current feature status, bug reports, or implementation state — those belong elsewhere.

## Step 3 — Update TODO.md (if needed)

`TODO.md` owns: open bugs and planned features only.

- Mark completed items with ✅ and a note of what branch fixed them
- Add newly discovered bugs with: area, description, expected behaviour, open questions
- Add newly planned features with enough context to pick up later
- Remove items that are no longer relevant

Do NOT add architecture docs or conventions to TODO.md.

## Step 4 — Update MEMORY.md

`MEMORY.md` (in `.claude/projects/.../memory/`) owns: current implementation state that isn't derivable from CLAUDE.md or TODO.md.

Update `project_state.md` to reflect:
- Current branch
- Current backend test count (run `cd backend && python -m pytest --collect-only -q 2>/dev/null | tail -3` to get it)
- What was completed or changed this session
- Any in-progress work

Do NOT duplicate anything already in CLAUDE.md or TODO.md.
