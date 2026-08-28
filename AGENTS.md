# AGENTS.md — Agent Workflow Rules for SparkDeck

## Project Overview

SparkDeck is a Python backend (`manager.py`, `cluster.py`, `disk_manager.py`,
`mcp_server.py`, and the `sparkdeck/` package) with a Node frontend in
`frontend/`. Tests live in `tests/` (pytest). Frontend tests live in
`frontend/` (including `frontend/e2e/`).

## 1. Always Work in an Isolated Worktree

For **every** new change or feature request, do NOT edit the main checkout.
Create a dedicated git worktree and branch first:

```bash
git worktree add ../sparkdeck-<short-name> -b <type>/<short-description> main
```

- `<type>` is one of `feat`, `fix`, `chore`, `refactor`, `docs`.
- Do all edits, builds, and tests inside the worktree directory only.
- Never commit directly to `main`.

## 2. Open a Pull Request After Every Change

When the change is complete and verified (relevant tests pass):

```bash
git push -u origin <branch>
gh pr create --base main --title "<concise title>" --body "<what changed and why>"
```

The PR description must summarize what changed. One PR per requested
change/feature — do not batch unrelated work into a single PR.

## 3. Wait for Codex Review and Fix All Comments

After the PR is opened, wait for Codex to review it:

- Poll for review comments, e.g.:
  - `gh pr view <pr-number> --comments`
  - `gh api repos/{owner}/{repo}/pulls/<pr-number>/comments`
- If Codex leaves review comments, address **every** comment: make the fixes
  in the worktree, commit, push, and continue waiting.
- Repeat this loop until there are no unresolved Codex comments.

## 4. Merge Only After the 👍 Approval

The waiting ends only when the PR's **main post** (the issue body) has a
thumbs-up (`+1`) reaction. Check with:

```bash
gh api repos/{owner}/{repo}/issues/<pr-number>/reactions
```

Merge **only** when both are true:

1. No unresolved Codex review comments remain.
2. The PR body has a `+1` (👍) reaction.

Then merge and clean up:

```bash
gh pr merge <pr-number> --merge
git worktree remove ../sparkdeck-<short-name>
git branch -d <branch>
```

## Safety Rules

- Never merge before the 👍 reaction appears on the PR body.
- Never push or commit directly to `main`.
- Keep each change minimal and scoped to the request — no unrelated refactors.
- Run the relevant tests before opening the PR.
