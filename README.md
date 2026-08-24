# PatchCraft

Point it at any real-world issue in a big, messy repository and get a correct,
well-tested patch — written like an experienced human engineer would write it.

```
issue URL ──► PatchCraft ──► green local result
```

## Quickstart (30 seconds)

```bash
export OPENROUTER_API_KEY=sk-or-...

# One command: fetch the GitHub issue and solve it headlessly.
patchcraft fix https://github.com/owner/repo/issues/123 ./local-repo

# Or with an explicit repository + bare issue number:
patchcraft fix owner/repo 123 ./local-repo
```

PatchCraft then runs its goal-driven loop: **diagnose → patch → test →
self-correct**, showing live progress (stage · iteration · tokens vs budget ·
last test verdict). When tests pass you get the final report and PR-ready diff.

* Take it further: open a **pull request** (opt-in, no side effects by default).

```bash
patchcraft fix --push https://github.com/owner/repo/issues/123 ./local-repo
```

This pushes the `patchcraft/123-...` branch to `origin` and opens (or updates,
idempotently) a **draft** pull request — the body references the issue with a
hidden marker so re-runs never create duplicates. Drafts stay drafts until the
evaluation milestone proves the pipeline reliable; set `pr.draft: false` in
`.patchcraft.yml` to opt out.

### All commands

| Command | Purpose |
|---|---|
| `patchcraft fix [-r OWNER/REPO] ISSUE LOCAL_REPO_PATH` | Fetch one GitHub issue (URL or number) and solve it headlessly. `--push` also opens/updates a draft pull request. **The recommended entry point.** |
| `patchcraft run REPO_PATH "ISSUE TEXT"` | Solve a locally described issue. |
| `patchcraft select GITHUB_REPO LOCAL_REPO_PATH` | Browse open issues interactively and pick one (`--yes` = solve the first automatically). |
| `patchcraft ask "PROMPT"` | One-shot LLM query with automatic fallback. |
| `patchcraft gui` | Interactive TUI (Textual) with credits widget and live log. |

Global flags: `-m/--model MODEL` (overrides config/env), `--yes/-y`
(skip confirmations).

## Exit codes (CI-friendly)

| Code | Meaning |
|---|---|
| `0` | Tests green, pipeline converged. |
| `1` | Pipeline ran but did not converge (tests still failing after retries). |
| `2` | Configuration or input error (bad `.patchcraft.yml`, invalid issue reference, unreachable repo/issue). |
| `3` | Halted by a guardrail: token budget, time budget, credit floor or iteration limit reached. |

Example CI gate:

```bash
patchcraft fix owner/repo "$ISSUE_URL" . || EXIT=$?
case "$EXIT" in
  0) echo "solved" ;;
  1|2|3) echo "not solved (exit $EXIT)"; exit "$EXIT" ;;
esac
```

## Configuration: `.patchcraft.yml` (optional)

Place it at the target repository's root; every field is optional.

```yaml
model: openrouter/deepseek/deepseek-chat     # primary LLM
fallback_models:                             # tried in order after `model`
  - openrouter/anthropic/claude-3.5-sonnet
  - openrouter/openai/gpt-4o
retrieval_k: 12                              # BM25 context width (files)
token_budget: 200000                         # max LLM tokens per task
time_budget: 1800                            # wall-clock seconds per task
min_credits: 1.0                             # OpenRouter credit floor
max_retries: 8                               # patch+test iteration cap
ignore_globs:                                # excluded from indexing/context
  - "vendor/**"
  - "**/*_pb2.py"
commit_style: conventional                   # conventional | repo-derived (git flow)
pr:
  draft: true                                # open PRs as drafts (git flow)
test:
  command: "python -m pytest"                # explicit test command override
  timeout_full_suite: 600                    # full-suite timeout (seconds)
```

Rules:

* **Precedence:** CLI flags > `.patchcraft.yml` > environment variables > defaults.
* Unknown keys produce a warning but never fail the run.
* Validation errors are reported in clear English and exit code `2`.
* The legacy `sandbox:` section (Step 2.2) is still honored for
  `command`/`timeout_full_suite`; prefer the typed `test:` section.

## How it works

* **Context engine** — structural repo index + BM25 retrieval pick the few
  files that matter (no alphabetical dumps).
* **Agents** — diagnostic → coder → self-correcting loop with structured
  failure reports fed back to the model.
* **Safety** — surgical patches applied atomically with snapshot/diff/rollback;
  sandboxed test runner with timeouts and process-tree kills; loop/stagnation
  detection; token/time/credit budgets.
* **Safe git workflow** — in git repositories PatchCraft works in an isolated
  worktree on a `patchcraft/<issue>-<slug>` branch (your checkout is never
  touched), commits only the files it changed with a message matching the
  repo's own commit style (`Fixes #N` when solving an issue), and deletes the
  branch automatically on failure. Dirty checkouts are refused unless you pass
  `--allow-dirty`. Nothing is ever pushed without an explicit `--push`.
* **Caching** — LLM memo cache (never shared across accounts) and targeted-test
  verdict cache; disable with `--no-cache` or `PATCHCRAFT_NO_CACHE=1`.
  Details in [`docs/caching.md`](docs/caching.md).

## Requirements

Python 3.11+, an OpenRouter API key, and Git. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: `GITHUB_TOKEN`/`GH_TOKEN` raises the GitHub API rate limit.
