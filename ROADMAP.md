# PatchCraft — Product Roadmap

> **Mission:** Turn PatchCraft into the fastest, most reliable autonomous issue-fixer:
> point it at any real-world issue in a big, messy repository and get a correct,
> well-tested pull request — written like an experienced human engineer would write it.
>
> **North-star metric:** time from `issue URL` → green, reviewable PR, with zero manual intervention.

---

## 1. Who this is for

| User | Need |
|---|---|
| **You (power user)** | Throw real issues from big/messy repos at it; iterate fast; pay as little as possible per solved issue. |
| **Early adopters** | One-command setup, obvious value in < 5 minutes, works on their stack (Python first, JS/TS second). |
| **Maintainers** | A bot that opens *reviewable* PRs that respect repo guidelines and doesn't spam. |

## 2. Guiding principles

1. **Correctness over cleverness** — never merge confidence; always prove with tests.
2. **Cheap iterations, expensive only when needed** — small models for triage, big models only for patch generation.
3. **Respect the repo** — follow its conventions, templates, lint rules, and its humans.
4. **Fail loudly, roll back safely** — the working tree is sacred.
5. **Every release must be demonstrably faster or more accurate** than the previous one (measured, not felt).

## 3. Where we are today (baseline inventory)

Already working:
- LLM access via `litellm` + OpenRouter with DeepSeek → Anthropic → OpenAI fallback and a single `OPENROUTER_API_KEY`.
- Goal-driven self-correction loop (diagnose → patch → test) with loop detection, stagnation guardrails, token/time/credit budgets.
- Complete-file patches applied to disk with snapshot/diff/rollback.
- Sandbox test runner auto-detecting pytest / npm / pnpm / yarn (Windows-safe).
- CLI (`ask`, `run`, `select`, `gui`), Textual TUI, OpenRouter credits widget.

Known gaps blocking real-world adoption:
- Context building walks files alphabetically and caps at 20 files — hopeless for hundreds-of-files repos.
- Patches are **whole-file rewrites** — expensive, truncation-prone, dangerous on large files.
- No git integration: no branches, no commits, no PR creation.
- No evaluation harness: we can't prove we got better.
- Not packaged; no docs; single-user toy.

---

## Milestone 1 — Context Engine: handle big, messy codebases

> **Exit criteria:** PatchCraft solves issues in a 500+ file repository without the user
> ever pointing it at files, and context cost per task drops ≥ 60%.

### Step 1.1 — Repository map & symbol index

**Goal:** Replace the alphabetical file walk with structural understanding of the repo.

**Prompt:**
```
You are extending PatchCraft. Add a repository indexing module `src/core/repo_index.py`
that builds a lightweight "repo map":

1. Walk the target repo once (reuse IGNORED_DIR_PARTS rules from src/orchestrator.py)
   and for every supported source file extract symbols (classes, functions, methods,
   signatures, line ranges) WITHOUT executing code. Pure-Python AST pass for `.py`;
   regex-based fallback extractor for JS/TS.
2. Persist the index at `<repo>/.patchcraft/index.json`, keyed by file hash + mtime so
   subsequent runs only re-index changed files (incremental indexing).
3. Expose `RepoIndex.build(repo_root)`, `RepoIndex.symbols(query)`,
   `RepoIndex.file_summary(path)` returning a compact tree like:
   `src/core/llm.py :: call_llm(...) L120-330 | build_fallback_chain(...) L98-115`
4. Integrate: `build_context()` now prepends this repo map instead of dumping raw
   files alphabetically. Respect MAX_CONTEXT_CHARS.
5. Unit tests on a fixture repo (>=30 files): incremental rebuilds, ignored dirs,
   symbol accuracy. All existing tests stay green.

Constraints: no heavy new dependencies; index build < 2 s warm / < 10 s cold on a
1,000-file repo. All user-facing strings in English.
```

### Step 1.2 — Smart file retrieval (which files matter for THIS issue?)

**Prompt:**
```
Add a retrieval stage to the diagnosis flow.

1. Implement `src/core/retrieval.py` with a two-stage ranker:
   a. Lexical: BM25-style scoring of issue title+body against file contents and
      indexed symbol names (pure Python, no external service).
   b. Structural: boost files connected via imports to high-scoring files using an
      import graph built from RepoIndex.
2. `select_files(issue_text, repo_index, k) -> list[str]` returns ranked relative paths.
3. Orchestrator: diagnosis prompt = issue + repo map + top-k excerpts; validate the
   diagnostic agent's `affected_files` against real paths, fuzzy-matching unknown ones
   via the index before discarding.
4. Env override PATCHCRAFT_RETRIEVAL_K (default 12). Retrieval must add <300 ms on a
   1,000-file repo.
5. Golden-case test where naive alphabetical walk misses the right file but BM25 +
   import boost finds it. English-only strings.
```

### Step 1.3 — Surgical patches (search/replace blocks, not whole files)

**Goal:** Stop rewriting entire files. Patches become small, reviewable, cheap edits.

**Prompt:**
```
Evolve the patch format from whole-file rewrites to surgical edits, backward-compatible.

1. Extend `src/agents/coder.py`: optional search/replace mode
   (`files[].edits: [{find: <exact snippet>, replace: <new snippet>}]`) alongside the
   existing `new_content` (still used for brand-new files). Update SYSTEM_PROMPT /
   CORRECTION_PROMPT to request edits with 2-4 lines of stable surrounding context.
2. Implement `apply_edits()` in the orchestrator: exact match first; one retry with
   whitespace-normalized matching; on repeated failure feed the failed hunk back into
   the self-correction loop as actionable feedback (never silently skip).
3. Guardrails: refuse ambiguous `find` (multiple matches) -> correction loop; keep
   path-traversal protections from _resolve_patch_path.
4. Lower dynamic_patch_budget defaults accordingly; document why.
5. Tests: multi-hunk edit, ambiguity rejection, whitespace-tolerant match, rollback
   still pristine, all legacy patch tests green.

This is the single biggest lever for cost AND correctness on big files — do not rush it.
```

### Step 1.4 — Context budgeter & prompt compiler

**Prompt:**
```
Centralize prompt assembly in `src/core/prompts.py`: a "prompt compiler" taking
(issue, repo_map, retrieved_files, diagnosis, test_feedback, history) and emitting each
agent's final prompts under an explicit token budget (~chars/4 estimate).
Trim priority when over budget: issue text > affected-file current content > repo map >
test feedback tail > everything else. Debug-log what was included/dropped.
Migrate orchestrator + agents to it; delete duplicated assembly logic; English-only
strings; full tests for trimming order.
```

## Milestone 2 — Execution engine v2: fast, targeted, trustworthy testing

> **Exit criteria:** average iteration time drops ≥ 50%; failure feedback is precise
> enough that the corrector usually converges in ≤ 2 attempts.

### Step 2.1 — Targeted test selection

**Prompt:**
```
Implement smart test selection in `src/sandbox/runner.py`:

1. `SandboxRunner.run_tests(targets=None)` accepts focused targets (pytest node ids,
   vitest/jest filters). Each iteration first runs ONLY tests plausibly affected by the
   patched files (map via import graph from Step 1.1 + naming heuristics:
   test_<mod>.py, <mod>_test.py, __tests__/). Success is only declared after ONE full
   confirmation run of the whole suite.
2. Discover targets with each framework's collection command
   (`pytest --collect-only -q`, etc.). Graceful degradation: if collection fails, fall
   back to running the full suite directly (current behavior).
3. Timeouts: keep 30 s default for targeted runs; add `timeout_full_suite`
   (default 300 s), configurable via env and config file.
4. TestResult gains a field describing which subset ran, so loop logs stay transparent.
5. Fixture-project tests proving: failing targeted test short-circuits iteration;
   green targeted run triggers the full suite; timeout kills still work.

Never let targeted runs mark success — the full suite is the gate.
```

### Step 2.2 — Environment hardening & structured failure extraction

**Prompt:**
```
Make sandbox execution fast and reliable across machines:

1. Reuse a single virtualenv/dependency state between iterations instead of touching
   the environment every run; detect missing-dependency errors and support optional
   auto-install behind an explicit --auto-install flag (default off).
2. Windows-first robustness: verify npm/pnpm/yarn cmd-shim handling and POSIX
   equivalents; reliable process-tree kills on timeout (already partially done).
3. Structured failure extraction: parse pytest/vitest/jest output into a normalized
   FailureReport {test_id, assertion, expected, actual, traceback_tail} that feeds the
   corrector prompt instead of raw dumps (raw output stays as fallback). This makes
   correction prompts smaller AND more actionable.
4. Make all runner options readable from .patchcraft.yml (Step 3.3).
5. Unit tests: timeout kills, malformed output parsing, dependency-error detection.
```

---

## Milestone 3 — Speed & UX: zero-friction daily driving

> **Exit criteria:** a new user goes from install to first solved issue in < 5 minutes;
> repeat runs on the same repo start in < 3 seconds.

### Step 3.1 — Caching layer

**Prompt:**
```
Add aggressive-but-safe caching:

1. Repo index cache (from Step 1.1) — already incremental; ensure it is never stale.
2. LLM memo cache: identical (model, messages-hash) completion calls within a single
   task reuse the earlier response (saves money when retries re-send identical prompts).
   Store under .patchcraft/cache/, opt-out via --no-cache. Never cache across different
   OPENROUTER_API_KEY accounts.
3. Test-result cache keyed by patch hash + test subset: if the exact same patch is
   re-proposed, reuse its verdict for the targeted run (never for the final gate).
4. Document cache invalidation rules in docs. Tests for each cache's hit/miss paths.
```

### Step 3.2 — Streaming, live iteration UI

**Prompt:**
```
Upgrade the CLI/TUX feedback loop so users always see what the agent is doing NOW:

1. Stream milestones through the existing event_sink into a rich Live panel:
   current stage, iteration n/∞, tokens spent vs budget, last test verdict, elapsed time.
2. Add a compact `patchcraft status`-style footer line to the TUI mirroring it.
3. Keep output pipe-friendly when not attached to a TTY (plain lines, no ANSI), so CI
   logs stay clean.
4. No behavior changes to the orchestrator itself — presentation only. English strings.
```

### Step 3.3 — Configuration & one-command flows

**Prompt:**
```
Introduce `.patchcraft.yml` at repo root (optional; sensible defaults otherwise):

  model / fallback models, retrieval_k, token_budget, time_budget, min_credits,
  max_retries, test.command override, test.timeout_full_suite, ignore globs,
  commit_style (conventional|repo-derived), pr.draft (bool).

1. Implement a small typed loader (pydantic) with clear validation errors in English;
   unknown keys warn but don't fail.
2. New CLI ergonomics:
   - `patchcraft fix <issue-url-or-number>` resolves GitHub issues directly (merging
     today's select flow) and runs headless.
   - Global --yes flag to skip confirmations; sensible exit codes (0 success,
     1 no-convergence, 2 config error, 3 budget halt) documented for CI use.
3. Update README quickstart to lead with `fix`. All prior commands keep working.
```

## Milestone 4 — Git & Pull Request automation (the "real product" milestone)

> **Exit criteria:** one command takes an issue URL to an opened, well-written draft PR
> on the user's repo; nothing is pushed without explicit intent.

### Step 4.1 — Safe git workflow

**Prompt:**
```
Add first-class git support behind a safety-first design:

1. Before patching: verify the target repo is clean (`git status --porcelain` empty);
   refuse to run otherwise unless --allow-dirty. Create a working branch
   `patchcraft/<issue-number>-<slug>` from the current HEAD.
2. On success (tests green): stage only files PatchCraft touched, write a commit whose
   message follows the repo's own style — detect it from `git log --oneline -n 30`
   (Conventional Commits vs free-form) and mirror it. Reference the issue
   ("Fixes #123") only when solving a GitHub issue.
3. On failure/rollback: delete the branch, restore HEAD exactly as before.
4. Work in an isolated worktree (`git worktree add`) so the user's checkout is never
   disturbed mid-run; clean up worktrees after every exit path (success, halt, crash).
5. Repos without git keep today's behavior (apply in place, rollback on failure).
6. Tests with real temp git repos covering: dirty-repo refusal, branch naming, commit
   content/style detection, rollback cleanliness, non-git fallback.

Safety rule: NEVER push or create PRs inside this step — push comes later, explicitly.
```

### Step 4.2 — GitHub integration & PR creation

**Prompt:**
```
Build `src/github/pr_publisher.py` on top of the existing issue_fetcher HTTP layer
(same token handling: GITHUB_TOKEN env; no new SDK dependency unless justified):

1. After a successful run with a clean branch and commit:
   - resolve the default branch and remote from git config;
   - push the branch to origin under the patchcraft/* namespace;
   - open a Pull Request via POST /repos/{owner}/{repo}/pulls;
   - honor pr.draft from .patchcraft.yml (default true — always draft until M5).
2. Idempotency: if a patchcraft/* PR already exists for this issue (search by header
   marker), update the same PR instead of opening duplicates.
3. CLI: `patchcraft fix ... --push` opts into publishing; default remains local-only.
   Print the PR URL prominently at the end.
4. Handle API errors (401, 404, rate limit, protected branch) with clear English
   messages and actionable hints; never leave a half-pushed state (delete remote
   branch if PR creation fails).
5. Tests against a fake GitHub HTTP server (responses library or mock transport):
   happy path, duplicate-PR update path, error paths, draft flag propagation.
```

### Step 4.3 — Human-grade PR writing (the differentiator)

**Goal:** PR descriptions indistinguishable from a good human maintainer's: correct
tone, structure, scope, and honesty about what was done and why.

**Prompt:**
```
Rebuild the reporter agent into a "PR writer" that produces genuinely human-quality
pull request content:

1. Guideline ingestion: read CONTRIBUTING.md, PULL_REQUEST_TEMPLATE.md,
   ISSUE_TEMPLATE, docs/ conventions, recent merged PR titles/bodies (via GitHub API,
   last ~20), and CODEOWNERS. Compile a concise "repo voice" profile cached alongside
   the repo index. If no templates exist, fall back to a clean generic structure.
2. Content requirements for the generated PR body:
   - Title mirrors the repo's own title conventions (length, prefixes, issue refs).
   - "What changed / Why / How it's tested" sections following the repo's template
     fields exactly (checkboxes included, honestly ticked).
   - Mention trade-offs or follow-ups a senior engineer would flag; do NOT invent
     facts; do not claim reviews or approvals; never fabricate benchmark numbers.
   - Include the diff summary (files + +/- counts) and test evidence (commands +
     result), formatted per repo custom.
3. Tone calibration: match formality level of recent merged PRs (emoji use, bullet
   density, language). Keep it concise — humans skim.
4. The writer receives ONLY verified data (final diff, passing test commands/output,
   issue text). Add a hard rule: any claim not derivable from these inputs is forbidden.
5. Self-review pass: a second LLM call critiques the draft against the repo voice
   profile and template compliance checklist; revise once; then output.
6. Tests: golden fixtures for a Conventional-Commits repo and a quirky no-template
   repo; assert template field coverage, no fabricated content markers present,
   title length within repo norms.

Acceptance: in blind review you cannot tell PatchCraft PRs from human ones on style;
zero hallucinated claims by construction (inputs are the only factual source).
```

### Step 4.4 — Review-response loop (iterate on PR feedback)

**Prompt:**
```
Close the loop with human reviewers:

1. `patchcraft followup <pr-url>` fetches unresolved review comments (GraphQL or REST
   reviews/comments endpoints), classifies them (must-fix, nit, question) and runs the
   existing self-correction loop scoped to must-fix items, pushing updates to the SAME
   branch (no new PRs).
2. Reply to each addressed comment with a short, polite English explanation of what
   changed; tick re-review requests per repo norms.
3. Reuse stagnation/token guardrails; hard cap follow-up iterations via config.
4. Tests: comment classification fixture, push-to-same-branch flow, guardrail halts.
```

## Milestone 5 — Quality, safety & trust (earn the right to be used by others)

> **Exit criteria:** a public eval board shows solve-rate/cost/time per release; PRs
> stop being draft-only once quality is proven.

### Step 5.1 — Evaluation harness

**Prompt:**
```
You cannot improve what you don't measure. Build `eval/`:

1. Task format: JSON per task {repo snapshot (git URL + pinned commit), issue text,
   expected-behavior test}. Seed 20 tasks: 10 from this repo's own history, 10 curated
   small OSS bugs (permissive license), plus an optional adapter for SWE-bench-lite.
2. `python -m eval.run --tasks eval/tasks/*.json --report eval/reports/<date>.json`
   runs PatchCraft end-to-end per task with budgets enforced; records success,
   iterations, tokens, cost estimate (OpenRouter pricing table), wall time.
3. `--compare old.json` prints a delta table and a markdown summary for release notes.
4. Deterministic mode where supported; report flakiness explicitly.
5. CI job (manual trigger) running a fast 5-task smoke subset.
```

### Step 5.2 — Cost & telemetry dashboard

**Prompt:**
```
Make spend and quality visible per run:

1. Extend RunResult with {tokens_by_agent, cost_estimate_usd, wall_time}; render a
   final summary panel and persist to .patchcraft/history/<timestamp>.json.
2. `patchcraft history [--last N]` prints recent runs (task, verdict, tokens, cost,
   duration) with per-repo aggregates.
3. Optional anonymous telemetry (counters only — model names, verdicts, durations;
   never code or prompts), explicit opt-in, documented.
4. Tests for aggregation math and file round-trip. English strings.
```

### Step 5.3 — Safety hardening

**Prompt:**
```
Before strangers rely on this:

1. Secrets hygiene: scan context payloads for high-entropy strings / key formats
   (.env values, PEM blocks) and redact before any LLM call and in all logs.
2. Prompt-injection resistance: treat repository file content as untrusted data —
   neutralize instruction-like patterns ("ignore previous instructions") in retrieved
   files; keep system prompts authoritative. Prove it with adversarial fixtures.
3. Destructive-action guardrails: refuse patches touching .github/workflows, lockfiles
   or dependency manifests unless explicitly allowed in config (supply-chain safety).
4. License awareness: detect the target repo's LICENSE and warn on restrictive cases.
5. Privacy statement in docs: what leaves the machine (prompts to OpenRouter) and what
   never does. Tests for redaction and injection fixtures.
```

## Milestone 6 — Distribution & community (get REAL users)

> **Exit criteria:** `pip install` works everywhere; docs site live; a GitHub Action
> lets third-party repos use PatchCraft on their own issues.

### Step 6.1 — Packaging & install

**Prompt:**
```
Package PatchCraft for one-command install:

1. pyproject.toml with floor-pinned dependencies (litellm, pydantic, rich, click,
   textual, PyYAML), console-script entry point `patchcraft`, Python >=3.10,
   long-description from README.
2. Verify install paths: pipx, uv tool install, plain venv+pip, and Windows PowerShell
   (primary dev platform). Provide a Docker image for CI usage.
3. First-run experience: `patchcraft doctor` checks OPENROUTER_API_KEY, git presence
   and network reachability with a friendly checklist; optional interactive setup.
4. Semver + changelog automation (newsfragments). Publish to TestPyPI then PyPI.
5. CI smoke tests in fresh venvs on windows-latest and ubuntu-latest.
```

### Step 6.2 — Documentation & demo

**Prompt:**
```
Create a docs experience that sells itself:

1. README rewrite leading with a 30-second GIF/asciinema of issue -> green PR.
2. MkDocs Material site: Quickstart; full .patchcraft.yml configuration reference;
   auto-generated CLI reference; architecture diagram (agents/orchestrator/sandbox);
   FAQ (costs, model choices, safety); troubleshooting page.
3. "Recipes" section: Python API repos, monorepos, JS/TS projects.
4. English, concise, example-driven; deploy docs on tag push.
```

### Step 6.3 — GitHub Action & bot mode

**Prompt:**
```
Ship a reusable GitHub Action so OTHER repos can use PatchCraft on their own issues:

1. Action inputs: issue-number, token, openrouter-key, config overrides, draft flag.
   Runs the M4 pipeline headless and opens a draft PR, posting milestone comments
   on the issue.
2. Trigger modes: workflow_dispatch (safe default); optional label trigger
   ("patchcraft:try") gated by an allowlist to prevent abuse and cost attacks.
3. Concurrency controls: cancel superseded runs for the same issue; CI uses stricter
   default budgets than local runs.
4. Least-privilege token guidance (pull-requests write only) and sandboxing notes.
5. Dogfood it on PatchCraft itself — every new issue gets an optional bot attempt.
6. Integration tests against a throwaway repo fixture; document limitations loudly.
```

### Step 6.4 — Feedback flywheel

**Prompt:**
```
Institutionalize learning from real usage:

1. Halted runs print a shareable "failure card" (issue class, halt reason, stage) with
   a command to attach it to a GitHub discussion — never code contents.
2. `--replay <history.json>` reproduces a past run locally to debug user reports
   without needing their code.
3. Maintain a public "known failure classes" doc from failure cards; each minor release
   converts the top recurring class into a guardrail or retrieval improvement.
4. Re-run the eval suite before every release; block release on regression.
```

---

## 4. Suggested order & dependencies

```
M1 (context engine) ──► M2 (execution v2) ──► M3 (speed & UX) ──┐
                                                                 ├──► M5 (eval/safety)
M4 (git/PR automation) ──────────────────────────────────────────┤
                                                                 └──► M6 (distribution)
```
- M1 and M4 are independent and can be developed in parallel tracks.
- M5's eval harness should actually start early (a 5-task smoke suite after M1 helps
  every later decision); the full board lands in M5.

## 5. KPIs to track per release

| KPI | Today | Target |
|---|---|---|
| Issue → green local result (500-file repo) | not feasible | < 10 min |
| Tokens per solved task (median) | whole-file rewrites | −60% via surgical edits |
| Self-correction convergence (≤2 corrective iterations) | unmeasured | ≥ 70% of solved tasks |
| Stagnation halts that were "correct" decisions | n/a | > 90% precision |
| PR body guideline compliance | n/a | 100% template fields |
| Install-to-first-success (new user) | n/a | < 5 minutes |





