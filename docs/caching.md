# Caching layer (Roadmap Step 3.1)

PatchCraft ships two independent, disk-backed caches plus an already
incremental repository index. Everything lives under
`<repo>/.patchcraft/cache/` (git-ignored) and can be deleted at any time —
every cache rebuilds or simply misses.

Global kill switch: `PATCHCRAFT_NO_CACHE=1` disables every cache, regardless
of flags. The CLI flag `--no-cache` (commands `run`, `select`, `ask`) does
the same for a single invocation.

## 1. Repository index (`<repo>/.patchcraft/index.json`)

Already incremental since Step 1.1 and **never stale by construction**:
every build re-hashes each source file and only reuses cached symbols when
the content hash matches. A changed, added, renamed or deleted file is
re-indexed automatically; a version bump of the index format triggers a full
rebuild; corrupt files are silently rebuilt.

**Invalidation:** automatic (content-hash keyed). No manual action needed.

## 2. LLM memo cache (`<repo>/.patchcraft/cache/llm/<account-hash>/`)

Identical completion calls reuse the earlier response instead of hitting the
network again — this saves money when retries within one task re-send the
same prompt.

A hit requires ALL of the following to be byte-identical:

* the OpenRouter account (`OPENROUTER_API_KEY` hash is part of the storage
  path — **entries are never shared across accounts**, even on the same
  machine);
* the requested model id;
* the full system prompt and user prompt;
* the response JSON schema (for structured output calls);
* the `max_tokens` output budget.

Rules:

* Responses are stored **only after validation** (e.g. after the JSON
  successfully parses into the expected Pydantic model). Unusable or corrupt
  entries degrade to a normal cache miss.
* Cache hits cost zero tokens: the per-task token accounting
  (`usage_sink`) is NOT incremented for them.
* The cache is enabled for pipeline runs by default; disable with
  `--no-cache`.

**Invalidation:** any change to model, prompts, schema or `max_tokens`
produces a different key. Delete `<repo>/.patchcraft/cache/llm/` (or run
with `--no-cache`) to start fresh.

## 3. Targeted-test verdict cache (`<repo>/.patchcraft/cache/test_results/`)

When the self-correction loop re-proposes the EXACT same patch for the exact
same targeted test subset, the previous verdict is reused without executing
the tests.

The key combines:

* a fingerprint of the post-patch state — the hash of the CURRENT content of
  every file touched by the patch (so surgical edits and whole-file rewrites
  that produce the same result share the same key), and
* the sorted list of targeted pytest node ids.

Rules:

* **Targeted runs only.** The final full-suite gate ALWAYS executes for real
  and is never served from cache.
* Verdicts are stored only when they are properties of the patch itself:
  missing-dependency failures (environment-dependent) and timeouts, exit
  code 124 (machine-load-dependent) are never cached.
* Cached results carry `TestResult.cached = True` so logs stay transparent.

**Invalidation:** any change to the patched content or to the selected test
subset produces a different key. Delete
`<repo>/.patchcraft/cache/test_results/` to clear all verdicts.
