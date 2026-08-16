# Agent instructions

## Research mindset

This is a research repository, not a production service or public library.
Optimize for clear experiments, fast iteration, and code that is easy to
change.

- Prefer the simplest implementation that answers the current question.
- Breaking backward compatibility is fine and encouraged. Update affected code
  and documentation together instead of adding compatibility shims,
  deprecations, or migrations.
- Do not add speculative validation, retries, fallbacks, abstractions, or
  production-grade defensive machinery.
- Keep changes focused and make experimental assumptions explicit.

## Human collaboration

Aggressively ask the human for input on how to build things. Ask early rather
than silently making product, research, or architectural decisions.

- Whenever a change adds complexity, first explain why it is needed, offer the
  simplest viable options, and ask which direction to take.
- Ask before introducing new abstractions, dependencies, workflows, data
  sources, infrastructure, or long-lived interfaces.
- Surface tradeoffs and uncertain assumptions while they are still cheap to
  change.
- Handle obvious, low-impact implementation details autonomously.

## Python

Use `uv` for dependency management and every Python command.

```bash
uv sync
uv run python -m unittest discover -v
```

When dependencies change, update both `pyproject.toml` and `uv.lock`.

## Testing

Keep tests minimal. Add them only for important scientific assumptions,
high-value boundaries, or bugs likely to recur.

- Prefer one focused regression test over a large matrix of cases.
- Do not test implementation details.
- Update or delete tests when an intentional breaking change replaces their
  old contract.
- Documentation-only changes do not require tests.

## Research integrity

- Prevent target leakage and preserve the time boundary between predictors and
  outcomes.
- Keep evaluation chronological and do not tune on held-out results.
- Do not rewrite reported results without a corresponding reproducible run.
- Keep generated data, caches, models, and run outputs out of version control.
- Avoid expensive builds, downloads, training, remote runs, or publishing when
  a small local check is sufficient.

Do not assume the current directory layout or module boundaries are stable.
Discover the repository as it exists before changing it.
