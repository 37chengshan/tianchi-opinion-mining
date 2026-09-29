# AI Bridge

Shared planning context for ChatGPT, other planning models, Codex, OpenCode, Pi, or another local implementation agent.

- current-plan.md: plan produced by ChatGPT or another planning model for the implementation agent.
- agent-status.md: generic implementation notes, touched files, test results, blockers, and review notes.
- implementation-diff.patch: final review diff from the implementation agent when practical.
- codex-status.md: legacy Codex-specific status file, kept for existing workflows.
- decisions.md: architectural decisions that should remain stable.
- open-questions.md: unresolved questions.
- execution-log.jsonl: append-only generic agent handoff and execution events.
- handoff-run-state.json: machine-readable run lifecycle (running/completed/failed/timed_out) written by execute-handoff/watch-handoff/loop-handoff and polled by the read-only wait_for_handoff tool.
- session-log.jsonl: append-only legacy session events.
