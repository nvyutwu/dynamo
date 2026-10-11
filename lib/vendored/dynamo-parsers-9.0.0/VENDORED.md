# Vendored `dynamo-parsers` 9.0.0

This directory is the crates.io `dynamo-parsers-9.0.0.crate` (sha256
`e316d4d0f5082257f1629ca61973eafc17cee3e6ee0c24f7a55a61beeff40d97`, the checksum this branch's
`Cargo.lock` pinned; `.cargo_vcs_info.json` points at ai-dynamo/frontend-crates `18327ab`, tag
`dynamo-parsers-v9.0.0`) extracted unchanged, plus two patches:

- ai-dynamo/frontend-crates#353 (`321fd9be`, "fix(kimi-k3): generate unique tool-call IDs across
  turns"): the Kimi K3 XTML parser returns `call-<UUIDv4>` instead of `<tool>:<index - 1>`. XTML
  indices restart on every response, so `Bash:0` repeated across turns and clients that key tool
  results by ID (Claude Code) re-ran calls or ended with `[Tool use interrupted]`.
- "fix(kimi-k3): drop closing markup cut by max_tokens" (not yet upstream): when `max_tokens`
  stops K3 inside its closing XTML, the jail no longer flushes the held partial boundary (e.g.
  `<|close|>response`) as content at end of stream, and the parser strips an unfinished trailing
  boundary before extracting response text. A suffix is dropped only when it starts with a whole
  `<|open|>`/`<|close|>` token; sub-token text such as `<|clo` is still released.

Both workspaces select it through `[patch.crates-io]` (root `Cargo.toml` and
`lib/bindings/python/Cargo.toml`). Remove the directory and both patch entries once a released
`dynamo-parsers` containing both fixes is pinned instead.

Verify the vendored bytes: extract the `.crate` and `diff -r` it against this directory; the only
differences are this file and the files the two patches touch (`reasoning/mod.rs`,
`tool_calling/jail/mod.rs`, `tool_calling/xtml/kimi_k3_parser.rs`, `tool_calling/xtml/mod.rs`).
