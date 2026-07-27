# container/patches/oakhaven — OakHaven (qwen3.8-max) text-causal arch patch

Vendored from the release repo (`runs/OakHaven/patches/`) so the Dynamo image build is
self-contained. Applied by `container/Dockerfile.oakhaven` as an overlay on the vLLM runtime image.

## What breaks without it
OakHaven's checkpoint declares `architectures: ["Qwen3_5MoeForCausalLM"]` /
`model_type: qwen3_5_moe_text` (flat, text-only). vLLM **v0.26.0** registers only the multimodal
`Qwen3_5MoeForConditionalGeneration`, so `vllm serve` fails *"architecture not supported"*. Neither
escape works: remapping to the VLM arch hits `config.vision_config` (absent → `AttributeError`);
a bare registry entry alone loads but leaves `is_hybrid == False`, so the Mamba/GDN `align` +
hybrid-KV path never triggers.

## The fix (12 lines, 2 hunks)
- `registry.py` → register `Qwen3_5MoeForCausalLM` + `Qwen3_5ForCausalLM` in `_TEXT_GENERATION_MODELS`.
- `qwen3_5.py` → add `IsHybrid` to `Qwen3_5ForCausalLMBase` (the class already implements
  `get_mamba_state_{shape,dtype}_from_config`, so the marker suffices).

## Applies to
vLLM **v0.26.0** (base cut at tag `568afb3a13`). `qwen3_5.py` is unchanged v0.26.0→main; `registry.py`
drifted post-release, so these files are cut from the **v0.26.0 base** — matching the
`v0.26.0-ubuntu2404` runtime pinned in `container/context.yaml`. **Inert for qwen3.5-397b/122b**
(they use the native multimodal arch).

## Files
- `qwen3_5-moe-text-causal-arch.patch` — the 12-line diff (primary; `git apply -p1` from site-packages).
- `vllm_v0.26.0/vllm/model_executor/models/{qwen3_5.py,registry.py}` — full patched files (fallback
  overlay if the base drifts). The Dockerfile tries the patch first, falls back to these, then
  self-verifies the `Qwen3_5MoeForCausalLM` marker.

## Upstream story
Not yet upstreamed. This is a permanent maintenance cost until vLLM registers the text-only
`Qwen3_5MoeForCausalLM` arch natively; re-cut against the pinned vLLM tag if `git apply` starts failing.
