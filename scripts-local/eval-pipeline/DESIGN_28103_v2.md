# 28103-v2 — kernel-capability model (derived copy emission)

Workspace: `w_e0e8cb3f76d6` ("eval-28103-v2 kernel-capability (impl)"),
branch `v2/kernel-capability` on top of the sealed 28103 before-state
(`55de3a6`). Motivation (user, 2026-09-11): in real vLLM the Q/K copies are a
*consequence* of the RMSNorm kernel being contiguous-only; PR #28103's fix is
the stride-aware kernel, and the copies disappear derivationally. v1 modeled
the consequence as a free flag (`Recipe::COPY_INPUTS`), which lets an agent
"remove the copies" without engaging with the kernel capability at all.

## v2 model

1. **Input layout is a fact of the arch**: Q/K enter qk_norm as strided views
   of the fused-QKV projection output. The worklet records
   `input_layout = StridedQkv` (before AND after — reality doesn't change).
2. **Kernel capability is a declared property of the modeled kernel**:
   `RmsNormKernelCapability { supports_strided: bool }`, initially a recipe
   const (`REQUIRES_CONTIGUOUS_NORM_INPUT`), later bound to distinct profiled
   backends (see phase 2).
3. **Copies are DERIVED, never configured**:
   `emit_copies = input_is_strided && !capability.supports_strided`.
   `QkNormLocalWorkletConfig.copy_inputs` is deleted.
4. **Cheat-proofing assert**: constructing the worklet with strided input, a
   contiguous-only kernel, and no copies is a `BuildError` — "removing the
   copies" without upgrading the kernel capability cannot build. The only
   valid fix is declaring/selecting the stride-aware kernel.

## Phasing (GPU pause constraint)

- **Phase 1 (now, CPU-only)**: implement 1-4 with the existing `flashinfer`
  proxy cost rows serving BOTH capability states (cache identity unchanged —
  capability deliberately NOT added to the profile-db key so the warm
  profile.db stays valid; encode variants by backend NAME when real rows
  exist). Re-validate: before arch → copies derived present, rungs identical
  to v1 before; capability flip → copies gone, rungs identical to v1 after.
- **Phase 2 (one GPU session, when pause lifts)**: extend the 27931-era vLLM
  RMSNorm runner to measure (a) the pre-PR kernel on contiguous input and
  (b) the post-PR stride-aware kernel on strided input, registered as two
  backend names (e.g. `vllm_rmsnorm_contig`, `vllm_rmsnorm_strided`).
  Capability table keys off backend name; the qk_norm cost switches from the
  flashinfer proxy to the real kernels. Re-extract GT rungs, update
  `issues/vllm-28103-v2.yaml`.
- **Phase 3 (eval)**: seed a de-spoiled v2 eval base, dress-rehearse
  (delete-copies-only must FAIL AT BUILD; capability fix must PASS), run k=3.

## Files to touch (phase 1)

- `simulator/src/worklet/qk_norm_local.rs` — config loses `copy_inputs`,
  gains `input_layout` + `norm_capability`; derivation + BuildError assert.
- `simulator/src/arch/qwen3_dense_vllm_common.rs` — plumb layout/capability
  from the recipe.
- `simulator/src/arch/qwen3_dense_vllm_{before,after}.rs` — replace
  `COPY_INPUTS: bool` with `NORM_SUPPORTS_STRIDED: bool` (before=false,
  after=true) + doc comments describing the kernel-capability semantics.
- Tests: before emits derived copies; after doesn't; the forbidden
  combination fails to build.
