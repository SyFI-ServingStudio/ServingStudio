# Driver extension spec — MIXED (prefill chunk + decode) and TARGET_VERIFY steps for one Kimi-K3 layer

*Follow-up #4 (user 2026-09-27). Read-only survey of sglang v0.5.20 (`iter_opt_eval_k3_mla/pristine_tree/srt` = `R`) and
the driver `kimi_single_layer_decode.py` (= `D`), produced 2026-09-27 16:45 before editing the driver. Line numbers are
for the pristine tree.*

## 1. ForwardMode and predicates (`R/model_executor/forward_batch_info.py`)
- Enum (107-131): `EXTEND`=110, `DECODE`=112, `MIXED`=114, `IDLE`=116, `TARGET_VERIFY`=119, `DRAFT_EXTEND_V2`=121,
  `PREBUILT`=125, `SPLIT_PREFILL`=128, `DLLM_EXTEND`=131.
- `is_extend` (136-144) is True for EXTEND, MIXED, TARGET_VERIFY, SPLIT_PREFILL, DLLM_EXTEND; `is_mixed` 160;
  `is_target_verify` 169; `is_extend_or_draft_extend_or_mixed` 176-182 (excludes TARGET_VERIFY); `is_cuda_graph` 184-190 =
  DECODE|TARGET_VERIFY|IDLE|DLLM_EXTEND (MIXED/EXTEND never graph-captured); `is_extend_without_speculative` 198-199.
- **The eager runner rewrites MIXED to EXTEND before planning/forward** (`R/model_executor/runner/eager_runner.py:216-222`)
  and re-plans in `_execute_extend` (290-298). Backends and the K3 model only see EXTEND for a mixed batch.

## 2. Code paths
**KDA (both modes):** `KimiK3DeltaAttention.forward` `R/models/kimi_k3.py:1992-2049` — non-decode: `a` is `[1,T,HV,K]`;
`beta.float().sigmoid()` ONLY when not TARGET_VERIFY (2013-2016); `b = beta.unsqueeze(0)`; fused o_norm handoff for
decode/TARGET_VERIFY (2025-2031) → `RadixLinearAttention.forward` `R/layers/radix_linear_attention.py:80-141` (trim by
`global_num_token_non_padded_cpu` only for non-verify extend, 116-123: leave None) → `HybridLinearAttnBackend.forward`
`hybrid_linear_attn_backend.py:1292-1338` → `KDAAttnBackend.forward_extend` `R/layers/attention/linear/kda_backend.py:795-953`,
forks to `_forward_target_verify` at 806-807. The driver's `KDADecodeShim` (`D:1069-1081`) reproduces the dispatch.
- MIXED/EXTEND body: 818-821 (prefix None → RuntimeError), `has_initial_state = extend_prefix_lens > 0` (822), trims to
  `query_start_loc[-1]` (824-829), `causal_conv1d_fn(..., cache_indices, query_start_loc, seq_lens_cpu=extend_seq_lens_cpu)`
  (843-853), `kernel_dispatcher.extend(... lower_bound=layer.lower_bound, extend_seq_lens_cpu=...)` (893-919) →
  `TritonKDAKernel.extend`/`chunk_kda` (`linear/kernels/kda_triton.py:219-254`). Decode tails = 1-token rows with
  `has_initial_state=True`.
- TARGET_VERIFY body (955-1203): reads `fm.query_start_loc/mamba_cache_indices/retrieve_*` (970-976), layer cache `conv[0]`,
  `temporal`, `intermediate_ssm` (981), `intermediate_conv_window[0]` (1013), `self.verify_intermediate_state_indices` (1014),
  `spec_info.draft_token_num` (1016), `spec_info.ragged_verify_layout` (1017). Dense path: `batch_size = seq_len // draft_token_num`
  (1050); fused chain verify only with env `SGLANG_OPT_FUSED_KDA_VERIFY` + triton verify (438-448) and
  `_can_run_fused_chain_verify` (1205-1378: retrieve_* all None, D≥3, fp32 `ssm_states`/`intermediate_ssm`, int32 index tensors,
  conv_states `[N,3,dim]`, window ndim 4). Otherwise: `causal_conv1d_update(dense[bs,dim,D], conv_states.T, ...,
  conv_state_indices=cache_indices[:bs], intermediate_conv_window, intermediate_state_indices[:bs], retrieve_*)` (1130-1145;
  kernel `kernels/ops/mamba/causal_conv1d_triton.py:1000-1014`), then `kernel_dispatcher.target_verify` (1164-1196) →
  `lower_bound` only accepted with `TritonKDAKernel` (307-313) → `TritonKDAKernel.target_verify` (kda_triton.py:159-217) →
  `fused_sigmoid_gating_delta_rule_update(initial_state_source=ssm_states, initial_state_indices=cache_indices,
  cu_seqlens=query_start_loc, is_kda=True, disable_state_update=True, intermediate_states_buffer, intermediate_state_indices,
  cache_steps=D, retrieve_parent_token, lower_bound)`; wrapper `kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:391-434`
  (`N = len(cu_seqlens)-1`, scratch pitch `stride(0)//(HV*K*V)` 429-434, `NK==1` assert 406).
- KDA metadata `MambaAttnBackendBase._forward_metadata` `hybrid_linear_attn_backend.py:121-302`: `mamba_cache_indices =
  req_to_token_pool.get_mamba_indices(req_pool_indices)` (138-143), padded rows → -1 via `_original_batch_size` (153-156);
  extend: `query_start_loc[:bs]=extend_start_loc; [bs]=start[-1]+extend_seq_lens[-1]` (253-260); TARGET_VERIFY:
  `query_start_loc = arange(0, input_ids.shape[0]+1, step=draft_token_num)` (236-242); `retrieve_next_token/next_sibling`
  copied and `retrieve_parent_token` derived only when `self.topk > 1` (244-251; `self.topk = get_spec().speculative_eagle_topk or 0`, 71).
  `ForwardMetadata` fields: `R/layers/attention/mamba/mamba2_metadata.py:29-82`.
- Post-verify commit (optional for a latency harness): `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify` (1340-1427)
  → `scatter_mamba_states_after_mtp_verify` (`kernels/ops/mamba/mamba_state_scatter_triton.py:678-690`), driven by
  `R/speculative/spec_utils.py:858+`.

**MLA:** `KimiK3MLAAttention` (`kimi_k3.py:2058`, `skip_rope=True` 2091, gate wraps o_proj 2214-2242) →
`DeepseekV2AttentionMLA.dispatch_attn_forward_method` `R/models/deepseek_v2.py:1992-2019`: decode → decode backend str;
TARGET_VERIFY/draft-extend → prefill backend str unless `get_spec().speculative_attention_mode == "decode"` (2006-2013); else
prefill str. Handler `handle_attention_trtllm_mla` `R/models/deepseek_common/attention_backend_handler.py:171-181`:
`MHA_CHUNKED_KV` iff `is_extend_without_speculative()` and (chunked prefix cache enabled or `sum(extend_prefix_lens_cpu)==0`,
92-97); anything else (incl. TARGET_VERIFY) → absorbed `MLA` (58-72). (`cutedsl_mla` is not registered → triton handler, 55;
the driver stamps `prefill_attention_backend_str` at `D:1015`.)
- MIXED/EXTEND (MHA_CHUNKED_KV): `forward_normal_prepare` `attention_forward_methods/forward_mha.py:179-292` (fetch_qkv_latent
  from attn-tp context set in `KimiK3DecoderLayer._run_self_attn_inner` `kimi_k3.py:2574-2589`; KV write `_set_mla_kv_buffer`
  583-594 at `out_cache_loc`), `forward_normal_chunked_kv_core` 334-393: `has_extend_prefix = any(extend_prefix_lens_cpu)`
  (342-344); once per step `prepare_chunked_prefix_cache_info` (355-363) in `R/model_executor/forward_batch_deepseek_mha_mixin.py:162-251`
  (**`prefix_chunk_len = capacity // batch_size` 199 — decode rows count in `batch_size`, so they shrink the prefix chunk and add
  prefix chunks**; `num_prefix_chunks` 201-203; kv indices from `req_pool_indices`/`req_to_token` 96-104); then current-chunk
  causal ragged kernel `TRTLLMMLABackend.forward_extend` `trtllm_mla_backend.py:1783-1805` (`forward_prefill_metadata
  .cum_seq_lens/max_seq_len/seq_lens`), then per prefix chunk `_chunked_prefix_attn_mha` 468-581 → 1732-1781 (non-causal,
  `prefix_chunk_seq_lens[chunk_idx]`, `fixup_zero_kv_rows` for rows with 0 prefix in that chunk) + `merge_state`. Prefill metadata
  `init_forward_metadata` 811-843: q lens = `seq_lens - extend_prefix_lens` (828), `max_seq_len = max(extend_seq_lens_cpu)` (837).
- TARGET_VERIFY (absorbed MLA): `forward_absorb_core` `forward_mla.py:673-752` → `attn_mqa(...)` (739-752; `rotary_emb=None` so
  no fused rope) → `RadixAttention.forward` → `AttentionBackend.forward` (`base_attn_backend.py:251-285`) →
  `TRTLLMMLABackend.forward_extend` verify branch 1473-1724: optional fp8 quantize (1487-1521), `set_mla_kv_buffer(layer,
  out_cache_loc, k, k_rope)` (1524-1535), q concat (1542-1550), `metadata = fb.decode_trtllm_mla_metadata or
  self.forward_decode_metadata` (1561-1573), `draft_token_num = spec_info.draft_token_num` (1584), `max_seq_len =
  metadata.max_seq_len_k + draft_token_num` (1586-1588), `q.view(bs, -1, heads, head_dim)` (1590-1592),
  `_run_decode_kernel(query, kv_cache, metadata.block_kv_indices, metadata.seq_lens_k, max_seq_len)` (1702-1709) →
  `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (1068-1083). Verify metadata `init_forward_metadata` 844-929:
  `max_seq = seq_lens_cpu.max() + self.num_draft_tokens` (861-869), `seq_lens_k = (seq_lens + num_draft_tokens).int32` (868-874),
  `self.num_draft_tokens = get_spec().speculative_num_draft_tokens` (284), block table `_create_block_kv_indices` (405-461) sized
  by `_calc_padded_blocks` (326-347). `CuteDslMLABackend` inherits all of this (`cutedsl_mla_backend.py:50-66`; with cp_world≤1
  defers to the parent, 92-104). Neither reads `custom_mask` or `retrieve_*` → **chain-causal verify only**.

## 3. Scheduler composition of MIXED (`R/managers/schedule_batch.py`, `scheduler.py`)
- Decision: `is_mixed_chunk = chunked_prefill_size and enable_mixed_chunk` (scheduler.py:1317-1319); 4038-4060:
  `running_batch.prepare_for_decode(); new_batch.mix_with_running(running_batch); new_batch.decoding_reqs = running_batch.reqs`.
- `prepare_for_extend` 2632-2679: `input_ids = fill_ids[len(prefix_indices):]`, `seq_lens = P+C`, `prefix_lens = P`,
  `extend_lens = C`, `out_cache_loc` from `alloc_for_extend` (`R/mem_cache/allocation.py:344`).
- `prepare_for_decode` 3424-3475: `out_cache_loc = alloc_for_decode(token_per_req=1)` (3464), `seq_lens += 1` (3471-3473) →
  decode row seq_len is L (written slot = index L-1).
- `mix_with_running` 3016-3087: MIXED (3017); `running_prefix_lens = seq_lens_cpu - 1` (3023); `out_cache_loc = cat([extend
  locs, running locs])` (3066); `merge_batch` (3068 → 3605-3627); `prefix_lens += running_prefix_lens`, `extend_lens += [1]*D`,
  `extend_num_tokens += D` (3080-3082). **Order: prefill rows first, decode rows after.**
- `ForwardBatch.init_new` 758-1000: `batch_size = len(seq_lens)` (817); non-decode: `extend_seq_lens = batch.extend_lens`,
  `extend_prefix_lens = batch.prefix_lens` (786-792); `seq_lens_sum` backfilled (811-812); EXTEND/MIXED: int32 tensors +
  `*_cpu` lists (934-945), `extend_num_tokens` (951), `positions, extend_start_loc = compute_position(...)` (952-957;
  `compute_position_torch` 1926-1940). DECODE/TARGET_VERIFY (930-932): extend_* stay None; `positions = spec_info.positions`
  if present (923-927) else `clamp_position(seq_lens)`.

## 4. TARGET_VERIFY prep (`R/speculative/eagle_utils.py`, `eagle_info.py`)
- `eagle_prepare_for_verify` 513-615: `batch.input_ids = verify_input.draft_token` (551); `out_cache_loc =
  assign_extend_cache_locs_uniform(req_pool_indices, req_to_token, start_offset=seq_lens, bs, draft_token_num)` (562-569 =
  `req_to_token[req, L:L+D]` int64, `kernels/ops/speculative/cache_locs.py:443-469`); `prepare_mamba_track_for_verify` (575;
  `spec_utils.py:768-794` sets `mamba_track_mask/seqlens = None`); mode TARGET_VERIFY (584-586); `init_new(capture_hidden_mode=FULL)`
  (592-597). `seq_lens` stay at the committed length L (D slots pre-allocated in `eagle_prepare_for_decode` 1004-1063).
- `EagleVerifyInput` (`eagle_info.py:16-41`): `draft_token, custom_mask, positions, retrieve_index, retrieve_next_token,
  retrieve_next_sibling, retrieve_cum_len, spec_steps, topk, draft_token_num, capture_hidden_mode, seq_lens_sum, seq_lens_cpu,
  draft_probs`; `num_tokens_per_req` defaults to `draft_token_num` (39-41); `ragged_verify_layout` class default None
  (`spec_info.py:394`). Minimal chain template: `_build_trivial_verify_input` `R/speculative/eagle_worker_v2.py:1371-1424`
  (`retrieve_index = arange(bs)[:,None]`, next/sibling = -1, `positions = seq_lens`); production assembly
  `R/speculative/eagle_worker_common.py:388-403`; tree builder `build_tree_kernel_efficient` (`eagle_utils.py:150-292`).
- KDA state buffers (`R/mem_cache/memory_pool.py`): `intermediate_ssm` `[layers, spec_state_size+1, D, HV, K, V]` in ssm dtype
  (765-776; fp32 required by the fused/dspark verify contracts kda_backend.py:1298/1495); KDA conv windows DENSE
  `[layers, spec_state_size+1, D, 3, dim]` (833-846; `KimiLinearStateShape.disable_conv_window_dedup=True`, `R/configs/mamba_utils.py:305`);
  per-layer view `at_layer_idx` (415-428; `conv`/`intermediate_conv_window` are lists); `spec_state_size = max_running_requests`
  (`R/mem_cache/kv_cache_configurator.py:553`). `verify_intermediate_state_indices = arange(req_to_token_pool.size)` int32
  (kda_backend.py:452-457; `linear/utils.py:106-138`) → `intermediate_*` row count ≥ `pool.size + 1`. The driver's stub
  `layer_cache`/`mamba_pool` (`D:569-577`) must grow `intermediate_ssm`, `intermediate_conv_window` (`replayssm_rawv` may stay None).

## 5. Field spec — MIXED (build as EXTEND; C new tokens for 1 chunk req with prefix P, plus D decode rows with context L)
| Field | KDA needs | MLA (MHA_CHUNKED_KV) needs | Scheduler computes |
|---|---|---|---|
| `forward_mode` | `is_extend()` → `forward_extend`, metadata extend branch (hybrid 221, 253-260) | `is_extend_without_speculative()` (handler 176-179); trtllm prefill metadata (811-843) | MIXED (3017); eager rewrites to EXTEND (eager_runner 216-222) |
| `batch_size` = 1+D | sizes `query_start_loc` (253-260) | `prefix_chunk_len = capacity // batch_size` (mixin 199); ragged kernel batch (1795) | `len(seq_lens)` (init_new 817) |
| `input_ids` `[C+D]` int64 | shape only | shape only | prefill ids then decode tokens (2641; 3034) |
| `req_pool_indices` `[1+D]` | `get_mamba_indices` (138-143) | prefix-chunk kv indices (mixin 96-104) | cat (3617-3619) |
| `seq_lens`/`seq_lens_cpu` = `[P+C, L, …]` | unused | q lens = `seq_lens - extend_prefix_lens` (828) | 2643, 3471-3472, cat 3623 |
| `seq_lens_sum` | unused | unused | `seq_lens_cpu.sum()` (811-812) |
| `out_cache_loc` `[C+D]` int64 | unused (slot-indexed state) | `set_mla_kv_buffer` for all C+D rows (forward_mha 590-594) | `cat([alloc_for_extend, alloc_for_decode])` (3066); decode slot = L-1 |
| `positions` `[C+D]` int64 = `[P..P+C-1, L-1, …]` | unused | unused (skip_rope) | `compute_position_torch` (1926-1940) |
| `extend_num_tokens` = C+D | unused | unused | 3082 |
| `extend_seq_lens` int32 `[C,1,…]` + `_cpu` | `query_start_loc` (256-259), conv `seq_lens_cpu` (852), `chunk_kda extend_seq_lens_cpu` (906) | `max_seq_len = max(extend_seq_lens_cpu)` (837) | `extend_lens + [1]*D` (3081), init_new 938-945 |
| `extend_prefix_lens` int32 `[P, L-1, …]` + `_cpu` | `has_initial_state = >0` (822) | q lens (828), prefix sum (92-97), chunking (forward_mha 342-344; mixin 181-219) | `prefix_lens + [seq-1]*D` (3080, 3023) |
| `extend_start_loc` int32 exclusive cumsum | `query_start_loc[:bs]` (256) | not used by trtllm | init_new 952-957 |
| `spec_info` | None | None (else absorbed MLA) | None |
| `mamba_track_indices/mask/seqlens` | None (149-152, 883) | n/a | only with mamba extra buffer |
| `_original_batch_size` | = bs (153-156) | n/a | DP padding only |
| `num_prefix_chunks`/`prefix_chunk_len` | n/a | None at step start (mixin 185-187; driver `_k3_pre_step` `D:1058-1061`) | fresh ForwardBatch |
| `global_num_token_non_padded_cpu` | None (RLA 116-123) | n/a | mlp-sync only |

## 6. Field spec — TARGET_VERIFY (B requests × D=k+1 draft tokens, context L)
| Field | KDA needs | MLA (absorbed, trtllm/cutedsl) needs | Scheduler computes |
|---|---|---|---|
| `forward_mode` TARGET_VERIFY | `_forward_target_verify` (806-807); metadata 230-251 | handler → MLA (176-181); dispatch uses prefill backend str unless `speculative_attention_mode=="decode"` (deepseek_v2 2006-2013); verify metadata 844-929, forward 1557-1724 | eagle_utils 584-586 |
| `batch_size` B | `seq_len // D == B` (1050) | `q.view(bs,-1,…)` (1591), block table rows (899-905) | `len(seq_lens)` |
| `input_ids` `[B*D]` | `query_start_loc = arange(0, len+1, D)` (236-242) | shape | `= spec_info.draft_token` (551) |
| `req_pool_indices` `[B]` | mamba slots (138) | block table (902) | unchanged |
| `seq_lens` `[B]` = L (no +D) / `seq_lens_cpu` | unused | `seq_lens_k = seq_lens + num_draft_tokens` (869-871), `max_seq = max+num_draft_tokens` (861-869) | stays at base |
| `out_cache_loc` `[B*D]` int64 = `req_to_token[req, L:L+D]` | unused | `set_mla_kv_buffer` (1524-1535) | 562-569 |
| `positions` `[B*D]` int64 = L + depth (chain L+j) | unused | fp8 fused-rope path only (off for K3) | `spec_info.positions` (init_new 923-927) |
| `extend_*` | None OK (guard 818 non-verify only) | None → prefix sum 0 (92-97) | init_new skips (930-932) |
| `spec_info.draft_token_num` = D | hybrid 239, kda 1016 | 1584 | EagleVerifyInput |
| `spec_info.ragged_verify_layout` = None | 231, 1017 | 1589 | default |
| `spec_info.retrieve_next_token/next_sibling` `[B,D]` long | only if `topk>1` (244-251) → conv tree walk (1142-1144) | not read | 216-219 |
| `custom_mask`, `retrieve_index`, `topk`, `spec_steps`, `capture_hidden_mode`, `seq_lens_sum/cpu`, `num_tokens_per_req` | not read | not read (flashinfer parent only, eagle_info 82-138) | `num_tokens_per_req` gates `can_run_graph` (decode_cuda_graph_runner 653-659) |
| State buffers | `conv[0]` `[N,3,dim]` bf16, `temporal` `[N,HV,K,V]` fp32, `intermediate_ssm` `[N,D,HV,K,V]` fp32 contiguous, `intermediate_conv_window[0]` `[N,D,3,dim]` bf16, `N ≥ pool.size+1`; `verify_intermediate_state_indices` from `pool.size` (452-457) | KV pool + `req_to_token` rows with L+D valid, 64-page-aligned slots; padded block table (326-347) | memory_pool 765-776, 833-846 |
| `get_spec()` (ServerArgs via `publish`, `R/runtime_context.py:1571`) | `speculative_eagle_topk` (hybrid 71, kda 422), `speculative_algorithm` (kda 491) | `speculative_num_draft_tokens` (trtllm 284, 869-870), `speculative_attention_mode` (deepseek_v2 2010) | ServerArgs |

## 7. Guards that reject a hand-built batch
- KDA: `extend_prefix_lens is None` in non-verify extend → RuntimeError (818-821); missing `extend_start_loc/extend_seq_lens`
  → `query_start_loc` failure (hybrid 253-260); TARGET_VERIFY with `spec_info=None` → AttributeError (hybrid 231/239);
  `intermediate_ssm` None → RuntimeError (987-991); `seq_len % draft_token_num != 0` breaks the dense view (1108); `lower_bound`
  with a non-Triton decode/verify kernel → NotImplementedError (262-267, 307-313); fused chain verify needs env
  `SGLANG_OPT_FUSED_KDA_VERIFY` (439) and D≥3 (1239) and int32 `cache_indices` (1357; the stub returns int32, `D:589`); verify
  wrapper needs a contiguous `intermediate_states_buffer` with pitch D*HV*K*V (429-434), `NK==1` (406). Do **not** pre-sigmoid
  `beta` under TARGET_VERIFY (kimi_k3.py:2013-2016).
- MLA verify: `get_spec()` must provide `speculative_num_draft_tokens` (else `max_seq + None`, 869-870; `dense_q_indptr_verify`
  285-289); `kv_cache.dtype == self.data_type` (1669); `k`/`k_rope` not None (1525-1527); `metadata.batch_size <
  forward_batch.batch_size` → re-plan (1570-1573); page_size must divide 128 (336-347); trtllm-gen rejects DCP for q_len>1 (1040-1049).
- MLA extend/MIXED: `extend_prefix_lens_cpu`/`extend_seq_lens_cpu` lists required (819, 837; handler 92-97);
  `disable_chunked_prefix_cache and has_prefix` → flashinfer fallback (820-826, 1473-1482); chunked MHA asserts
  `prefix_chunk_idx`/`prefix_chunk_cu_seq_lens` not None and `q_rope/k_rope is None` (1734-1737); mixin asserts an MLA kv pool
  (176-179) and `max chunk tokens ≤ capacity` (241); reset `num_prefix_chunks`/`prefix_chunk_len` per step (185-187, 355).
- CUDA graph: MIXED/EXTEND never; TARGET_VERIFY only via the decode graph runner with `spec_info.num_tokens_per_req ==
  captured_req_width` (653-659) and `load_batch` + `mark_forward_metadata_ready` (eagle_utils 606-610); hybrid
  `init_cuda_graph_state` asserts `max_num_tokens % max_bs == 0` (503-507). Tree verify (`topk>1`) copies `retrieve_*` into
  static buffers (806-817).
- `MambaAttnBackendBase.__init__` tolerates a non-`HybridReqToTokenPool` stub (82-86).

## 8. Implementation plan for the driver (to do after the running KDA lcprefill rounds finish — a driver edit re-keys goldens)
- New tags: `mx<C>[p<P>]` on a decode point `B,L` = B decode requests at context L + one prefill chunk of C tokens with prefix P
  (e.g. `64,8192,mx16384`, `128,8192,mx4096p49152`); `vk<k>` on `B,L` = TARGET_VERIFY with D=k+1 draft tokens per request
  (e.g. `32,8192,vk3`). Both timed eagerly (MIXED) / eagerly first, graph later (verify is graph-eligible).
- MIXED state = existing `KDAPrefillState`/`MLAPrefillState` for the chunk row + `KDAState`/`MLAState` decode rows merged into one
  ForwardBatch in the scheduler's order (prefill rows first), fields per §5; MLA prefix-chunk planning per step via the
  existing `_k3_pre_step` hook. Golden: output rows for all C+D tokens + post-step KDA state / MLA KV rows of the D decode
  requests and the chunk; row-wise prefill rule for the chunk rows, strict rule for the decode rows.
- VERIFY state: extend the KDA stub cache with `intermediate_ssm` `[N,D,HV,K,V]` fp32 and `intermediate_conv_window`
  `[N,D,3,dim]` bf16; publish a spec config (`speculative_num_draft_tokens=D`, `speculative_eagle_topk=1`,
  `speculative_algorithm="EAGLE"`, `speculative_attention_mode="prefill"`) through the same `publish` path the driver uses
  for ServerArgs; build a chain `EagleVerifyInput` (`_build_trivial_verify_input` pattern) with `draft_token`, `positions = L+j`,
  `retrieve_*` = -1/arange, `draft_token_num=D`; `out_cache_loc = req_to_token[req, L:L+D]`; MLA KV pool sized L+D per request.
  Golden: output logits-equivalent hidden rows for all B*D tokens; KDA `intermediate_ssm` rows; MLA written KV rows.
