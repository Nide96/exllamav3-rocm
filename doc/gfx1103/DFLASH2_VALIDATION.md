# DFlash2 validation on Radeon 780M (gfx1103) — 2026-11-04

Native DFlash2 generated text successfully on the real Radeon 780M with the previously tested Heretic target. Three draft tokens improved both short workloads over fresh MTP 2 controls; seven improved Python more but slowed Linux. These are measurements of existing fork functionality, not new DFlash2 kernels or a runtime gain caused by the host include-order fix.

The [historical compatibility report](COMPATIBILITY_REPORT.md) retains the November 3 MTP and llama.cpp results. [BENCHMARK_EVIDENCE.json](BENCHMARK_EVIDENCE.json), section `dflash2_validation_2026_11_04`, supplies the new per-request timings, full output/reference token IDs, exact rendered prompts/input IDs, environments, aggregate calculations, numerical checks and source-file hashes. Its previous sections remain unchanged.

## Source and checkpoints

The measured checkout was upstream [360c902](https://github.com/phoenixhaxor/exllamav3-rocm/tree/360c9025188810a0ac9d434c837dc3cb6b8e39cd) with the historical two-line `setup.py` pre-include-`format` workaround and its already compiled native extension. No extension rebuild was required for DFlash2. The separately proposed pre-include-`chrono` patch in [PR #3](https://github.com/phoenixhaxor/exllamav3-rocm/pull/3) has separate build validation; these model timings must not be attributed to it.

- Target: [andrevp/Qwen3.8-27B-EXL3-3.5bpw-heretic](https://huggingface.co/andrevp/Qwen3.8-27B-EXL3-3.5bpw-heretic/tree/d5ef074d2f32ec6e01d670c5ed0bef10fef899cd), revision `d5ef074d2f32ec6e01d670c5ed0bef10fef899cd`. Native Heretic projections stayed enabled: 61 attention / 39 MLP.
- Draft: [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/tree/4f0436269bca761b071f05319e8e04a87cc633f9), revision `4f0436269bca761b071f05319e8e04a87cc633f9`.
- Draft weights: 1,470,916,078 bytes, SHA-256 `6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56`, verified before timed runs. The manifest contained 189 tensors and five draft layers, native block size 8 (up to seven draft tokens), target taps 5/19/33/47/61, hidden size 5120 and vocabulary 248320. Structural checks matched the 64-layer target. The draft was trained for the base target; these measurements establish its observed acceptance with this Heretic target, not universal compatibility with altered checkpoints.

No model-supplied Python was executed. The host remained CachyOS, ROCm 7.2.4 / Clang 22 / GCC 16 headers, Python 3.12.16 and PyTorch `2.13.0+rocm7.2` (`torch.version.hip=7.2.53211`). The native reported architecture was `gfx1103`, without an HSA override. Main/draft loading used `cuda:0`, PyTorch's HIP interface; no CPU layer-offload comparison was performed.

## Matched protocol

Six independent processes ran in this order: MTP 2, DFlash2 7, DFlash2 3, DFlash2 3, DFlash2 7, MTP 2 (ABCCBA). Each process generated both the same Linux explanation (80 input tokens) and Python CSV/Decimal/argparse program (91 input tokens), giving two repetitions per workload/configuration and 12 timed requests total.

All used one sequence, a 16,384-token cache capacity, FP16 main/draft KV, eight CPU threads, seed 1234, greedy temperature 0, thinking disabled, ignored EOS and exactly 256 output tokens. A separate 24-token warmup and a separate one-token first-logit validation before each timed request were excluded from timing. Each request had a fresh Generator/page table, no recurrent prefix cache and zero prompt-cache reuse; maximum chunk size was 512. `EXL3_ADAPT_WINDOW=0` and dynamic drafting was disabled.

The same previously validated settings were retained for all six runs: system rocBLAS Tensile data, `EXL3_GDN_MK=0`, `EXL3_NOGRAPH=mlp,gdn`, `EXL3_GDN_REG=1`, native ablation/fusions, MGEMM, presampling/rewind, target blocks 36 and attention split multiplier 16. The JSON records the complete common environment. MTP's recurrent history was four; DFlash2's was seven, including the three-token window, as allocated from the model's native default. AC power and the performance governor were observed throughout the 383 telemetry records. Preflight required AC, available memory and temperatures at most 65 °C. All six timed runs started after the draft download/hash verification, with no competing llama-server or Ollama model recorded.

For each request, decode throughput is `255 / (last synchronization time - first token time)`. Pooled throughput is `sum(255) / sum(decode seconds)`, not the arithmetic mean of rates. Gain is `100 * (DFlash2 rate / fresh MTP 2 rate - 1)`. TTFT and model load are excluded, so these are not end-to-end latency gains. The no-draft reference was obtained while the download was active: its token IDs/logits are used for correctness, but its throughput is deliberately excluded.

## Recalculated results

| Configuration | Linux tok/s | Change vs MTP 2 | Python tok/s | Change vs MTP 2 | Pooled tok/s | Pooled change |
|---|---:|---:|---:|---:|---:|---:|
| Fresh MTP 2 | 9.602 | baseline | 12.560 | baseline | 10.884 | baseline |
| DFlash2, 3 tokens | 10.251 | +6.76% | 14.721 | +17.20% | 12.086 | +11.05% |
| DFlash2, 7 tokens | 9.171 | −4.48% | 17.297 | +37.71% | 11.987 | +10.14% |

Individual repeat rates, Linux / Python:

| Configuration | Linux repeats | Python repeats | Aggregate draft acceptance |
|---|---|---|---:|
| MTP 2 | 9.529, 9.676 | 12.667, 12.455 | 600 / (600 + 248) = 70.75% |
| DFlash2 3 | 10.198, 10.305 | 14.826, 14.618 | 662 / (662 + 424) = 60.96% |
| DFlash2 7 | 9.129, 9.214 | 16.635, 18.014 | 718 / (718 + 1424) = 33.52% |

Acceptance is a descriptive engine counter, not a quality score. Longer proposals can still accelerate Python despite lower acceptance. Three tokens is the balanced option for these two prompts; seven is faster on the tested Python prompt. The 0.83% pooled difference between the DFlash2 windows, with two repetitions and no confidence interval, does not establish a general winner.

The controls above are fresh November 4 MTP 2 measurements. November 3 MTP 3, the optimization sweep and llama.cpp are separate campaigns. MTP 3 was not retested here; no matched DFlash2-versus-MTP-3 or DFlash2-versus-llama.cpp gain is claimed.

## Output and numerical validation

All eight DFlash2 continuations (two windows × two prompts × two repeats) matched all 256 no-draft reference output IDs exactly, not merely their decoded text. The JSON contains the arrays for independent comparison. This establishes identity for these eight greedy continuations only.

Both MTP Python outputs matched; both MTP Linux outputs first diverged at zero-based output index 79. That cause was not isolated. Accordingly, the speed comparison matches settings/prompts but not every MTP output trajectory; it cannot prove performance at universally equivalent outputs or quality.

All 12 separate first-token checks matched the forbidden (`-inf`) mask and argmax, contained no NaN or positive infinity, and passed `rtol=0.001, atol=0.05` on finite logits. Maximum observed absolute difference was 0.04296875. This is a first-token gate, not a proof for every intermediate score or token.

### Original top-k failure and bounded follow-up

Of 18 upstream test functions, 16 passed when invoked directly. `test_grouped_dynamic_convolve_cuda_rejects_bad_geometry` was not run because pytest was absent. `test_topk_cuda_matches_torch` failed its strict exact-index comparison at `(8, torch.float16, 1, 5)`. The original test was neither changed nor reclassified as passing.

Follow-up diagnostics tested six combinations (k=8/16/32, FP16/FP32), 84 rows total. Every case passed sorted-score comparisons at `rtol=atol=1e-5`, valid unique indices and the condition that no omitted score exceeded the minimum selected score beyond tolerance. Maximum sorted-score error was 3.814697265625e-6. A separate adversarial case with vocabulary 40,960 and all top-16 values in one strided thread slice also passed.

Every reproduced index-set discrepancy was at an exactly tied cutoff, with zero selection margin and equal raw/transformed scores for the exchanged elements. One example exchanged indices 3440 and 1997, both raw 12.671875 / transformed 18.4714298248291. The [current PyTorch top-k API documentation](https://docs.pytorch.org/docs/2.15/generated/torch.topk.html) explicitly allows unstable selection of indices among ties; this is a contract reference, not the tested PyTorch version. Thus strict agreement with PyTorch's chosen indices is stronger than mathematical top-k correctness for these inputs.

This supports the observed score/rank validity without dismissing the original failure. Different tied IDs may still change a draft path, acceptance or throughput. Target verification and the eight reference-identical DFlash2 outputs provide bounded complementary evidence, not proof for arbitrary prompts, sampling temperatures or dtypes. A non-tie ranking inversion, duplicate/invalid index or omitted higher score would undermine this reasoning. Broader selector stability remains untested.

## GPU attribution and resources

A separate 16-token smoke trace, outside the timed runs, reported 7,039 GPU events. Direct inspection found 7,026 `kernel` trace events, including 132 native `dflash2` and 2,912 `exl3_rdna3` events. These are different event categories, not interchangeable counts. DFlash2 names included dynamic convolution (half and float residual paths), top-k half/16 and selector walk half/BF16. This demonstrates process-attributed native GPU execution, beyond a global GPU busy reading.

Across 383 samples, GPU maximum was 78.0 °C and CPU maximum 78.375 °C (78.4 °C rounded). The original 96 °C CPU/GPU safety cutoff remained unchanged. Minimum available system memory was 27.706 GiB. All six timed processes exited 0 with no recorded guard event, OOM or GPU fault; this is a sampled short-run observation, not long-duration stability certification.

PyTorch allocated memory after main/draft load was approximately 13.348 GiB for MTP and 15.311 GiB for DFlash2 (+1.963 GiB). This excludes later Generator allocations and is not total peak memory or dedicated VRAM: the 780M uses shared system memory.

## Reproduction and evidence availability

Use the patched checkout and its configured environment described in the [compatibility report](COMPATIBILITY_REPORT.md). Download the pinned native EXL3 target and separate native DFlash2 draft; GGUF files cannot serve as these inputs. Model Python must not be executed. The [portable helper](reproduce_gfx1103.py) now accepts the draft and an explicit validation pass:

```bash
python reproduce_gfx1103.py --repo "$REPO_DIR" --model "$MODEL_DIR" --output "$NEW_MTP_DIR" --mode on --draft-tokens 2 --validation-pass
python reproduce_gfx1103.py --repo "$REPO_DIR" --model "$MODEL_DIR" --draft "$DRAFT_DIR" --output "$NEW_DFLASH3_DIR" --mode dflash --draft-tokens 3 --validation-pass
python reproduce_gfx1103.py --repo "$REPO_DIR" --model "$MODEL_DIR" --draft "$DRAFT_DIR" --output "$NEW_DFLASH7_DIR" --mode dflash --draft-tokens 7 --validation-pass
```

`python` runs the helper in the selected environment. `--repo` selects the already compiled checkout, `--model` the target and `--draft` the separate DFlash2 directory. Each `--output` must be a new directory, so evidence cannot be overwritten. `--mode on` selects integrated MTP; `--mode dflash` selects the native external draft. `--draft-tokens` fixes its proposal window; `--validation-pass` saves a separate one-token logit array before each timed workload, matching this round's warmup policy. The helper applies child-process settings and the 96 °C thermal stop; it changes no services, weights or system configuration.

The commands illustrate one run per variant. To reproduce the measured design, use six fresh output directories and the ABCCBA order above, with AC, no competing inference/download, sufficient available memory, monitored GPU/CPU and temperatures at most 65 °C before each process. The full measured supervisor also checked AC/memory/kernel faults; the portable helper supplies the thermal stop, not all supervisor checks. Compare saved logits and full token IDs to a separately generated `--mode off --validation-pass` reference; the helper saves these data but does not automatically certify cross-run equivalence.

The helper is an adaptation of the measured harness. This documentation update checked its syntax, CLI guards and unchanged prompts without another GPU benchmark; it does not claim a new run of the wrapper or include-order patch.

Original per-request JSON/text, first-logit NPY arrays, metadata, six telemetry/kernel logs, GPU trace, unit logs, worker/supervisor/guard/campaign/analysis scripts and checkpoint provenance were found in the local `benchmark-exllamav3-dflash2-20261104` campaign. The published JSON contains raw timing/token evidence and a 91-file relative-path SHA-256 manifest. Full telemetry, arrays, trace and original scripts remain local and are not individually downloadable from this documentation commit; their publication after privacy review would be needed for independent reprocessing of every trace/logit sample. Hashes identify retained artifacts, but do not substitute for making them available. No new GPU benchmark was run during this audit/publication.

## Remaining limits

Two short prompts, two repeats per variant, one device, one target/draft pair and greedy sampling do not establish general speedups, statistical significance, sustained thermals, long-context correctness/performance or complete-answer quality. Reserving 16K cache is not a 16K input test. Vision, MoE, CPU layer offloading, other draft models, DFlash v1, concurrent serving and other GPU architectures remain outside this validation. The original strict top-k test and the skipped pytest-dependent check remain explicit follow-up items.
