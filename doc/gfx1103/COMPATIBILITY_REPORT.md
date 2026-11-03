# [Compatibility Report] ExLlamaV3 ROCm running on Radeon 780M (gfx1103) with Qwen3.8-27B EXL3

Supporting files: [benchmark evidence](BENCHMARK_EVIDENCE.json) and [reproduction helper](reproduce_gfx1103.py). The separate [host build fix](https://github.com/Nide96/exllamav3-rocm/commit/c111c2f5c02e8cfa30ca9b9de31250c044ee992b) is published on `fix/rocm-clang22-gcc16-format`; use that patched checkout for reproduction. This documentation branch preserves the measured upstream source.

Text-only generation, native EXL3 RDNA3 kernels and integrated MTP worked on a Radeon 780M with the configuration below. This reports one checkpoint and a bounded set of workloads. Thank you to phoenixhaxor and the other contributors for the ROCm port and kernels: the work reported here is build diagnosis, gfx1103 validation and measurement of existing options, rather than development of those kernels.

Tested repository: [phoenixhaxor/exllamav3-rocm at 360c902](https://github.com/phoenixhaxor/exllamav3-rocm/tree/360c9025188810a0ac9d434c837dc3cb6b8e39cd). Measurements and validation were recorded on 2026-11-03. The host build workaround is proposed separately; this report does not request making every runtime setting a project-wide default.

## Environment

| Component | Tested configuration |
|---|---|
| Laptop / CPU | Lenovo ThinkPad P14s Gen 5 AMD; Ryzen 7 PRO 8840HS |
| GPU | AMD Radeon 780M Graphics, native `gfx1103`, wave32 |
| GPU resources | 12 physical CUs; HIP reports 6 multiprocessors/WGPs, not 6 physical CUs |
| Memory | 64 GiB installed, as reported by the owner; OS-visible `MemTotal` 57,212,488 KiB, about 54.56 GiB, after hardware reservations |
| OS | CachyOS, Arch-based Linux, x86_64 |
| Python | 3.12.16, isolated virtual environment |
| PyTorch | `2.13.0+rocm7.2`; `torch.version.hip = 7.2.53211` |
| System ROCm | 7.2.4; this is separate from the wheel's ROCm 7.2.1 build metadata |
| Host compiler | AMD Clang `22.0.0git` from `rocm-llvm 7.2.4` |
| C++ headers | GCC/libstdc++ 16.2.1; GCC is not the compiler used for the recorded extension build |
| Other installed compiler | System Clang 23.1.4; used for additional preprocessing probes only |
| Build target | Explicit `PYTORCH_ROCM_ARCH=gfx1103`; no `HSA_OVERRIDE_GFX_VERSION` |
| Benchmark conditions | AC power, performance power policy, one active inference sequence; CPU threads 8 |

A successful extension build does not by itself establish that a PyTorch wheel ships BLAS data for this GPU. That distinction mattered here.

## Checkpoint and integrity

Model: [andrevp/Qwen3.8-27B-EXL3-3.5bpw-heretic](https://huggingface.co/andrevp/Qwen3.8-27B-EXL3-3.5bpw-heretic/tree/d5ef074d2f32ec6e01d670c5ed0bef10fef899cd), revision `d5ef074d2f32ec6e01d670c5ed0bef10fef899cd`.

The checkpoint uses the `Qwen3_5ForConditionalGeneration` architecture in its config, despite the repository's Qwen3.8 model name. It has 64 text layers (48 GDN / 16 full attention), one integrated MTP layer, EXL3 `mul1` quantization averaging 3.5 bpw, 6-bit output head and 4-bit MTP weights. These results are specific to this Heretic checkpoint with native runtime abliteration enabled.

Both complete weight shards were SHA-256 hashed and matched the recorded Hugging Face ETags for that revision. All 2,426 indexed tensors were present, including MTP tensors. The abliteration sidecar was also hashed:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 8,530,774,298 | `7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6` |
| `model-00002-of-00002.safetensors` | 6,807,634,163 | `411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2` |
| `abliteration.safetensors` | 6,176,960 | `ab7b6a64e8583bfe30da36f5f16c6793e189016ad12e58ba232a49aa416f9538` |

Native model loading reported 61 attention and 39 MLP abliteration projections. No Python code downloaded with the checkpoint was executed. A different checkpoint, quantization or ablation policy requires its own validation.

## Build and runtime requirements

### Host compilation

The first blocking compiler diagnostic was `expected identifier` in GCC 16 `<format>`, inside `[[__gnu__::__noinline__]]`. HIP's host headers had already defined `__noinline__` as an empty macro. The original two-line `setup.py` workaround pre-included `<format>` on HIP host compilation; it allowed the complete gfx1103 build to finish and was used for the measurements below.

The separate minimal PR proposal pre-includes `<chrono>` instead, under a non-Windows HIP guard, in `setup.py` and the independent JIT flag list. It avoids a direct `<format>` dependency on older libraries while loading the failing GCC 16 header chain early. It changes no RDNA3 kernels. Fresh checks of that variant are reported separately from the historical model measurements.

`-Ofast` deprecation, fast-math NaN/Inf warnings, feature-macro redefinitions and unused `-lineinfo` diagnostics were warnings in these logs. Ninja's stop and Python's final extension-build exception were consequences of the compiler error. A `(null): No such file or directory` message also persisted on successful runs; its origin has not been resolved and it is not claimed as fixed.

The build script defaults to gfx1100, so explicitly select gfx1103. Use the separate PR's build instructions in a disposable checkout and dedicated Python environment.

### Dependency provenance

The final working system had development headers/libraries supplied by `rocthrust`, `rocprim`, `hipsparse`, `rocsparse`, `hipblas`, `hipblaslt`, `hipsolver`, `rocsolver`, `hiprand`, `rocrand`, `rocblas`, the HIP runtime and `rocm-llvm`, all from the system ROCm 7.2.4 package family. These are dependencies used or encountered by this environment, not a claim that every distribution must install every package separately.

Earlier user diagnostics had identified missing development headers, including hipSOLVER; by the recorded compilation audit hipSOLVER was already installed. The dependency installation directly established during this diagnosis was `hiprand` with its `rocrand` dependency, installed by the owner after a missing-header failure. Existing packages were not all newly installed. System rocBLAS was already present; selecting its Tensile data below was a runtime path workaround, not a new BLAS installation or replacement of PyTorch's shared libraries.

No system ROCm, PyTorch or libstdc++ header was patched. No GPU architecture spoofing was used.

### Runtime configuration

These settings were passed to the inference child process:

```bash
env ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library EXL3_GDN_MK=0 EXL3_NOGRAPH=mlp,gdn EXL3_ABLATION=1 python reproduce_gfx1103.py --repo "$REPO_DIR" --model "$MODEL_DIR" --output "$NEW_OUTPUT_DIR" --mode on --draft-tokens 2
```

Set `REPO_DIR` to the patched checkout, `MODEL_DIR` to the existing checkpoint and `NEW_OUTPUT_DIR` to an unused results directory. `env` applies variables to this process, without changing shell startup files or services. `ROCBLAS_TENSILE_LIBPATH` selects installed native gfx1103 kernel data: the bundled PyTorch rocBLAS data lacked gfx1103 files and the first model attempt aborted. Pointing to the system data passed the tested FP32/FP16/BF16 matrix checks. This is validated for the recorded wheel/system combination; it is not a general guarantee that arbitrary wheel and system Tensile versions can be mixed.

`EXL3_GDN_MK=0` disables the grid-barrier variant discussed below. `EXL3_NOGRAPH=mlp,gdn` retains the measured graph policy. `EXL3_ABLATION=1` enables the native Heretic projections. `--mode on` loads the checkpoint's MTP component; `--draft-tokens 2` fixes the speculative window at two. The companion `reproduce_gfx1103.py` is an adaptation of the measured native harness, with paths supplied as arguments, CPU/GPU monitoring and the same 96 °C cutoff. It is intended to accompany this report; its new portable wrapper has been syntax-checked, not used to claim an additional benchmark run.

The model and MTP components were explicitly loaded on `cuda:0` (PyTorch's HIP device interface), avoiding automatic split/calibration. Native CPU-preferred embeddings remain an implementation detail. This is not a tested CPU/GPU layer-offload speedup. The tested dense 27B checkpoint does not exercise MoE expert offloading.

### Why the GDN MK path stays disabled

In the tested [GDN source](https://github.com/phoenixhaxor/exllamav3-rocm/blob/360c9025188810a0ac9d434c837dc3cb6b8e39cd/exllamav3/exllamav3_ext/gdn.cu#L2235), the normal kernel launch uses a software grid-wide barrier: resident blocks wait for every block to arrive. For batch size 1 and this model's 48 value heads, the [launcher](https://github.com/phoenixhaxor/exllamav3-rocm/blob/360c9025188810a0ac9d434c837dc3cb6b8e39cd/exllamav3/exllamav3_ext/gdn.cu#L2783) creates `1 × 48 × 4 = 192` blocks of 128 threads. At wave32 that requires 768 simultaneously resident waves.

The recorded hardware limits are 12 CUs and at most 32 resident waves per CU: an upper bound of 384 waves for the entire GPU, even before register and other occupancy limits. The whole grid therefore cannot reside simultaneously. Blocks spinning at the barrier may prevent unscheduled blocks from ever reaching it. HIP reporting six WGPs does not double the physical wave capacity. This configuration fails the path's stated co-residency assumption; the path was excluded by analysis and was not deliberately executed to induce a hang. `EXL3_GDN_REG=1`, a different existing register-state path, remains enabled.

## Generation results

| Complete configuration | Pooled decode throughput |
|---|---:|
| llama.cpp Vulkan, GGUF UD-Q4_K_XL, no MTP | 4.120 tok/s |
| ExLlamaV3 ROCm, EXL3 3.5 bpw Heretic, no MTP | 5.806 tok/s |
| llama.cpp Vulkan, GGUF UD-Q4_K_XL, MTP | 7.767 tok/s |
| ExLlamaV3 ROCm, EXL3 3.5 bpw Heretic, MTP 2 | 10.714 tok/s |

The measured complete configurations differ by +40.93% without MTP and +37.94% with MTP. Within ExLlamaV3, MTP improves the pooled rate by +84.53%. These are not isolated engine-only gains: weight format, quantization and Heretic ablation differ, and output sequences can differ. No equal-quality claim is made. The compiler workaround itself is not a runtime optimization.

The llama.cpp Vulkan binary was build 10666, commit `aa33ba6`, using RADV PHOENIX/Vulkan0 with all model layers on the GPU. The GGUF was `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 bytes. Both MTP modes used their checkpoint's integrated draft head, with two draft tokens for the MTP comparison.

### Measurement method

- Two fixed French requests: a Linux memory explanation and a standard-library Python CSV/Decimal/argparse program. The rendered inputs were identical across engines, including the disabled-thinking template, and contained 80 and 91 input tokens respectively. Exact prompts are included in the companion reproduction helper.
- One sequence, 16,384-token cache capacity, FP16 KV cache, eight CPU threads, greedy sampling, seed 1234, 256 generated tokens per request, EOS ignored for fixed work, and zero reused prompt-cache tokens. ExLlamaV3 used maximum chunks of 512; llama.cpp used batch 2048 / microbatch 512 and flash attention enabled.
- A separate 24-token warmup was excluded. Four process runs per engine followed ABBA: no MTP, MTP, MTP, no MTP. Each process ran both requests, giving two measurements per workload and mode.
- ExLlamaV3 rate was `(256 − 1) / (last synchronization time − first token time)`. llama.cpp's reported decode rate likewise corresponds to 255 timed token intervals and `predicted_ms`. Pooled rate was `sum(timed token intervals) / sum(decode seconds)`, not an arithmetic mean of rates. The companion `BENCHMARK_EVIDENCE.json` preserves all 24 measured rows across the initial comparison and final profile comparison.
- Rates exclude TTFT and model load. The first no-MTP ExLlamaV3 Linux request had a 65.77 s cold TTFT despite the short warmup; the other initial requests were about 1.89–2.24 s. The exact initialization cause was not isolated. This cold cost must not be hidden in end-to-end latency claims.
- Aggregate draft acceptance was 588/854 = 68.85% in llama.cpp and 600/(600+248) = 70.75% in ExLlamaV3, over those measured MTP requests. These counters have engine-specific implementations; they are descriptive rather than a quality measure.

### Profile confirmation

A later, separate ABBA comparison confirmed fixed MTP 2 versus fixed MTP 3:

| Workload | MTP 2, balanced | MTP 3, code | Change with MTP 3 |
|---|---:|---:|---:|
| Linux explanation | 9.529 tok/s | 9.151 tok/s | −3.96% |
| Python code | 12.346 tok/s | 13.755 tok/s | +11.41% |
| Pooled | 10.756 tok/s | 10.991 tok/s | +2.18% |

These figures are from the later profile comparison, not replacements for the initial 10.714 tok/s measurement. MTP 2 remains the general profile; MTP 3 is an option for this coding workload. Both retain fixed drafting, FP16 KV, eight threads, `EXL3_RDNA3_TARGET_BLOCKS=36`, `EXL3_RDNA3_ATTN_SPLIT_MULT=16`, the GDN/graph settings above, and the fork's existing enabled fusions, MGEMM, presampling and rewind cache.

## Numerical, GPU and optimization evidence

1. **Build and target:** the historical full extension build exited 0, loaded the compiled shared object and produced native `hipv4-amdgcn-amd-amdhsa--gfx1103` code. This establishes the target without architecture spoofing.
2. **RMSNorm:** four extension calls, shapes `(1,256)` and `(4,1024)`, FP16 and FP32, passed CPU-reference comparisons at `rtol=1e-3`, `atol=1e-3`. Maximum observed FP16 absolute error was 0.0009765625; FP32 was 9.54e-7. This smoke test alone did not exercise rocBLAS; model testing subsequently exposed its missing Tensile data.
3. **BLAS path:** with system gfx1103 Tensile data selected, FP32, FP16 and BF16 GPU matrix products passed CPU-reference checks. Observed maximum absolute errors were approximately 1.34e-5, 0.015625 and 0, respectively. These apply to the recorded small test, not arbitrary shapes or precisions.
4. **Quantized GEMM:** the fork's existing `rocm_tests/test_rdna3_gemm.py` tested the real `model.language_model.layers.0.mlp.down_proj` projection (`k=17408`, `n=5120`, K4, mul1). Sampled weight reconstruction matched the independent NumPy trellis decode exactly. Fourteen row counts from 1 to 144, each with FP16 and FP32 output, passed the reconstructed-weight matrix reference, with `FAILS: 0`. The script prints representative successful cases only; their relative errors were about 5.5e-4 to 6.1e-4, below its 2e-3 criterion. This is one projection and quantization path, not every weight or kernel.
5. **GPU attribution:** a separate four-token PyTorch GPU trace contained 4,726 kernel events, including 1,860 `exl3_rdna3` kernel events and five GPU memcpy events. Profiling was outside the timed benchmarks. HIP DRM fdinfo did not expose per-process engine-time counters, so missing counters were recorded as unavailable rather than zero execution. Global busy readings alone were not used as proof of process attribution.
6. **Optimization campaign:** 34 distinct variants were measured against periodically refreshed controls. Twenty-nine passed the predefined first-token-logit gate on both requests: same forbidden-token mask, no NaN/+Inf, same argmax and `rtol=0.001`, `atol=0.05` on unmasked logits. The excluded five were disabling Heretic fusion, target grids 72 and 96, KV Q8, and prefill tile 64x64 with four warps. Their short texts were identical despite exceeding the score tolerance; this is not sufficient to call them bugs. Some accepted variants produced different continuations, so passing the gate does not certify output equivalence.

The sweep covered existing fusions and MGEMM, register GDN, presampling, rewind, activation epilogues, graph policies, grid sizes 24/36/48/72/96, attention split multipliers 4/8/16/32, MTP lengths 1/2/3/4 and dynamic policies, KV Q8, CPU thread counts and prefill settings. Screening used 96 output tokens; the selected profile comparison was then repeated with 256. Single screening deltas were not added together or promoted to general performance claims. Most useful options were already enabled by the fork. An environment variable being set does not prove every optional sub-path was exercised.

For the existing quantized GEMM check, from the repository root and configured virtual environment, under CPU/GPU monitoring with the same 96 °C stop threshold:

```bash
env ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library EXL3_GDN_MK=0 EXL3_ABLATION=1 python rocm_tests/test_rdna3_gemm.py "$MODEL_DIR" model.language_model.layers.0.mlp.down_proj
```

`env` selects the recorded runtime data and safe GDN setting for that command. The Python script loads one existing projection, compares reconstruction and matrix multiplication, prints representative errors and returns failure if its checks fail. It does not alter the weights.

## Thermal observations and retained evidence

The owner's CPU/GPU stop threshold remained 96 °C. The initial llama.cpp comparison reached at most 74.25 °C across monitored CPU/GPU sensors. Initial ExLlamaV3 inference reached GPU 76.0 °C and CPU 86.125 °C. The optimization campaign, final profile runs and reference check reached GPU 73.0 °C and CPU 79.75 °C, with at least 30.85 GiB of available system memory recorded. These are separate sampling windows, not a single interchangeable maximum.

No thermal cutoff, OOM or GPU error was recorded in the successful inference/optimization runs. The earlier rocBLAS abort is documented above; the unsafe GDN MK variant was never run. These bounded observations do not establish long-duration thermal stability.

Local evidence retained by the reporter includes full build logs, original per-request JSON, generated token IDs, runtime environments, telemetry, matrix-check output, model hashes and the GPU trace. Sanitized rate rows are supplied in `BENCHMARK_EVIDENCE.json`; the local validation summary maps claims to their evidence. Full private logs are not included in the proposed PR and should be reviewed for personal paths before any optional upload.

## Limitations and maintainer feedback requested

This does not establish equal quality between EXL3 3.5 bpw and GGUF Q4_K_XL; long-context performance or correctness; vision, MoE or DFlash support; all project compute paths; or performance comparable to a Radeon RX 7900 XTX. Reserving 16K cache is not a 16K prompt test. Fixed 256-token continuations are not complete-answer quality evaluations. This report covers a single sequence on one 780M and one checkpoint.

No gfx1100 hardware was available. A narrow gfx1100 compile probe in the build-fix audit must not be interpreted as hardware testing. The system Tensile-path workaround also needs revalidation when changing the wheel or ROCm version.

Maintainer feedback would be useful on the host include-order fix, an appropriate small-GPU eligibility check for the GDN MK grid, and whether a gfx1103 smoke-test recipe belongs in the fork's documentation. No automatic eligibility change is proposed without broader device validation.
