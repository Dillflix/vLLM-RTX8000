# GLM-5.3-Flash on Quadro RTX8000 (experimental)

The development target is symmetric INT4 W4A16 group-128 experts on four
48 GiB Quadro RTX8000 GPUs at 200 W. TP2 stays within each NVLink pair;
PP2 connects GPU0/1 on NUMA node 0 to GPU2/3 on node 1.
A bounded images-only serving profile has passed the capacity, API and workload
checks below. This establishes the tested behavior, not broad model-quality
equivalence or a production soak test. Earlier experimental results are retained
in the later sections.

## Current profile: MTP2, images and 512K input

The selected profile now supports one **524,288-token input plus 1,024 output
tokens**, with TP2/PP2, partition `24,21`, **200 W/GPU**, group-128 W4A16 experts,
a separately quantized MTP layer, **two speculative tokens**, local draft argmax,
prefix caching, images and asynchronous scheduling. Video remains disabled.
The fixed KV allocation is **3,581,476,864 bytes/GPU**: 259 shared hybrid-cache
blocks, 258 usable, with 255 required by this single-context workload. This is
not capacity for five simultaneous 512K requests. The long-text and four-image
tests were separate workloads.

The complete current configuration is in
[`glm53_rtx8000_target.json`](../../examples/deployment/glm53_rtx8000_target.json).
Its `engine_kwargs` object contains engine options; `environment` and `api_flags`
are separate. Replace the model paths and CPU affinity for the installation.
The selected prefill budget remains 512. The pinned Turing compiler described
below is required; changing the compiler requires repeating exactness checks.

Lossless embedding/output-head packing and bounded shared workspaces retain
the loaded weight values. Native tiled indexer scoring and hierarchical exact
top-512 selection avoid materializing the full score matrix. Both FP16-key and
FP8-key chunks preserve their original specialization's FP32 multiply/add
order. FP8 keys are decoded losslessly once into existing FP16 scratch; their
distinct contraction order remains explicit. Native dispatch starts at 128 rows
and 16,384 pooled keys. Smaller chunks and paged decode retain existing kernels.
The scorer is opt-in through `glm_sm75_native_topk`; the example enables it.

The normal SM75 build installs `libglm_indexer.so` beside the Python adapter as
part of `_rtx8000_C`. The standalone `csrc/rtx8000` CMake build does the same.
Rebuild and install that component when enabling this profile; copying Python
files onto an older binary installation is insufficient. Lossless-head kernels
are also included in `_rtx8000_C`. No private prototype-library path is needed.

Relative to the preceding native FP16-key implementation, extending native
scoring to FP8-key chunks reduced the 512K-equivalent scoring/selection case
from 51.8 to 31.3 ms, including conversion. Full 512K prefill changed from
394.7 to 371.9 seconds (1,328 to 1,410 input tokens/s, **6.1%** higher throughput).
Single-user decode at 512K remained approximately 70 output tokens/s. The
full-context comparison is one run per version with differing warmup histories.
Three matched short-context decode repeats did not reproduce an earlier apparent
slowdown; these short sequential cohorts do not establish a decode speedup.
Five-user decode likewise has no demonstrated gain from this prefill change.

Qualification checked 1,489,907 synthetic scores and 269,824 real-model scores
across all four ranks with zero bit mismatches, exact selection, changed-input
graph replay, MTP verification, image inputs, functional/logprob checks and
512K cold/reused-prefix retrieval. Scratch remained 125.6 MiB/GPU, with no
allocation retries or OOMs in the final run. These checks do not establish broad
model-quality equivalence, long-term hardware stability or a production soak.
The [compact measurement record](../../benchmarks/kernels/results/glm53_sm75_native_indexer.json)
preserves the timings and their limitations. The sections below retain earlier
qualification history; they do not override this current profile.

## Earlier images, MTP1 and 384K shared-cache profile

The earlier qualified candidate used the pinned main checkpoint below and the
separate INT4 group-128 MTP artifact, TP2/PP2 with partition `24,21`, 200 W,
one speculative token, local draft argmax, prefix caching, asynchronous
scheduling and a 512-token prefill budget. Images remain enabled at their native
processor settings; video admission is explicitly disabled. The configured
image-count limit remains 999, but that is an admission ceiling, not a claim
that 999 maximum-resolution images fit together.

Cache allocation increases from **2.5 to 3 GiB per GPU**, a **20%** increase,
with 233 physical shared hybrid-cache blocks instead of 194 (232 versus 193
available after the reserved null block). The per-request limit
is `394240`: **393216 input tokens (384K) plus 1024 output tokens**. This is a
larger tested request limit; it does not imply that the physical pool grew 50%,
or that five requests can each occupy that entire window. The explicit
`kv_cache_memory_bytes=3221225472` fixes the pool size and overrides automatic
KV sizing from `gpu_memory_utilization`. Cache blocks are still shared,
allocated and reused dynamically among requests. Extensible allocation is off.

Two opt-in changes reduce memory pressure without additional weight rounding:

- `glm_sm75_lossless_embedding` stores input-embedding dictionary indices in
  12 bits and reconstructs the original FP16 words exactly. It saves **151.24
  MiB on each GPU**, including the draft embeddings on the second pipeline
  stage. This is lossless storage, not 12-bit numerical quantization. The
  output head is unchanged. All 634,388,480 checkpoint elements were compared
  exactly across both TP shards and four kernel tile sizes. The original GPU
  parameter is released. The feature fails closed outside the tested
  SM75/FP16/GLM/TP2/PP2 configuration, or if a dictionary exceeds 4096 entries;
  LoRA, CPU offload, tied embeddings and direct weight reload are unsupported.
- `glm_sm75_stream_vision` transfers complete images from CPU in bounded
  groups and writes the output directly into preallocated storage. It preserves
  every image patch, spatial-attention sequence and processor setting. Groups
  of at most 32768 patches retain the original single encoder call. Larger
  groups change GEMM batch shapes and can therefore change FP16 rounding;
  whole-model bitwise equivalence is not claimed. An individual image larger
  than the chunk budget remains whole.

Both flags default to false. The selected profile enables them, retains the
existing native dense path, and leaves `glm_native_dense_mtp` and
`glm_sm75_shard_encoder_cache` false. The unqualified direct-indexer-gather
prototype is not part of this profile. Piecewise graphs cover rows 1 through
10; FULL graph execution remains unqualified. No native extension rebuild was
needed for these two memory changes; all eleven loaded library hashes match
the reference.

The following checks passed with MTP, images and prefix caching enabled:

- A cold **384K + 1024** request and its repeated-prefix request, including
  retrieval. The repeat reused **391680 prompt tokens**. Complete-request time
  was 278.45 seconds cold and 22.30 seconds with reuse; these include 1024
  generated tokens and are not pure prefill measurements.
- One **256K + 1024** request together with **four 24K + 512** requests.
  Scheduler observations confirmed all five were fully prefetched and resident
  simultaneously. All retrieval checks passed, with no request preemptions.
- Nineteen real OpenAI-compatible API cases: auto/required/named tools,
  streaming and nonstreaming, tool-result round trips, reasoning separation,
  image color, OCR, multi-image order and image-sensitive prefix reuse.
  Changing the image correctly invalidated reuse; repeated images reused
  4352 prompt tokens. Four 2240-by-2800 images retained **8000 image tokens
  each**, with a verified total prompt length of 32037 tokens. Video requests
  returned HTTP 400 as configured.
- Six focused public regressions, plus the checkpoint-scale exact embedding
  component gate. Empty embedding inputs now return before the zero-byte
  collective, which previously left a CUDA invalid-value error pending.

Capacity runs force the requested output length with `ignore_eos` and matching
minimum/maximum token counts. Retrieval must succeed in the first 64 output
tokens; the subsequent continuation exercises cache allocation and is not a
quality benchmark. Cold versus reused-prefix numerical differences remain
recorded rather than treated as exact output equivalence.

The matched warm text workload used 8192 input tokens and 128 output tokens
per request, with two repetitions. Prefill below is effective aggregate input
rate through first responses; decode uses the common interval after all
requests have received their first output burst and excludes prefill.

| Users | Cache/GPU | Aggregate prefill tokens/s | Aggregate decode tokens/s | Average per-user decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5 GiB reference | 2073 | 54.4 | 54.4 |
| 1 | 3 GiB candidate | 2085 | 55.1 | 55.1 |
| 5 | 2.5 GiB reference | 2196 | 145.1 | 29.0 |
| 5 | 3 GiB candidate | 2198 | 147.2 | 29.4 |

These observations show no clear warm performance regression, rather than a
demonstrated speedup. Generated sequences can differ even between repetitions
of the reference, so they are not matched-route causal kernel measurements.
Image timings were also broadly unchanged. The reference's completed text and
image timing observations are retained, although that run later failed a
120-frame video semantic check. Its overall failure status is preserved; video
is outside the user-selected scope of the current candidate.

No OOM occurred. Each second-stage GPU made one additional allocator retry on
the first 384K request, with no further retries in the mixed-capacity workload.
At the tightest observation, driver-free memory was approximately 8–10 MiB,
while approximately 212–214 MiB remained in wholly inactive ordinary allocator
segments. This is a tight, measured configuration, not a guarantee for every
combination of long prompts and images. The 384K/mixed text capacity tests and
four-native-image API test were separate workloads. The first-use retry and
shape-specific JIT compilation should be included in deployment warmup rather
than hidden by warm timing averages. Broad accuracy and sustained-load
evaluation remain separate from the passed functional and capacity checks.

See `examples/deployment/glm53_rtx8000_target.json` for the complete selected
engine arguments and environment. This profile was tested through a temporary
loopback API; it has not replaced the installed service.

## Checkpoint and placement

The initial checkpoint is
[`wtdcode/GLM-5.3-Flash-AWQ-W4A16`](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16/tree/abd7b07719111f137e1de8a0c1b7e01c11b74d1a),
pinned to `abd7b07719111f137e1de8a0c1b7e01c11b74d1a`.
Its quantization method is `compressed-tensors`, not the legacy `awq` loader.
Non-expert BF16 weights are loaded as FP16. The main-model loader skips the
unquantized MTP layer when speculation is disabled.

Audited tensor payload, before runtime overhead:

| Contents | GiB |
| --- | ---: |
| Main routed experts, including scales | 146.180 |
| Other main-model weights | 16.617 |
| Vision weights | 1.050 |
| Unquantized MTP layer | 13.844 |
| Main model plus vision, no MTP | 163.847 |

Quantizing the MTP experts with the same INT4 group size would reduce total
checkpoint payload to approximately 167.672 GiB, including vision. This is
a storage estimate. A separate round-to-nearest INT4 group-128 MTP artifact has
since been converted and exercised as described below; it is not a broad
accuracy-qualified checkpoint.

Start placement experiments with `VLLM_PP_LAYER_PARTITION=24,21`, which assigns
21 routed-expert layers to each pipeline stage. Individual 48 GiB limits,
replicated parameters, cache/state allocations, vision placement, MTP placement,
temporary repacking, activations and CUDA contexts must be measured per rank.
Aggregate weight size alone is not a fit test.

## Optional native dense projections

`--additional-config '{"glm_native_dense":true}'` selects the experimental
FP16 dense kernels for the measured GLM attention projections and M1 output
head. It defaults to `false` and requires SM75, TP2/PP2, compilation mode 0,
`linear_backend="auto"`, and no LoRA. Selected output projections cover M1-M5;
other rows, shapes, dtypes, biased or noncontiguous inputs, and batch-invariant
execution retain their original linear method.

Rebuild the normal `_rtx8000_C` extension from this source before enabling it.
The M3 vector specialization is part of that extension; no separate library,
local artifact path, or binary hash is required. The existing Qwen dispatch
plans remain unchanged, including its M3 fallback. The integrated library passed
six expanded native/fallback tests and 22 actual-M3 comparisons, including
changed-input graph replay. Existing vector kernels and the new M3 kernel have
the same canonical machine instructions as their qualified references. The
paired performance experiment below used the separate M3 prototype. The
ordinary extension also passed the final TP2/PP2 integration smoke described
below; no prototype library is required by the implementation.
This option does not change expert quantization or accumulation policy.

The additional `glm_native_dense_mtp` option extends this path for MTP1. It
defaults to false and requires `glm_native_dense`, one native MTP draft token,
and the same SM75/FP16/TP2/PP2/configuration guards. Target output projections
use the 256-thread vector kernel at six rows and the 128-thread kernel at eight
rows. The draft's MLA output projection uses vector kernels at two, four and
eight rows. Existing target plans, merged KDA inputs and the shared output head
retain their previous selections. Ten-row verification retains its standard
linear method; unsupported shapes, strides, alignment, bias and batch-invariant
execution also retain the existing fallback. Draft weights, quantization,
embedding ownership and the shared target head are preserved.

The selected plans were measured using 35 input fixtures saved from actual
target and draft graph replays for one through five users. Their active rows
equalled the graph rows; dummy capture values were not used. The original
screen passed 427 fixture/variant checks, and the added vector/split-K screen
passed 315. Changed-input graph replay and poisoned/shared partial buffers
were checked without widening numerical tolerances. Repeated output-projection
measurements improved target six-row operations by approximately 3.2–5.8%,
target eight-row operations by 4.6–8.2%, and selected draft operations by
4.0–8.4%. These are operation-level results, not whole-model speedups. The new
ten-row vector kernel and six deterministic split-K Tensor Core plans did not
consistently improve performance and were excluded. Draft six-row gains were
also insufficiently consistent for selection; KDA input gains were marginal.

Only the six-row vector instantiation is added to the ordinary RTX8000 native
extension, through the existing shared variant list and a separate compilation
unit. All 28 existing kernels retain identical encoded instructions. The new
kernel matches its measured prototype and passed 14 real-input cases with 42
changed-graph checks in the ordinary namespace. The final public regression
gate passed 11 real-configuration/layer/native cases and eight gather-bound
cases, including changed-input graph replay, shared-head ownership, quantized
layer exclusion, fallback behavior and exact bound/overflow checks. Rebuild the
ordinary extension before enabling this option; the private experiment libraries are
not runtime dependencies. Full-model results for the composed profile are
recorded with the cache-capacity qualification below.

## Compatibility implementation

Compiler selection is material. Stock Triton 3.8 in the existing environment
compiled the FP16 indexer, KDA and sparse MLA matrix operations to scalar SIMT
instructions on SM75. Merely using FP16 operands did not enable Tensor Cores.
The isolated [Turing compiler](https://github.com/Chennesxu/triton-turing/tree/8da2085caa00c79cf7ce99c7927bb72135e36af1)
at `8da2085caa00c79cf7ce99c7927bb72135e36af1` (Triton 3.7) emits
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` for the qualified matrix
kernels. The indexer passed 43 GPU cases, sparse MLA passed 18, and the
pooling/query/KDA suite passed 29 with this compiler. The existing production
environment was not modified. Reduced random-weight models also passed the
real engine's TP2/PP2 prefill and decode paths, including concurrent prompts
longer than 2048 tokens that exercise sparse selection. The full checkpoint
has since passed the text checks and offline long-context workloads below;
general accuracy and production serving remain unqualified.

The pipeline boundary completes the deferred mHC residual update and transfers
the expanded streams. The receiver must retain all streams, not contract them
into an ordinary hidden vector. Missing-stage weights are skipped before FP8
dequantization helpers inspect target parameters.

SM75 FP16 mHC defaults to the PyTorch reference computation. The optional
`--additional-config '{"glm_mhc_backend":"turing"}'` path fuses gates,
Sinkhorn iterations, stream contraction and residual post-mixing, while keeping
the FP32 cuBLAS projection, original FP32 RMS reduction and existing native
RMSNorm. Gate and residual-mixing arithmetic remain FP32, with the reference's
FP16 rounding boundaries preserved. Selection is cached when the operators are
constructed and requires SM75, FP16 GLM and four residual streams. An initial
candidate failed the engine comparison because approximate softmax/Sinkhorn
math changed the FP32 mixing matrices. The corrected implementation uses
precise exponentiation/division and the reference's column-reduction order.
All 25 public-operator, numerical-reference, configuration and graph-replay
checks pass, including strengthened exact mixing-matrix checks. Replaying eight
captured real operations matches every reference output bitwise. The fresh
TP2/PP2 engine pair also passed its original gates: all eight captured final
hidden tensors and generated tokens match exactly, with maximum sparse-prompt
log-probability difference of 9.54e-7. These reduced-model checks do not establish
full-checkpoint quality. The real-checkpoint candidate below also passes its
functional checks, but the default remains `"torch"` while broader accuracy
and serving qualification continue.

The corrected-source synthetic fused post/pre operation at H4096, four streams
and 20 Sinkhorn iterations produced identical outputs for these fixtures:

| Tokens | Eager reference / Turing (ms) | Graph reference / Turing (ms) |
| ---: | ---: | ---: |
| 1 | 3.182 / 0.668 | 0.474 / 0.057 |
| 5 | 3.212 / 0.700 | 0.543 / 0.073 |
| 512 | 4.828 / 0.851 | 2.382 / 0.796 |
| 2048 | 7.351 / 3.032 | 7.354 / 3.017 |

These are isolated operation measurements, not serving results. Eager timings
include host launch effects; graph timing does not qualify model graph execution.

`TRITON_MLA_SPARSE_SM75` is an explicitly selected FP16 NoPE-512 sparse MLA
backend. It preserves the padded index buffer and incomplete-pool tail, maps
request-local indices through paged/interleaved cache layouts, and supports
single-pass and split-KV computation. The initial 32-key split tile exceeded
SM75's 64 KiB shared-memory limit; the 16-key tile passes the numerical tests.
A 40-case synthetic profile at 200 W used 32 query heads, NoPE-512, 2176 padded
indices and 32K cached rows per request, including physical and interleaved pages.
All eager and graph checks passed an independent FP32 softmax reference.
Automatic splitting selected 16/4/1/1 splits for 1/5/512/2048 queries. Warm graph
times were approximately 0.04/0.15/6.30/24.51 ms for index conversion plus one
attention layer; eager times were 0.52/0.51/6.36/24.50 ms. Splitting large prefill
further increased latency. These are kernel measurements, not serving rates;
the existing autotune key also omits query count. The selected four-warp,
single-stage prefill kernel uses 166 registers per thread and 33,280 bytes of
dynamic shared memory, just over the 32 KiB boundary that would permit two
resident blocks on SM75. Reducing this footprint is a candidate for further
tuning, without changing numerical arithmetic. Full-decode graph support remains
disabled for this backend; the partial-graph qualification below preserves eager
attention boundaries.

An isolated 32-head tile with four warps and one stage passed numerical and
graph checks, including interleaved pages. Its interleaved-cache graph times
were 3.78 ms at 512 queries and 14.08 ms at 2048, versus 6.26 and 24.73 ms for
the current 16-head tile. Contiguous single-query decode regressed from
0.040 to 0.058 ms; interleaved single-query decode was effectively unchanged.
This candidate uses 249 registers/thread and 50,176 shared-memory bytes, so
the gain does not come from resolving the shared-memory residency limit.
The optional `glm_sm75_prefill_h32` additional-config flag now selects this
tile for actual per-request prefill spans of at least 480 query rows. It uses
four warps and one stage and requires 32 heads with NoPE-512. Decode and smaller
spans retain the original tile, including in mixed batches. The flag defaults
to false while broader context and serving qualification continue.

The backend also rejects prefill context parallelism and adaptive verification
with differing CPU/device query boundaries: its inherited metadata builder
constructs request IDs from CPU lengths. Neither restriction prevents TP2/PP2
or fixed-depth MTP by themselves. The bounded fixed-depth MTP1 qualification
below does not establish all draft depths or production serving support.

The initial indexer keeps the checkpoint implementation's FP8 cache layout.
Turing has no native FP8 arithmetic: GPU integer operations encode/decode the
cache bytes, while matrix multiplication uses native FP16 Tensor Cores with
FP32 accumulation with the qualified Turing compiler. This preserves a
reference path; it is not a claim that
FP8 storage is optimal. Compare FP16 queries and FP16 cache against FP16 queries
and compressed FP8 cache, including conversion cost, memory and routing quality.

`--additional-config '{"glm_indexer_fp16_queries":true}'` enables the qualified
FP16 query producer and logits path on SM75; its default is false pending
full-model routing/quality validation. The existing compressed FP8 key cache
is retained. A fully FP16 indexer key cache is not implemented yet.

At 32 indexer heads, dimension 128 and 32K pooled keys (128K input context),
the warm logits microbenchmark measured these CUDA-event latencies:

| Workload | FP8 queries, eager | FP16 queries, eager |
| --- | ---: | ---: |
| 1 decode query | 0.199 ms | 0.196 ms |
| 5 decode queries | 0.292 ms | 0.192 ms |
| 512 prefill tokens | 18.816 ms | 5.370 ms |
| 2048 prefill tokens | 54.954 ms | 24.843 ms |

These cover the logits operation and submission gaps, excluding query
preparation, top-k selection and the rest of the model. They are not per-user
token rates or a serving benchmark, and the paired inputs represent the same
query values to isolate storage/decoding overhead.

A separate matched key-format experiment kept FP16 queries and identical
decoded keys/scales. Persistent FP16 keys improved large prefill but slowed
single-query decode at 32K pooled keys. The key/scale footprint rises from
4.125 MiB to 8.125 MiB, crossing this GPU's 6 MiB L2 capacity; no hardware
cache-hit counters were collected, so this is a plausible contributor rather
than a measured causal attribution.

The better prefill candidate retains the persistent FP8 cache and converts
gathered keys once into temporary FP16 storage. Including that conversion,
warm graph logits time improved from 1.640 to 1.174 ms at 512 queries/8K keys,
and from 24.412 to 19.862 ms at 2048 queries/32K keys. Eager calls included the
scratch allocation and showed similar gains. The four tested prefill shapes
improved by approximately 16–28%, using 2 or 8 MiB of temporary FP16 keys.
All comparison outputs matched the compressed-key path exactly, and GPU
conversion matched all 254 finite E4M3 encodings bit-for-bit. Tensor Core
instructions were retained. These measurements exclude cache gathering, top-k
and the rest of the model.

The optional `glm_sm75_prefill_fp16_keys` additional-config flag integrates
this approach for actual per-request prefill spans of at least 480 query rows.
It reserves 256 additional bytes per gathered pooled key during memory
profiling, converts keys lazily after a gather, and reuses the converted keys
across eligible subwindows until the next gather invalidates them. Persistent
cache bytes and scales are unchanged; FP16 query selection is independent.
This flag also defaults to false. Across the updated MQA and sparse MLA suites,
78 cases passed, including FP8/FP16 queries, actual metadata construction,
mixed batches, locked workspace reuse, repeated gathers and graph replay.
The full-checkpoint singleton probes below verify that both optional paths
actually execute on all four ranks and return the correct retrieval answer.

The sparse MLA algorithm and PP boundary handling were informed by
[`wtdcode/vllm-backport` at `29e66dad`](https://github.com/wtdcode/vllm-backport/tree/29e66dad461b11caae6039f510eaa7913f9dfb19).
The donor's BF16/Ampere paths are not evidence of SM75 support.

## Integration and accumulator qualification

The reduced-model PP comparison used identical source files and verified all
99 parameter tensors per TP rank after merging the pipeline stages. Final
hidden states for a short request were bitwise identical between TP2/PP2 and
TP2/PP1, including prefill and three decode steps. Sequential request logprobs
also matched exactly. One concurrent-request logprob differed by 0.000244140625,
which failed the original 1e-5 absolute tolerance. A separate PP1 control
reproduced exactly that difference when changing batch shape. The controlled
comparison passed; this does not establish batch-invariant output or actual
checkpoint accuracy.

A separate four-layer test exercised the real conditional-generation wrapper,
the checkpoint's compressed-tensors configuration and ignore rules, and a
288-expert/top-8 layer with deterministic synthetic packed INT4 weights.
Normal Marlin repacking and TP2/PP2 execution passed concurrent 2305/2561-token
requests. Worker reports confirmed group-128 Marlin and the requested CPU
affinities. This qualifies the quantized integration path, not the downloaded
weights or the full model's memory fit.

The installed Marlin INT4 group-128 kernels use FP16 Tensor Core accumulation,
followed by FP32 split-K reduction. An isolated candidate changes the Tensor
Core accumulation to FP32 without replacing the installed extension. A matched
288-expert, TP2-shard benchmark produced these warm CUDA graph timings:

| Tokens | Installed FP16 MMA | Candidate FP32 MMA |
| --- | ---: | ---: |
| 1 | 0.1515 ms | 0.1516 ms |
| 5 | 0.5167 ms | 0.5166 ms |
| 512 | 4.6157 ms | 4.5483 ms |
| 1024 | 7.2251 ms | 7.3309 ms |
| 2048 | 11.2208 ms | 11.7670 ms |

Both candidates passed eager and graph numerical checks against a dequantized
PyTorch reference. FP32 MMA reduced relative numerical error by approximately
4.4–6.3 times. These are synthetic expert-layer measurements at 200 W, not
end-to-end rates or model-quality results. The existing accumulator remains
the default: the untuned FP32 candidate has no measured decode-speed advantage
and is approximately 5% slower at 2048 tokens. Only the automatically selected
tiles were numerically benchmarked; other compiled variants remain unqualified.

## Required context and cache capacity

The production requirement is **at least 262,144 input tokens**, with generation
headroom in addition. The current qualification profile therefore sets
`max_model_len=262400`, reserving 256 output tokens. The checkpoint advertises
1,048,576 positions; 128K was an earlier test-profile choice, not a model limit
or the production target. Prefix caching is also required. Historical 128K
profiles and results below remain evidence for those exact configurations.

A CPU calculation using the current planner estimates 130 cache blocks for one
262,144-input/256-output request with aligned prefix caching, plus one shared
null block. With the measured TP2/PP2 geometry this is 1,807,825,152 bytes on each
first-stage GPU and 1,506,520,960 bytes on each second-stage GPU. This calculation
does not establish runtime fit by itself. The real-model qualification with a
fixed 2 GiB KV budget per rank and extensible allocation disabled initialized
155 blocks. It accepted exactly 262,144 input tokens and generated 256 output
tokens, correctly retrieving the midpoint marker. The repeated request reused
261,120 cached tokens and produced identical token IDs. Cold/repeated wall
times were 161.01/8.46 seconds; the cold phase included first-use compilation and
low-priority CPU conversion, so it is capacity evidence rather than a controlled
warm-performance result. Generation was forced to 256 tokens to exercise output
headroom; continuation after EOS is not generative-quality evidence.

This profile is not yet production-qualified. Both first-stage GPUs needed one
recoverable allocator retry (zero final allocator OOMs), reaching 46.65 GiB
reserved versus 43.95 GiB peak live allocation. Telemetry peaked at 48,340 MiB.
Cached/uncached outputs had matching token IDs on the marker tests but significant
log-probability differences: up to 2.344 among shared top-five tokens for the
256K test. A same-engine zero-cache-hit short-prompt control also differed, so
prefix reuse has not been isolated as the cause and numerical equivalence is
not established. Recurrent-state/checkpoint comparisons remain necessary.

The optional `glm_sm75_mqa_scratch` flag reserves a bounded, contiguous FP32
logits buffer in the existing workspace and reuses it across prefill chunks.
It defaults to false. The buffer is disjoint from the live gathered keys and
scales; runtime rows cannot exceed the profiled bound. The kernel arithmetic
and static/dynamic dispatch are unchanged. All 33 GPU component checks passed,
including bit-exact output, changed-input graph replay, poisoned-buffer reuse,
invalid-output guards and the actual 512-by-65,600 bound.

The otherwise identical full-model 2 GiB/512-token/256K profile passed all six
capacity/retrieval cases with this flag. Reused-prefix tokens stayed at 261,120;
allocation retries and final OOMs were zero on all four GPUs. First-stage peak
reserved memory fell from 46.654 to 46.086 GiB (582 MiB per GPU), while the
persistent scratch allocation increased live storage by 128 MiB. Generated
token IDs were not identical in every case, so this establishes the observed
memory improvement and functional checks, not whole-model numerical equivalence.
The earlier cold run also included compilation and CPU conversion; its wall
time is not a controlled performance baseline for this allocation change.

The separate `glm_sm75_bound_gather` option, also disabled by default, bounds
gathered indexer keys by the configured number of requests times each request's
maximum compressed length. It requires SM75 FP16, compressed GLM cache and
PCP=DCP=1. It retains the existing smaller bound when applicable and rejects
overflow before gathering; it does not truncate keys or change attention math.
At `max_model_len=263168`, five requests and compression factor four, this
reduces reserved gather rows from 1,052,672 to 328,960 while retaining the
65,792-key maximum per request. Combined with the logits scratch option, the
indexer requests 251.22 MiB instead of 519.02 MiB per workspace slot. Native
MTP target and draft share that slot; the saving is not doubled on the draft
stage. Eight GPU component checks and 650 CPU planner cases passed. Actual
allocator savings and full-model MTP capacity are measured separately below.

The initialized independent-request geometry permits five 32K requests, three
64K requests, two 128K requests, or one 256K request at this budget. One 256K
request leaves 24 usable blocks, enough geometrically for one additional 8K
request. A subsequent fixed 2.5 GiB pool initialized 194 blocks and passed an
actual mixed-context stress test: one 262,144-input/1,024-output request plus
four independent 8,192-input/128-output requests. `max_model_len` was 263,168;
the scratch option was enabled and MTP was disabled. A read-only scheduler
observer verified that all five were simultaneously fully prefetched and
generating, with output counts 1/98/76/55/34 at that moment. All five retrieved
their distinct markers, had zero prefix hits, and completed without preemption
or fatal OOM. Each first-stage GPU recovered one allocation retry while creating
the cache pool, with no additional runtime retries. Peak live/reserved allocation
was 44.53/46.06 GiB on stage 0 and 43.33/45.29 GiB on stage 1. This forced-output
stress is capacity evidence, not a serving-rate or broad quality benchmark.
That run did not combine 2.5 GiB with MTP; the later combined qualification is
documented below. Vision remains unqualified. Shared prefixes have
different block-sharing requirements; independent-request counts are not their
capacity limit.

Five unrelated requests at that maximum length require an estimated 651 blocks,
or about 8.37 GiB per first-stage GPU for the cache alone. This is distinct from
five shorter requests or a mixed workload containing one long request. Admission
limits must come from the actual initialized cache and leave room for outputs;
`max_num_seqs=5` does not promise five simultaneous 256K requests.

The supplied Blackwell recipe allocates 2.25 GiB per GPU and uses `fp8_ds_mla`.
This SM75 path currently stores the MLA latent cache in FP16, pooled indexer keys
in FP8, and KDA recurrent state in FP32. TP2/PP2 also partitions layers across
pipeline stages, unlike that recipe's TP2-only configuration. Matching aggregate
VRAM therefore does not establish matching cache capacity or performance.

### Fixed allocation and comparable throughput metrics

`kv_cache_memory_bytes` sets a shared cache/state budget per GPU. The block
manager still allocates, shares, evicts and reuses pages between requests;
this setting does not reserve a full context for every user or impose a
128K limit. It overrides the automatic cache budget derived from
`gpu_memory_utilization`. This profile also disables extensible KV allocation,
so changing the pool requires a restart. Explicit sizing makes the capacity
reproducible, but activations, graphs, temporary buffers, MTP and vision still
need separate headroom. Increasing the pool can exhaust that headroom.

Do not compare decode-only output rates with end-to-end output rates that
include prefill. In the historical dense/threshold experiment, the candidate's
one-user complete-request rate was 18.16 tokens/s and its five-user aggregate
complete-request rate was 25.66 tokens/s. Its approximately 35.55 tokens/s
single-user decode rate used a different denominator. Those numbers do not
show aggregate decode dropping when five users are present.

The newer 256K-capable, prefix-enabled profile produced the following matched
8K-input/128-output workloads at 200 W, with async scheduling and 512 tokens per
step. Each cold-prefix row averages two warm-kernel repetitions after cache
reset. Decode is measured only within the interval after all users receive
their first token and before any user finishes.

| Workload | Complete-request output rate, aggregate | Decode-only output rate, aggregate | Decode-only rate per user |
| --- | ---: | ---: | ---: |
| One user, cold prefix | 17.00 tokens/s | 35.60 tokens/s | 35.60 tokens/s |
| Five users, cold prefixes | 26.76 tokens/s | 110.97 tokens/s | about 22.19 tokens/s |
| Five users, reused prefixes | 56.59 tokens/s | 87.49 tokens/s | about 17.50 tokens/s |

The reused-prefix row is one observation with 6,528 cached tokens per request.
Its decode requests split into different pipeline cohorts from the cold-prefix
case, so the decode difference cannot be attributed to caching alone. During
other users' prefills, the cold-prefix run also had substantial gaps: first-token
times were approximately 4.23, 7.84, 11.50, 15.13 and 18.51 seconds. These user
arrival measurements include frontend/IPC overhead and are not GPU kernel rates.
Generated sequences may differ between runs; the rates are workload observations,
not isolated causal kernel speedups or quality-equivalence evidence.

An otherwise matching async-off control reduced single-user decode to 20.46
tokens/s and five-user cold-prefix end-to-end output to 25.91 tokens/s. Async
scheduling remains selected. The baseline engine had previously exercised the
256K request and the control had not; their allocator histories differ.

A benchmark-only scheduler prototype coalesces requests onto one PP2 decode
phase once all their prefill work has been scheduled, delaying an eligible
request by at most one scheduler step. Prefill work can still be in flight:
the measured trace includes one request with 651 queued tokens at alignment.
It retains the two-step PP stagger and declines speculation, unscheduled
prefills, structured output and cohorts larger than five. Fifteen real-scheduler CPU
tests cover delayed/reordered output, abort, preemption and request identity.
The real TP2/PP2 engine checked 635 received-token/input mappings on each
first-stage rank, including 600 at batch five; the observers were removed before
timing. All six functional checks passed.

For the identical reused-prefix five-user workload, graph cohorts changed from
mostly batches two and three to batch five. Aggregate decode rose from 87.49 to
111.31 tokens/s (22.15–22.34 per user), and complete-request output rose from
56.59 to 65.61 tokens/s. Single-user decode remained 35.61–35.67 tokens/s and
cold-prefix five-user decode stayed near 111 aggregate. First-token times in the
cached workload were essentially unchanged. This is one cached observation per
engine, with different preceding workloads and sometimes different generated
routes; it is not a matched-route causal kernel benchmark. The prototype remains
outside the core scheduler and does not change the default policy. Allocation
retry and final OOM counters were zero after this control.

This decode-cohort result does not resolve long-prefill fairness. In the 2.5 GiB
mixed-capacity stress, all five requests were submitted within 41 ms, with the
256K request first. The four 8K requests reached their first tokens at roughly
24/53/79/108 seconds. The current scheduler distributes the 512-token budget
in running-request order; the uncapped long request can consume it before later
requests. KDA alignment occasionally leaves spare capacity. This is a
source-supported explanation, not a per-step causal trace of that run.
An existing adaptive long-prefill threshold can share the budget across active
requests, but five-way sharing reduces spans to about 102 tokens and loses the
current prefill fast paths' 480-row threshold. Rotating full-size prefill chunks
is another candidate that may retain kernel efficiency. Neither policy has yet
been performance-qualified with this mixed-context workload.

### Historical cache and prefix audit

The full-weight 8K baseline committed 127 cache blocks: approximately 1.63 GiB
per GPU on pipeline stage 0 and 1.36 GiB on stage 1, using the 24/21-layer
partition. Its reported **115,598 cache tokens is not a hard context ceiling**.
The reporting function multiplies maximum-length request capacity by
`max_model_len`: `127 / (4 KDA states + 4 MLA pages + 1 tail) * 8192`.
Fixed recurrent-state overhead makes that normalized token count depend on the
configured request length.

A CPU-only calculation using the actual planner, the full checkpoint's layer
counts and reconstructed baseline cache specs gives the following geometry.
It holds the measured 127-block pool fixed and does not qualify longer-context
execution, activation peaks or a later launch's cache budget. MTP is disabled.

| Context per request | Blocks, prefix off | Blocks, prefix align |
| --- | ---: | ---: |
| 8K | 9 | 13 |
| 32K | 21 | 25 |
| 64K | 36 | 40 |
| 128K | 66 | 70 |

One shared null block is additional. Thus five 32K requests with prefix
alignment need `5 * 25 + 1 = 126` blocks against 127, **as allocation geometry
only**, with essentially no spare block capacity. At 128K, one aligned request
plus the null block occupies approximately 0.913/0.760 GiB per GPU on stages
0/1. Retained prefixes, concurrent requests and runtime workspace still require
validation. Extensible-cache sizing commits its final budget after warmup; it
does not establish unlimited later growth.

The prefix-disabled dummy's 69,632-token scheduling LCM, and the full 8K run's
139,264-token LCM, are not enabled-prefix hit boundaries. Enabling caching puts
KDA into `align` mode and makes KDA/MLA blocks 2,176 tokens. Matching defaults to
that size. The existing `--prefix-match-unit 128` or `32` produces finer hit
alignment in the CPU resolver and respects the indexer's four-token pooling;
the non-cacheable tail ring does not constrain matching.

Fine matching does not retain a recurrent state at every hash boundary.
Default retention is semantic checkpoints only (`prefix_cache_retention_interval=0`),
including the final prompt boundary. Triton KDA currently requires a scheduler
split there because it cannot export an internal prefill checkpoint. Identical
prompts and extensions therefore have an existing path to qualify. Arbitrary
shared-prefix junctions need separate tests: the optional Mamba junction feature
is currently gated on EAGLE/MTP group presence and no multi-module re-prefill
tail, so enabling its flag alone does not activate it for this no-MTP baseline.

## Selective INT8 output-projection experiment

A separate checkpoint/source candidate quantizes the 11 MLA output projections
to symmetric per-channel INT8. All 34 KDA output projections remain FP16, and
the routed experts retain INT4 W4A16 group-128. Per-rank ownership and parameter
accounting were checked after real TP2/PP2 loading: six selected projections
on each first-stage rank and five on each second-stage rank. Persistent
parameter storage falls by 191.953 MiB per first-stage GPU and 159.961 MiB per
second-stage GPU, approximately 703.828 MiB across the system.

The small-row path uses a private INT8 Marlin specialization with FP32 Tensor
Core accumulation. The stock FP16-accumulation W8 path failed the fixed large-
prefill numerical tolerance; the FP32 specialization passed it without widening
that tolerance. This choice is specific to this candidate. It does not change
the INT4 expert accumulator or establish a universal accumulation policy.

For total batches of at least 508 rows, a tiled unpack converts the canonical
packed weights into a reusable INT8 workspace, followed by per-token activation
quantization and native SM75 CUTLASS W8A8. The conversion is included in kernel
timings; there is no permanent second copy of the dense weights. Decode rows
inside such a mixed batch also use W8A8. Smaller batches use W8A16, so this is
a batch-size policy rather than a strict per-request prefill/decode switch.

Eighteen numerical/graph cases cover 508/512/2048 rows, both tested widths and
three unpack implementations. The selected tiled implementation makes MLA
large-row operations approximately 1–6% faster at 508/512 rows and 28–29% faster
at 2048 rows, including conversion. KDA at 512 rows regressed materially,
which is why the first model candidate preserves all KDA projections in FP16.
These isolated measurements do not establish whole-model gains. The separate
full-model candidate retains 262,400 maximum context, a fixed 2 GiB KV budget,
512 tokens per step, prefix caching, TP2/PP2 and 200 W, without MTP or the new
logits scratch option. It is compared with the matching earlier baseline;
composing it with MTP or changing the prefill budget needs a separate gate.

That full-model gate passed six functional checks, cold/cached 262,144-input
plus 256-output retrieval, four short stale-prefix checks, and all seven
one/five-user stream phases. The long repeated request reused 261,120 tokens.
Untimed checks on every rank compared actual pure 512-row prefill and mixed
one-decode/511-prefill operands against independently unpacked checkpoint
weights. Packed bytes and scales matched exactly. Arithmetic relative RMS error
against the quantized integer oracle was 0.00020–0.00035%; total weight-plus-
activation quantization error against the original local FP16 projection was
0.91–1.82%. These are distinct checks: correct quantized arithmetic does not
make the quantization lossless. The diagnostic hooks restored before timing.

| Warm 8K-input/128-output workload | Original projections | MLA INT8 candidate |
| --- | ---: | ---: |
| One user, complete-request time (s) | 7.531 | 7.496 |
| One user, decode arrivals/s | 35.60 | 36.23 |
| One user, first-token latency (s) | 3.963 | 3.991 |
| Five users, complete batch time (s) | 23.916 | 23.863 |
| Five users, aggregate common-window decode arrivals/s | 110.97 | 112.36 |
| Five users, aggregate output/s including prefill | 26.76 | 26.82 |

Each cold-prefix comparison averages two warm repetitions. The cached five-user
observation improved from 11.309 to 11.197 seconds, with 6,528 cached input
tokens per request. These are small workload-level differences, not evidence
of a material prefill gain at the current 512-token budget. Generated routes
and prior untimed allocation work differ; statistical significance is not
established. Whole-model 2048-token prefill with this INT8 candidate has not
been tested. The 256K cold/cached token IDs matched the baseline in this test,
but shared-prefix top-logprob differences reached 2.6875 across profiles, so
broad quality and numerical equivalence remain unqualified. Each first-stage
GPU recovered one allocation retry, with zero final OOMs; weight savings alone
did not eliminate the baseline's long-prefill allocation pressure. This remains
an isolated source/checkpoint prototype rather than a selected default.

## Quantized MTP experiment

The separate draft artifact quantizes all 864 expert matrices in the MTP layer
to symmetric INT4 group-128 using round-to-nearest quantization. Its 39 shards
contain 5,375,586,432 tensor bytes, including a copy of the base embedding needed
on the last pipeline stage. Non-expert draft tensors remain unchanged. The
conversion manifest and every tensor hash were checked before GPU use.

A separate checkpoint requires the draft's own `ModelConfig` and quantization
configuration when constructing its layers. Reusing the target configuration
incorrectly applies its MTP ignore rules and constructs FP16 experts, causing
an avoidable OOM. The corrected constructor uses the existing draft-config
resolver without mutating the target. Under PP2 the drafter also loads its own
TP-sharded embedding; the output head remains shared with the target.

The corrected MTP1 profile retained TP2/PP2, 200 W, 512 tokens/step, aligned prefix
caching, the fixed 2 GiB cache budget and `max_model_len=262400`. It loaded the
target plus draft at approximately 43.06 GiB per last-stage GPU. Six functional
checks, six boundary/prefix checks, and both cold/cached 262,144-input/256-output
retrieval tests passed. The latter reused 261,120 tokens. Across the functional,
boundary and long-context checks, 455 of 533 draft tokens were accepted and 78
verification steps rejected the draft, exercising both paths. These are greedy
spot checks, not a broad quality or speculative-distribution evaluation.

The first matched warm 8K/128 streams improved single-user completion time from
7.531 to 6.338 seconds and after-first-burst output rate from 35.60 to 55.92
tokens/s. Five-user completion time regressed from 23.916 to 27.260 seconds.
That inherited profile only captured token rows 1–5; MTP1 verification at rows
6, 8 and 10 ran outside those graphs. It is not the optimized multi-user MTP
result. Output-token IDs sometimes differed from no-MTP, including divergence
after 18 tokens in the forced long output, after both had already produced the
correct reference value. That forced-tail difference is not an observed answer
regression. The 8K stream continuations also sometimes diverged. Strict greedy
equivalence remains unqualified, even though expected answers and retrieval
passed.

The no-MTP baseline also differs between repeated greedy runs (as early as
10–25 tokens for some five-user topics). That prevents attributing every
cross-profile difference to quantized drafting, but does not itself establish
equivalence or resolve the numerical-review requirement.

Peak live CUDA allocation was approximately 43.95 GiB on the first stage and
45.73 GiB on the last stage. Allocation retries recovered (one per first-stage
GPU and three per last-stage GPU), with zero final allocator OOMs. Temporary
buffer reuse and expanded graph coverage were subsequently qualified together
in a separate full-model run, as follows.

The combined MTP1 profile enables `glm_sm75_mqa_scratch` and captures token rows
1 through 10. All six functional checks, six prefix/boundary checks, and the
262,144-input/256-output cold/cached retrieval tests passed. Controlled cohorts
and naturally scheduled workloads exercised target and draft graph rows
2/4/6/8/10; five-user streams actually replayed row 10 on every participating
rank. Allocation retries and final OOMs were zero on all four GPUs. Peak
reserved memory was 46.09 GiB on the first stage and 45.96 GiB on the last.

The first timing pass accidentally retained a Python dense-operation counter.
Its correctness, capacity and replay observations are retained, but the rates
below come from a fresh repeat that verified the counter's original method
identity and unchanged counts before and after all requests. The model source,
native library, compiler and configuration matched the combined capacity run.
Lightweight graph-replay observers remain, including a draft observer additional
to the no-MTP control. The clean timing repeat also has a different preceding
allocation history because it does not repeat the long capacity probes.

For the same 8,192-input/128-output prompts, two warm repetitions gave:

| Workload and metric | No MTP | MTP1 + scratch + rows 1–10 |
| --- | ---: | ---: |
| One user, complete-request time (s) | 7.531 | 6.243 |
| One user, output/s including prefill | 17.00 | 20.50 |
| One user, decode arrival tokens/s | 35.60 | 55.20 |
| Five users, complete batch time (s) | 23.916 | 22.435 |
| Five users, aggregate output/s including prefill | 26.76 | 28.53 |
| Five users, aggregate common-window decode arrivals/s | 110.97 | 145.83 |
| Five users, per-user common-window decode arrivals/s | 22.19 each | 26.8–31.0 |

The five-user reused-prefix observation, with 6,528 cached tokens per request,
improved from 11.309 to 8.891 seconds overall, from 56.59 to 71.99 aggregate
output tokens/s including prefill, and from 87.49 to 135.59 aggregate
common-window decode arrivals/s. This is one cached observation per profile.
Common-window rates count frontend arrivals between the latest first burst and
earliest last burst, assigning whole boundary bursts at their arrival time.
They are neither isolated kernel rates nor per-token interarrival latency.
Generated continuations differ, so these are matched-workload observations
for the combined profile, not identical-route causal gains or broad quality
equivalence. A subsequent 2.5 GiB mixed-context MTP qualification follows;
deeper speculation and production API serving remain unqualified.

### MTP with the larger cache and native verification kernels

The combined profile uses fixed `kv_cache_memory_bytes=2684354560` (2.5 GiB per
GPU), `max_model_len=263168`, five requests, a 512-token scheduling budget,
prefix caching and one quantized MTP draft token. It enables
`glm_sm75_mqa_scratch`, `glm_sm75_bound_gather` and `glm_native_dense_mtp`,
retaining the earlier native dense and prefill options, TP2/PP2, layer partition
24/21, 200 W and graph rows 1 through 10. This is a text-only offline profile.

All six functional checks and cold/cached 262,144-input/256-output retrieval
checks passed; prefix reuse accounted for 261,120 tokens. The stronger mixed
workload also passed: one 262,144-input/1,024-output request alongside four
independent 8,192-input/512-output requests. Every request retrieved its own
marker and completed its specified output count with zero prefix hits or
preemptions. The scheduler recorded all five fully prefetched and resident
while generating, with output counts 1/177/141/101/61 at that instant. The
initialized pool contained 194 physical blocks; the conservative planner
budgeted 191 for this workload, leaving two after the null block. This does
not imply capacity for five independent 256K requests.

Runtime inspection confirmed the bounded indexer workspace is 251.224 MiB
per GPU, shared by target and draft on the final stage. Peak live allocation
was 45,339.02 MiB on the first stage and 47,164.87 MiB on the second, about
268 MiB per GPU below the earlier unbounded run. Peak reserved allocation was
46,900 and 47,292 MiB respectively. The final-stage GPUs had 497.75 MiB free
after the workload. Each GPU recovered one allocator retry during initialization;
there were zero additional runtime retries and zero allocator OOMs. That
headroom is specific to this profile, not an allowance for untested features.

Controlled one-through-five-request cohorts exercised actual target and draft
graph replays at rows 2/4/6/8/10. The selected native installation remained
unchanged throughout. Separate token-identity diagnostics checked accepted
and rejected drafts on both first-stage ranks, following real request slots
and their generations into the next verification input. All checks passed,
including the five-request case and slot reuse. Hooks were restored. The
functional/capacity phases accepted 1,807 of 1,875 proposed draft tokens, with
68 rejected-draft steps; these forced-output retrieval workloads do not
estimate acceptance on general user conversations.

Two earlier attempts remain part of the evidence. The unbounded pool completed
all requests, but 128-token short outputs finished too early to prove five
simultaneously prefetched residents. The initial bounded run stopped before
requests because its audit inspected outer wrappers instead of the NVIDIA
indexer implementations. The successful retry changed only that audit; the
model, native library and workload were unchanged from the bounded attempt.

This establishes the examined capacity and execution contracts. Cold/cached
outputs shared 67 generated tokens before diverging, and shared-prefix
top-logprob differences reached 1.374. Numerical equivalence and broad model
quality remain open; correct retrieval markers do not resolve them. Native
performance is measured separately with fresh matched engines, without the
capacity scheduler observer.

### Matched native MTP performance

Fresh OFF and ON engines used the same checkpoint, source, native library,
compiler, prompts, cache budget and MTP1 profile above. Only
`glm_native_dense_mtp` changed. Each engine passed six functional checks and
13 stream phases: first use plus two warm repetitions at one, three, four and
five users, followed by a separate five-user cached-prefix observation. Every
request had 8,192 input and 128 output tokens. The dense Python counter was
removed before requests; symmetric graph-replay observers remained. Actual
target six/eight-row and draft two/four/eight-row replays were observed.

Two warm repetitions gave the following means:

| Users | Effective input tokens/s OFF / ON | Aggregate decode tokens/s OFF / ON | Per-user decode range, ON |
| --- | ---: | ---: | ---: |
| 1 | 2077.00 / 2078.03 | 54.97 / 55.45 | 55.45 |
| 3 | 2191.21 / 2191.13 | 116.11 / 114.52 | 37.91–38.33 |
| 4 | 2195.64 / 2198.65 | 137.49 / 137.82 | 32.48–35.89 |
| 5 | 2191.05 / 2193.19 | 146.10 / 146.10 | 27.62–31.28 |

Effective input rate divides total input tokens by the interval from first
request arrival until every request has received its first output. It includes
scheduling, mixed decode and first-token work. Decode counts frontend token
arrivals only while all requests have started output and none has finished;
whole MTP bursts belong to their arrival timestamp. These shared windows were
roughly 2.2–2.5 seconds for multiple users. Per-user ranges are ranges of the
two-repeat means, not percentiles or isolated GPU token latencies.

| Users | Complete batch seconds, OFF | Complete batch seconds, ON | Aggregate output/s including prefill, ON |
| --- | ---: | ---: | ---: |
| 1 | 6.255 (6.230–6.280) | 6.233 (6.212–6.254) | 20.54 |
| 3 | 14.367 (14.362–14.373) | 14.430 (14.360–14.500) | 26.61 |
| 4 | 18.285 (18.271–18.299) | 18.210 (18.151–18.270) | 28.12 |
| 5 | 24.467 (22.653–26.281) | 22.589 (22.557–22.620) | 28.33 |

The five-user OFF mean contains a retained 3.802-second pause between the last
user's output counts 123 and 124, after the other four had finished. GPU
utilization was nearly zero during that pause; no corresponding JIT warning
or allocator retry was recorded. Its cause remains unresolved. Prefill and
the shared decode interval were stable, so the lower ON batch mean must not
be attributed to the new kernels. ON mean TTFT was 3.942 seconds for one user;
with five users, request-order TTFT means ranged from 4.271 to 18.670 seconds.

Cached five-user requests reused exactly 6,528 input tokens per user in both
profiles. This separate single observation took 8.926 seconds OFF and 8.973
seconds ON, with aggregate shared-window decode of 135.63 and 134.95 tokens/s.
Neither engine had a runtime allocator retry or OOM; each retained one startup
retry per rank. Functional answers passed, while some generated explanations
and timed continuations differed. The OFF repeats also varied, so these are
matched-workload measurements with two observations and fixed OFF-then-ON
ordering, not identical-route or statistically established kernel gains.

The native option remains experimental and disabled by default. The measured
operation-level improvements did not produce a meaningful, consistent engine
speedup in these workloads. The larger cache and bounded workspace are the
demonstrated capacity result of this experiment.

## Qualification sequence

The pinned checkpoint's 24 files (190,843,146,350 bytes) passed integrity and
safetensors/index validation. The first full real-weight run used TP2/PP2,
partition 24/21, explicit NUMA placement, 200 W, FP16 activations, group-128
Marlin, torch mHC, FP8 indexer queries, eager execution, 512-token prefill
budget, 8192 maximum context and five sequences. Vision, MTP and prefix caching
were disabled for this initial functional baseline; they remain production
qualification requirements.

| Per GPU | PP stage 0 (GPU0/1) | PP stage 1 (GPU2/3) |
| --- | ---: | ---: |
| Reported model loading memory | 41.33 GiB | 40.50 GiB |
| Highest observed GPU memory during the test | 45.30 GiB | 44.47 GiB |

All six spot checks returned finite log probabilities and completed normally:
two arithmetic questions, a capital-city fact, integer sorting, a Python
addition function, and retrieval from a 2590-token prompt. These are functional
checks, not a comparison against unquantized quality. The 25-token arithmetic
prompt returned 10 output tokens in 4.59 seconds end to end. That cold reference
request is not an isolated decode benchmark, and the reference path is not a
performance-qualified configuration.

A second profile enables native mHC and both optional prefill flags, retaining
FP8 queries. The same six functional questions plus two singleton retrieval
probes returned correct answers with finite log probabilities. Five original
answers matched exactly; the concurrent retrieval answer changed wording but
retained the correct phrase. A subsequent reference run also changed the
length of its Python explanation, so these asynchronous mixed-workload runs
do not establish batch-invariant output or controlled five-user throughput.

The repeated singleton uses the same 2590 input token IDs and identical 12
output token IDs in both profiles. Prefix caching is disabled and counters
confirm that the entire prompt is recomputed. The second probe is warmed:

| Metric | Reference | Native mHC + prefill candidates |
| --- | ---: | ---: |
| Frontend time to first token | 2.115 s | 1.381 s |
| Effective prompt processing | 1238 tokens/s | 1893 tokens/s |
| Decode after the first token | 2.208 tokens/s | 7.189 tokens/s |
| End-to-end request time | 7.097 s | 2.911 s |

Effective prompt processing includes first-token work, PP communication and
scheduling. Decode divides the 11 subsequent tokens by the first-to-last-token
interval. These are instrumented offline observations on a short output,
not HTTP serving rates, long-context results or statistical estimates. The
combined change does not isolate individual kernel contributions. Peak GPU
memory remained approximately 45.30/44.47 GiB per GPU on stages 0/1 at 200 W.

Partial CUDA graphs provide a further decode improvement with unchanged model
source and binaries. The configuration uses `enforce_eager=False`, breakable
graphs, compilation mode 0, `cudagraph_mode="PIECEWISE"`, and the
`prefill_cudagraph_capture_token_sizes=[1,2,3,4,5]` grid with
`max_prefill_cudagraph_capture_tokens=5`. Despite that field's name, this runner
uses the token grid for piecewise decode too. The full-decode request grid is
empty. Indexer, sparse MLA and KDA core stay at eager boundaries; dense, Marlin
and mHC operations are captured. Large prefill remains eager.

Capture added 0.10–0.11 GiB per GPU, with zero allocator retries or OOMs. The
six original answers matched the native eager run token-for-token. The warmed
2590-token singleton then measured 1.345 s TTFT and 35.38 decode tokens/s,
versus 1.381 s and 7.189 tokens/s without graphs. This short comparison has
only 11 tokens after the first.

A separate fixed-length diagnostic uses one or five copies of the same
23-token prompt and forces 64 output tokens per request. It is a throughput
and graph-coverage probe, not a model-quality or representative traffic test:

| Natural request admission | Decode per user | Aggregate output including prefill |
| --- | ---: | ---: |
| One user | 35.84 tokens/s | 30.94 tokens/s |
| Five users | 17.79–20.02 tokens/s | 81.59 tokens/s |

The five-user workload used overlapping pipeline batches of two and three.
It therefore did not exercise the five-token graph. Its explicit size-5
coverage assertion failed, and that failed report is retained separately from
these valid workload timings. Capturing a graph size is not proof of replay.

A separately recorded coverage run pauses scheduling with `LLM.sleep(level=0)`,
queues all five requests, and resumes with `wake_up(tags=["scheduling"])`.
This documented level-0 operation retains weights and cache; it changes only
the final diagnostic's admission policy. Every rank then replayed the five-token
graph 63 times, and the one-user phase replayed the one-token graph 63 times.
The deliberately coalesced cohort measured 31.98 decode tokens/s per user,
159.92 aggregate post-first tokens/s and 139.16 output tokens/s including
prefill. These numbers describe identical short prompts queued together, not
natural traffic. The natural-admission measurements above remain separate.

All these runs retained zero allocator retries/OOMs. Later repetitions varied
the open-ended Python answer's explanatory prose; its first `add(a, b)` function
remained correct. A natural-admission run also showed this variation before the
scheduling barrier was introduced. Identical output across arbitrary batches
and full model-quality equivalence are not established.

### Long-context baseline and current tuning

A subsequent offline baseline retains TP2/PP2, partition 24/21, 200 W, the
native mHC/prefill candidates and piecewise graphs. It uses a 512-token prefill
budget, maximum sequence length 131328, five sequence slots and 96% GPU memory
utilization. Prefix caching, speculation and vision are disabled in this
control. Each performance request produces exactly 128 tokens; these forced
length outputs are not quality evaluations.

| Single-user input | Warm scheduled prefill rate | Warm TTFT | Warm decode after first token |
| --- | ---: | ---: | ---: |
| 32768 tokens | 2447–2468 tokens/s | 13.28–13.40 s | 35.19 tokens/s |
| 65536 tokens | 2376–2387 tokens/s | 27.46–27.60 s | 35.05–35.06 tokens/s |
| 131072 tokens | 2179–2212 tokens/s | 59.26–60.17 s | 34.75 tokens/s |

Each range covers two warmed repetitions. Scheduled prefill divides input
tokens by first-token minus first-scheduled time, including communication,
scheduling and first-token work; it excludes the initial queue. TTFT includes
that queue. Decode divides 127 subsequent tokens by the first-to-last interval.
The first-use total times were approximately 101, 121 and 250 seconds,
respectively, with repeated scorer specialization during prefill. They must
not be mixed with warm results. Separate midpoint retrieval spot checks passed
at 64K and 128K; these are not comprehensive long-context quality evaluations.

Five distinct 8192-token inputs measured 24.66–24.81 aggregate output tokens/s
including prefill, completing all 640 output tokens in 25.80–25.96 seconds.
Individual post-first decode rates ranged from approximately 6.0 to 19.7
tokens/s: early users' decode overlapped later users' prefill. They are not
five isolated steady-state decode rates. A scheduling-only admission barrier
did not improve this workload (24.41–24.47 aggregate output tokens/s). Actual
decode cohorts remained 1+4, including the overlap after all five first tokens;
the barrier did not exercise a size-five graph for these long prompts.

This maximum-length configuration has 86 shared cache blocks after warmup.
After the null block, the planner admits four simultaneous 32K requests, two
64K requests or one 128K request at this memory budget. Five simultaneous long
requests were explicitly skipped, not silently preempted or described as
five-user throughput. All completed non-profiled runs had zero preemptions,
allocator retries and CUDA OOMs. Higher memory utilization remains a separate
capacity experiment, not a qualified default.

The bounded profiler produced usable traces but its decode cleanup segfaulted
inside libkineto. Completed 32K timings were preserved; the remaining benchmark
ran in a fresh engine without profiling. Communication kernel durations include
peer waiting and are not direct measures of link bandwidth. GPU-only trace
aggregation places Marlin at 31–43% and sparse MLA at 9–12% of prefill kernel
duration across ranks. Decode still launches 27–31 graph segments per stage;
full-decode capture is an isolated candidate, not yet a qualified setting.

The existing Qwen-specific INT8 Marlin policy does not select GLM INT4/group128.
An isolated dispatcher screened GLM tiles and exact shared-memory allocations
against captured layer-3 weights/512-token routing. It queried actual compiled
register/shared-memory occupancy before explicit launches. Of 390 screened
configurations, 104 passed, 192 exceeded supported resource configurations and
94 failed the unchanged numerical gate. Repeated finalists did not establish
a useful 512-token prefill or graph-decode improvement; the baseline policy
remains selected. M1/M5 inputs in this first screen were slices of prefill.
Subsequent actual-decode captures passed 72 configurations without numerical
failures, but repeated graph timings did not establish a useful improvement.
Those M1/M5 captures exercised only eight unique experts; they do not establish
performance for diverse multi-user routing. At 1024 prefill tokens, the best
repeated explicit tile differed from the baseline by less than roughly 1%,
comparable to timing-order variation. The 2048-token screen likewise found only
roughly 1–2.5% component differences against approximately 1.8% baseline drift.
Reducing a finalist's shared allocation from 64 KiB to 26 KiB left occupancy at
one CTA per SM because of its register demand. No tile policy is promoted.

Native dense kernels subsequently passed 78 actual-input component cases,
including both vocabulary shards and global greedy selection. Leading M1
projections and the output head improved approximately 7–10%; selected output
projections improved approximately 4–9% for M2–M5. M2/M4 inputs were slices of
M5 captures; M1/M3/M5 were genuine decode captures. An isolated M3 vector
specialization fills the existing dispatch gap. The conservative shortlist
retains cuBLAS for other shapes. Whole-model performance and quality checks
remain required; neither this shortlist nor full-decode graphs are defaults.

The isolated full-decode candidate captured and replayed sizes 1–5 on all four
ranks, including short five-request cohorts and singleton cache boundaries.
However, comparison with piecewise execution found multi-request output
divergence and same-prefix log-probability differences exceeding 1. Long-prompt
cohorts also did not reliably exercise size 5 under asynchronous scheduling.
These are unresolved qualification gaps: successful graph replay does not
establish numerical correctness, and timings with different generated tokens
and expert routes are not an attributable performance gain. The candidate
remains isolated and disabled while same-engine graph/eager controls are run.

The subsequent same-engine control alternated FULL and eager execution while
retaining the FULL configuration, source and graph-safe metadata policy. Actual
prefill/decode shapes matched across all ranks. Both repeated FULL and repeated
eager runs varied: the short five-request cohort could diverge at output token
index 5 in either mode, and first-token log-probabilities already differed before
decode replay. This does not identify a capture-specific fault, nor rule out
state reuse or prefill numerical variability. It also does not establish
equivalence with the original piecewise policy. FULL therefore remains
unqualified. An isolated expert-alignment diagnostic found bitwise-stable
outputs for fixed alignments, but small output differences when atomic integer
assignment changed routed-row positions across Marlin tiles during prefill.
FP32 accumulation reduced that spread in the component experiment. This does
not establish how much whole-model variability comes from Marlin, nor select
a different accumulation policy.

A separate prefill-indexer experiment reduces length-specific compilation.
The original unrestricted dynamic-length variant failed bitwise comparison on
two of 119 logits at width 17, with a maximum difference of 2.38418579e-7;
compiler inspection identified a different FP32 multiply/FMA contraction order.
That failed gate is preserved. A guarded variant uses dynamic lengths only
when both key count and output width are at least 64 and divisible by 16,
retaining the original implementation otherwise. It passed 84 GPU cases:
76 dynamic cases and eight fallback cases, with exact logits and identical
CUDA top-k sets. Across those tested lengths and FP8/FP16 operand combinations,
the dynamic path compiled four variants rather than 84 baseline variants.
An additional 95-case gate removed the divisibility restriction while keeping
the minimum of 64. It passed exact logits, CUDA top-k agreement and independent
valid-count/bounds/sentinel checks, including odd sizes and 511/512-query chunks.
That fixture compiled 12 dynamic variants versus 87 static variants. These are
component results. A subsequent full-checkpoint run using the broader guard
passed six functional checks and 128K retrieval, with unaligned dynamic dispatch
on every rank. Each worker instantiated five dynamic prefill variants across
the measured workloads. Warm performance was effectively unchanged: 7.039 s for
one 8K request, 24.974 s for five 8K requests and 17.225 s for one 32K request,
each with 128 output tokens. First-use five-user time was 25.26 s versus 74.26 s
for the static candidate. Shared compiler-cache history prevents interpreting
that difference as a controlled cold-start speedup. No OOMs, allocator retries
or preemptions occurred. Generated-token variation remains recorded; these are
functional spot checks, not model-quality equivalence.

The repo-native `glm_sm75_dynamic_mqa` additional-config flag defaults to false.
It selects a separate runtime-length prefill kernel sharing the original
helpers, retaining the static path below 64 keys/columns or for unsupported
layouts. Paged decode is unchanged. The distilled public wrapper independently
passed the same 95-case exact component gate and the final combined integration
smoke with the rebuilt native dense extension.

The completed prefill-budget comparison gives the following warm means:

| Budget | 32K prefill tokens/s | 64K prefill tokens/s | 128K prefill tokens/s | Single 8K TTFT | Five-user 8K aggregate output tokens/s, including prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512, extensible KV | 2457 | 2381 | 2196 | 3.645 s | 24.73 |
| 1024, extensible KV | 2653 | 2538 | 2325 | 3.498 s | 25.92 |
| 2048, fixed 1 GiB KV | 2730 | 2646 | 2426 | 3.638 s | 26.19 |

Single-user decode remained approximately 35 tokens/s. Each mean covers two
warm repetitions, with the metric definitions above. The 2048 profile uses a
different KV allocation policy: extensible profiles failed 128K initialization
at both 96% and 98% utilization because upstream reserves an extra transient
memory margin. A fixed 1 GiB-per-rank KV allocation, with extensible allocation
disabled, passed 32K/64K/128K/8K retrieval spot checks and all performance phases
with zero allocator retries, OOMs or preemptions. The 1024 extensible run had one
recoverable allocator retry on each PP0 GPU during the first 128K pass, and no
OOMs or preemptions. Fixed allocation is qualified here only for text without
MTP, vision or prompt-logprobs; adding these requires renewed memory testing.

The 1024 and fixed-2048 profiles have 77 and 76 usable shared cache blocks,
respectively, sufficient for three 32K, two 64K or one 128K request, or five 8K
requests. The 2048 profile
improves long prefill but only about 1% over 1024 for this five-user workload,
and its single-user 8K TTFT is worse. No universal budget default is selected.
Full hidden-state reductions are 8/16 MiB at 1024/2048 and use NCCL under the
current strict 8 MiB custom-all-reduce cutoff.

A subsequent matched fixed-cache experiment combined native dense projections
with a 480-row threshold for the optional FP16-key/H32 prefill paths. The lower
threshold keeps these paths active when concurrent decoders reduce a 512-token
prefill chunk to 508–511 rows. Fifty component cases passed; conversion-included
FP16-key latency improved approximately 31–34% and H32 MLA approximately 36–40%
at 508/511 rows. Those component gains are not whole-model gains.

The real-model comparison retained 512 tokens/step, TP2/PP2, partition 24/21,
200 W, fixed 1 GiB KV, prefix off, no MTP and text only. Two warm repetitions
gave these complete-request times, with 128 output tokens per request:

| Workload | Reference | Dense + threshold-480 candidate |
| --- | ---: | ---: |
| One user, 8K input | 7.068 s | 7.048 s |
| Five users, 8K each | 25.737 s | 24.938 s |
| One user, 32K input | 17.280 s | 17.212 s |

The five-user aggregate output rate, including prefill, was 24.87 versus 25.66
tokens/s. Single-user decode increased from approximately 35.2–35.3 to 35.45–35.55
tokens/s. All six functional spot checks and 128K retrieval passed, and both
prefill paths executed on every rank. No allocator retries, OOMs or preemptions
occurred. Actual M3 replay also ran on every rank. Generated sequences often
differed, however, so these are workload observations rather than matched-route
causal kernel gains or proof of model-quality equivalence. M3 generated tokens
matched, while shared-prefix top-five log-probabilities differed by up to 0.626.
Broader quality evaluation remains required.

First-use five-user time was 74.26 s for the candidate versus approximately
25 s warm. The reference's first-use time was 33.33 s, with previously populated
compiler caches; this is not a controlled cold-cache comparison. Dynamic-length
MQA qualification is separate. The original reference completed its timings and
retrieval but then failed in a diagnostic prompt builder before submission;
the repaired streaming/M3 diagnostics ran in a fresh reference engine. Those
recovery results are preserved separately and are not substituted timing data.

Streaming observations show substantial gaps during other users' prefills;
after every request received its first token, per-user gaps settled near 56 ms.
The reference diagnostic used a fresh engine and the candidate a warmed engine,
so their gap difference is not an isolated measure of the candidate's benefit.
Queue policy, prefix reuse and cold compilation therefore remain material
latency targets. Existing both-pair 200 W collective tests showed little eager
16 MiB benefit from merely raising the custom-all-reduce cutoff.

The final integration smoke used the actual repo-native dynamic flag and rebuilt
ordinary RTX8000 extension, with the prototype library absent. Six functional
answers passed manual review, and each rank executed seven M3 graph replays.
Untimed checks compared static and dynamic MQA on three actual mixed-prefill
operand shapes per rank: 511 query rows, FP8 queries, FP16 gathered keys, and
key-count/output-width pairs 631/631, 887/887 and 1142/1024. All twelve comparisons
matched raw FP32 bits. Hooks restored successfully, and no allocator retries or
OOMs occurred. Runtime source matched the local worktree after newline
normalization. These checks qualify integration and the examined operations;
they do not establish whole-model bitwise invariance or broad model quality.
The final targeted public GPU regressions also passed: 37 selected cases, with
46 unrelated cases deselected, against the integrated source and extension.

1. Test FP16 mHC, pooling, KDA recurrence/prefill, sparse logits, sparse MLA,
   and physical cache addressing against numerical references on SM75.
2. Run reduced random-weight models through real engine entry points to find
   initialization, scheduler, metadata and TP2/PP2 integration failures.
   These are execution tests, not model-quality tests.
3. Load the pinned W4A16 checkpoint using Marlin, without MTP initially.
   Record per-rank persistent and peak memory, loading/repacking peaks, recurrent
   state capacity, and cache capacity.
4. Qualify vision/video, automatic/required/streamed tool calls, reasoning
   separation, prefix reuse and rejection paths. GLM uses `glm47` reasoning and
   tool parsers, with automatic tool choice enabled.
5. Compare 512/1024/2048-token prefill budgets at 32K/64K/128K/256K input context and one/five
   users. Report prefill throughput, TTFT, per-user decode rate and aggregate
   decode throughput separately. Include cold and reusable prefixes.
6. Quantize and qualify MTP, measure acceptance and memory, then compare draft
   depth, CUDA graphs, native kernel candidates and collective choices.

No Blackwell throughput or context result is claimed for this implementation.
Keep experimental source and results separate from the frozen Qwen installation.
