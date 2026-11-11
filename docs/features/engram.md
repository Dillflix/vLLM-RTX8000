# Engram: conditional memory via n-gram lookups

Engram is a *conditional memory* module attached to specific layers of a
transformer backbone. At each position, the preceding few tokens are hashed
into large static embedding tables, and the retrieved rows are fused into the
hidden state, gated by how well they match it. Where Mixture-of-Experts scales
capacity through conditional computation, Engram scales it through conditional
memory: parameters that are never computed over, only looked up, so capacity
grows at O(1) per-token cost. The mechanism and its design rationale are
described in [Conditional Memory via Scalable Lookup: A New Axis of Sparsity
for Large Language Models](https://arxiv.org/abs/2601.07372).

vLLM enables Engram automatically whenever the checkpoint declares Engram
layers, so basic serving needs no extra flags. This page explains what Engram
does and how to control its memory footprint and multi-GPU topology.

## Supported models

Engram layers are declared by the checkpoint, not chosen at serve time:

| Architecture | HF config field | Notes |
| --- | --- | --- |
| `DeepseekV41ForCausalLM` | `engram_layer_ids` | e.g. `deepseek-ai/DeepSeek-V4.1-Flash` |
| `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` | `ple_layer_ids` | same mechanism, called PLE |

Passing `--engram-config` for a model without n-gram embedding layers fails at
startup, as does running on a non-CUDA-alike platform.

## How it works

Engram processes each position in two phases: *retrieval* and *fusion*.

**Retrieval** maps the local token context to static table rows:

1. **Tokenizer compression.** At startup, vLLM builds a many-to-one map from
   the tokenizer that collapses tokens which normalize alike (NFKC, accent
   stripping, lowercasing, whitespace collapsing), so `" The"`, `"the"` and
   `"THE"` hash identically. This concentrates semantic density into a much
   smaller id space; its size must match `engram_compressed_vocab_size` from
   the checkpoint, or startup fails — every hash multiplier derives from it.
2. **Multi-head hashing.** For each n-gram order `n` (up to
   `engram_max_ngram_size - 1`), several hash heads each apply an independent
   multiplicative-XOR hash over the compressed suffix n-gram. Every (n-gram
   order, head) pair indexes its own prime-sized bucket range in the layer's
   table — disjoint across pairs, so collisions cannot collide twice — and
   the retrieved rows are concatenated into one memory vector per position.

**Fusion** turns the retrieved rows into a context-aware contribution to the
hidden state. The rows are context-independent priors and may be noisy (hash
collisions, polysemy), so they are gated: a `wkv` projection produces one key
per hyper-connection branch plus a shared value, and each branch computes a
sigmoid gate over the normalized dot product between its hidden state —
which has already aggregated global context through preceding attention —
and its key. If the retrieved memory contradicts the current context, the
gate tends toward zero and suppresses it. The gated value is added
residually into the branch's stream (`hidden + gate * value`); positions that
take no part in an n-gram (e.g. image spans) are masked so they pass through
untouched.

Engram sits only on the layers listed in the checkpoint. Placement is a
deliberate hardware-algorithm trade-off: early enough that the backbone is
relieved from reconstructing static patterns in its first layers, deep enough
that vLLM can hide table lookups behind the compute of the preceding layers
(see below).

In vLLM, hashing, lookup, and gating run as fused Triton kernels over FP8
rows with ue8m0 block scales. The layer layout — table sizes, n-gram size,
head count — is fixed by the checkpoint and is not user-configurable. Only the
storage and sharding described below are configurable, via
`--engram-config`, which takes a JSON object and can also be set field by
field as `--engram-config.<field>`
([CLI reference](../cli/serve.md#-engram-config)).

## CPU offload

Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
carries two Engram layers of roughly 384M table rows each, with 256-byte FP8
entries plus their ue8m0 block scales — about 200 GB of table weights alone.
Because lookups are deterministic — the indices are known from the token ids
alone, before the forward pass — the tables need not sit in GPU memory at
all. By default vLLM stores them in **pinned host memory** and serves lookups
directly from the Triton kernel through unified virtual addressing (UVA),
prefetching on a side CUDA stream so transfers overlap with the compute of
preceding layers. The freed GPU memory goes to the KV cache.

`cpu_offload` defaults to `true` (on)

To keep the tables resident on GPU instead (e.g. when host memory is
scarce):

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --engram-config.cpu_offload false
```

CPU offload requires a GPU with UVA support;
vLLM fails fast when it is unavailable.

### Experimental Qwen3.8-Flash-Next on Quadro RTX 8000

This fork runs `Qwen/Qwen3.8-Flash-Next-FP8` on four SM75 GPUs using two
NVLink TP pairs and two pipeline stages. The profile below was tested on
four Quadro RTX 8000 GPUs on November 10, 2026. Qualification covers text
serving at 1-5 concurrent short requests and single requests containing
32K, 64K and 128K input tokens. It does not establish broad model accuracy
or optimal settings for every workload.

The FP16 QSA path uses FP16 activations and unquantized KV/indexer caches.
`--attention-config.indexer_kv_dtype auto` follows the Qwen4Exp model dtype;
`fp16` can also be selected explicitly. FP8 weights use Marlin's weight-only
path on Turing. The PLE table stays in pinned CPU memory. This does not add
native FP8 arithmetic to Turing or change the model's expert routing.

The tested launcher is
[`examples/deployment/serve_flashnext_rtx8000.sh`](../../examples/deployment/serve_flashnext_rtx8000.sh):

```bash
bash examples/deployment/serve_flashnext_rtx8000.sh /path/to/Qwen3.8-Flash-Next-FP8
```

It selects TP=2, PP=2, Marlin for both linear and MoE kernels, five sequence
slots, a 512-token prefill budget, and an 8192-token initial context.
`FULL_DECODE_ONLY` CUDA graphs capture batches of 1-5 requests, without
Torch compilation; prefill runs eagerly. Pass `--enforce-eager` for the
eager comparison or a later `--compilation-config` to override this graph
profile. Speculative decoding and sequence parallelism are disabled by
default; the tested MTP option is described below.

On the tested dual-socket Xeon E5-2630L v4 host, with the existing 200 W
GPU limits, a 26-token prompt and 128 generated tokens gave the following
medians over three repetitions. Thinking and prefix caching were disabled.
Aggregate throughput includes prefill and request completion; per-request
decode rate measures the interval between the first and last output chunks.

| Concurrent requests | Eager aggregate tokens/s | Graph aggregate tokens/s | Graph decode tokens/s per request | Graph first token (s) |
| --- | --- | --- | --- | --- |
| 1 | 9.09 | 48.59 | 54.10 | 0.286 |
| 2 | 17.57 | 83.23 | 47.15 | 0.371 |
| 5 | 22.43 | 192.77 | 46.63 | 0.593 |

All 24 benchmark responses matched between the two modes. Both modes
also passed four basic exact-answer checks and three retrieval checks
with 998, 4064 and 7145 prompt tokens. The GPU qualification additionally
passed 39 FP16 kernel tests and TP2 PP2 Marlin checks on all four ranks.
These checks do not replace a broad accuracy evaluation.

The same graph profile was also tested with **actual input lengths** of
32,768, 65,536 and 131,072 tokens, each followed by 128 output tokens.
The configured limit was 131,200 to leave room for the output. One request
ran at a time, with one warmup and three measured repetitions per length;
prefix caching and speculative decoding remained disabled.

| Input tokens | Prefill tokens/s | First token (s) | Decode tokens/s |
| --- | --- | --- | --- |
| 32,768 | 2,455 | 13.40 | 50.86 |
| 65,536 | 2,341 | 28.08 | 50.46 |
| 131,072 | 2,142 | 61.36 | 49.65 |

Values are medians. Prefill rate divides computed input tokens by vLLM's
request prefill duration (first scheduling to first token), including
chunk scheduling and pipeline overhead. It is not isolated GPU kernel
throughput. First-token latency also includes frontend and initial queue
time. Decode rate uses vLLM's server decode duration to include generated
tokens with no visible text. The input corpus comprised repository
documentation, with retrieval keys at its beginning, middle and end.
All 12 requests, including warmups,
returned the three keys in order; none was preempted. This is a retrieval
smoke check, not a comprehensive long-context accuracy evaluation or a
measurement of multiple simultaneous long requests.

To reproduce the tested context capacity:

```bash
MAX_MODEL_LEN=131200 bash examples/deployment/serve_flashnext_rtx8000.sh /path/to/Qwen3.8-Flash-Next-FP8
```

#### Speculative decoding

The launcher keeps speculation disabled by default. Native MTP with three
draft tokens was tested at TP=2, PP=2 using the checkpoint's existing MTP
weights and FP16 execution. With the same 131,200-token configuration,
26-token prompt and 128 output tokens, the short-request comparison was:

| Concurrent requests | Baseline aggregate tokens/s | MTP aggregate tokens/s | Speedup |
| --- | --- | --- | --- |
| 1 | 48.20 | 70.19 | 1.46x |
| 2 | 82.09 | 108.50 | 1.32x |
| 5 | 191.99 | 222.69 | 1.16x |

These are medians over three repetitions and include the full request
duration. All 24 measured responses matched the baseline exactly; the four
basic exact-answer checks also passed. Draft-token acceptance was about
50% on this short prompt. Acceptance depends on the workload.

MTP also completed the same actual-length context tests, with one warmup
and three measured repetitions at each length:

| Input tokens | Prefill tokens/s | First token (s) | Decode tokens/s | Draft acceptance |
| --- | --- | --- | --- | --- |
| 32,768 | 2,254 | 14.58 | 69.56 | 42.7% |
| 65,536 | 2,151 | 30.56 | 90.76 | 71.5% |
| 131,072 | 1,973 | 66.59 | 65.02 | 70.3% |

Timing definitions match the baseline context table. All 12 retrieval
checks passed with no preemption. Six outputs matched the baseline
exactly, and nine matched after stripping outer whitespace. Some 64K
explanations differed in wording; the baseline also varied between
identical repeated requests. Exact long-context greedy equivalence is
therefore not established by these tests.

For these 128-token completions, the extra prefill time outweighed the
decode saving: median total request time increased from 15.87/30.55/63.81
seconds to 16.36/31.91/68.45 seconds at 32K/64K/128K. Longer generations may
repay the prefill cost, but that crossover was not measured.

MTP is an opt-in tradeoff: the server-reported KV capacity fell from
636,395 to 214,690 tokens with this configuration, and drafting adds prefill
work. These capacity estimates do not qualify five simultaneous 128K
requests. To enable the tested three-token draft configuration:

```bash
MAX_MODEL_LEN=131200 bash examples/deployment/serve_flashnext_rtx8000.sh /path/to/Qwen3.8-Flash-Next-FP8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

[DFlash2 has a Flash-Next draft checkpoint](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
whose published recipe uses SGLang at TP=1 on a GB300. It is not enabled
here. The current target configuration explicitly accepts only native MTP
and linear n-gram speculation. DFlash2 integration additionally needs the
correct target features from layers 5, 15, 25, 35 and 45, averaging the
four HC streams, and transport of the early-stage features across PP.
The draft's FP16 attention path then needs SM75 correctness and performance
validation before relaxing that configuration guard.

Multimodal inputs, alternative checkpoints, one/two-token MTP tuning and
simultaneous long-context requests remain unqualified.

For GPU order `0,1,2,3`, the default NUMA node list is `0 0 1 1` and the CPU
lists are `0-9,20-29 0-9,20-29 10-19,30-39 10-19,30-39`. Override
`NUMA_BIND_NODES` and `NUMA_BIND_CPUS` with space-separated per-worker
entries when the host differs. `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, and
`MAX_BATCHED_TOKENS` expose the scheduler settings. Extra launcher arguments
are passed to `vllm serve`.

The FP16 enablement and initial SM75 QSA profiles derive from
thecyberdevl's FlashNext patches; see
`examples/deployment/flashnext_rtx8000_provenance.json` for pinned sources
and the scope of this port. The newer fused preparation kernel also needs
model-dtype rounding, which is specific to this integration.

## Disk offload (Qwen4Exp)

`disk_offload` keeps Qwen4Exp PLE tables in the checkpoint files: no table
is allocated in GPU or pinned host memory, and nothing is copied to an offload
directory. Each step reads that step's rows from the safetensors files on the
host. Use it where neither `cpu_offload` nor a resident table fits, for example
on unified-memory GPUs such as DGX Spark, where pinned host memory comes out of
the same pool as the weights and the KV cache:

```bash
vllm serve <qwen4exp-checkpoint> --engram-config '{"disk_offload": true}'
```

It applies only to Qwen4Exp on CUDA and takes precedence over `cpu_offload`.

Rows are read with `preadv` and copied to fixed device buffers before the
forward, so CUDA graphs replay unchanged. They stay in the checkpoint format
(BF16, FP8 or NVFP4) and are decoded on the GPU, so an unquantized table must
match `--dtype`. Rows that were read recently are served from the OS page
cache, which is shared by co-located ranks, not counted in the worker's RSS,
and reclaimed by the kernel under memory pressure.

Costs and limits:

- The read needs the step's token ids, so it waits for the previous step on
  the GPU, and every TP rank reads all of its rows.
- A row that is not in the page cache is read from disk. For a fast first
  request, read the PLE shard files once after the server has started.
- The checkpoint files must not change while the server runs. Truncation stops
  the engine with an error naming the file; deleting or renaming is safe.
- Weight reload and elastic EP are unsupported. Loading fails when tensors are
  not file-backed safetensors views (`.bin`, runai, `fastsafetensors`,
  `instanttensor`, or `--safetensors-load-strategy eager`);
  `--load-format dummy` serves zeros.

### Experimental INT8 experts and selective dense prefill

Expert-only compressed-tensors W8A16 checkpoints can use symmetric INT8
group-64 or per-channel scales with the same TP2 PP2 profile. Conversions from
the official FP8 checkpoint add a quantization step and require model-quality
evaluation. Preserve the PLE tensors and set `text_config.ple_embedding_dtype`
to `float8_e4m3fn` so that CPU-offloaded table storage remains unchanged.

For those checkpoints, `VLLM_RTX8000_CUTLASS_PREFILL=1` enables an opt-in SM75
experiment on the Gated DeltaNet `in_proj_qkvz` and `out_proj` layers. It retains
the loaded FP16 weights for decode and caches an additional per-output-channel
INT8 copy for CUTLASS. Only non-speculative prefill portions containing at least
128 tokens use dynamic INT8 activation quantization. Decode rows in mixed
batches, smaller prefill portions, and speculative batches use the original
FP16 path. The extra copy consumes VRAM and must be included in memory and
KV-cache comparisons. This switch is disabled by default; it does not enable
W8A8 experts or quantize the shared experts.

#### Per-channel W8A16 prefill budgets and native MTP

The 2026-11-10 RTX 8000 comparison used the per-channel INT8 expert
conversion of Qwen3.8-Flash-Next, TP2 PP2, NUMA-local NVLink pairs,
CPU-offloaded PLE tables, FP16 activations and caches, decode CUDA graphs,
five sequence slots, and the existing 200 W GPU limits. Prefix caching and
the selective CUTLASS prefill experiment were disabled. GPU memory
utilization stayed at 0.90 as an experimental control.

Cold-prefill screening results without MTP were:

| Token budget per step | 32K prefill tokens/s | 64K prefill tokens/s | 128K prefill tokens/s | Longest streaming pause during other users' prefills |
| --- | --- | --- | --- | --- |
| 512 | 2,579 | 2,472 | 2,268 | 0.45 s |
| 1024 | 2,770 | 2,657 | 2,425 | 0.83 s |
| 2048 | 2,818 | 2,736 | 2,512 | 1.60 s |

Each long-context value is one screening observation with one request.
Prefill throughput uses server prefill time. The pause measurement uses
five simultaneous 8K prompts and is a maximum observed gap between delivered
chunks after a request starts streaming; it excludes that request's initial
wait. A larger budget improves long-context prefill but can interrupt other
users' streams for longer. At 8K, 1024 had lower single-request time to first
token than 2048 (3.06 versus 3.23 seconds).

The following decode comparison fixes the token budget at 2048. Each request
has 8,192 input tokens and generates 768 tokens of a long-form tutorial.
Results are medians of two measured repetitions after warmup:

| Native MTP draft tokens | One-user decode tokens/s | Five-user mean decode tokens/s per user | Five-user total decode tokens/s |
| --- | --- | --- | --- |
| Disabled | 52.4 | 37.6 | 187.8 |
| 1 | 73.4 | 46.2 | 230.9 |
| 3 | 82.2 | 47.6 | 237.9 |

Decode rates count actual streamed token IDs in a shared wall-clock window
after every prompt has completed prefill and before any response finishes,
with 0.5 seconds trimmed from each boundary. They exclude prefill and are
not whole-request throughput. No measured tutorial response reached an
end-of-answer token. Individual MTP streams have different acceptance rates;
the five-user mean is total throughput divided by five.

Three-draft MTP reduced the one-user 8K/768-token request from 17.87 to
12.66 seconds. Its five-user whole-request output throughput increased from
109.7 to 122.4 tokens/s, a smaller improvement than its sustained decode
gain. MTP adds prefill work: three-draft prefill rates were 2,642/2,543/2,322
tokens/s at 32K/64K/128K. Long prompts with short answers can therefore
finish sooner with MTP disabled.

To reproduce the tested long-response configuration:

```bash
MAX_MODEL_LEN=131200 MAX_BATCHED_TOKENS=2048 \
VLLM_RTX8000_CUTLASS_PREFILL=0 \
bash examples/deployment/serve_flashnext_rtx8000.sh \
  /path/to/Qwen3.8-Flash-Next-experts-W8A16-channel \
  --served-model-name Qwen3.8-Flash-Next-W8A16 --no-enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

All 15 long-context retrieval checks passed, with no preemption across the
75 timed requests. The fixed 8,184-token teacher-forced diagnostic scored
perplexity 3.273/3.267/3.277 for disabled/one-draft/three-draft MTP at budget
2048. These are narrow diagnostics, not broad accuracy qualification.
Repeated greedy tutorial outputs varied even without MTP; token-for-token
equivalence is not established by this comparison. Five-user timing was
tested at 8K per request; 32K/64K/128K cases used one request.

Reported hybrid-cache capacities at budget 2048 were 1,541,222 tokens
without MTP, 944,351 with one draft token, and 860,088 with three. These
estimates do not independently qualify five simultaneous 128K requests.

Separate profiler captures identify sparse attention as the largest prefill
kernel cost (about 41–42% of summed GPU kernel time at budgets 1024/2048).
Dense FP16 projections are a major decode compute cost. At 2048, a 10 MiB
hidden-state reduction exceeds the custom all-reduce's 8 MiB cutoff and uses
NCCL. These measurements motivated the native-kernel and NVLink experiments
below. Profiler time is excluded from timing results.
Repeated profiler sessions crashed inside libkineto cleanup in the first
run; subsequent runs used one capture per worker lifetime. The baseline
five-user trace is incomplete, while its timing runs, prefill capture, and
single-user capture completed.

#### Native SM75 attention, dense projections, and collectives

The RTX 8000 experiments add an optional `_rtx8000_C` extension. Its sparse
attention kernel is specific to the Flash-Next TP2 geometry: 12 query heads,
one KV head, head dimension 256, and FP16 K/V. It uses Turing Tensor Core
instructions for both QK and PV, FP32 accumulation and online softmax,
shared K/V gathers across heads, and a fused output gate. It preserves the
selection list and the FP16 rounding before gating. Cache pages may have
padding or other layers between them; the kernel uses the actual page stride.
Unsupported layouts retain the existing attention path.

The native CUDA kernel uses 64 registers per thread, 29,056 bytes of
shared memory, no spills, and two resident 256-thread blocks per SM (50%
theoretical warp occupancy). The baseline Triton kernel used 233–241
registers and had 25% estimated occupancy. On random sparse selections at
128K context, the prototype was 3.3–3.5x faster at 512–2048 query tokens.
These are isolated kernel measurements, not serving speedups. The prototype
screen used 2052 selected entries; the model's production packed buffer has
2051 selection columns plus a count column. The integrated kernel supports
the runtime width and is tested against the production selection builder.

Native dense dispatch selects vectorized FP16 CUDA kernels with FP32
accumulation only for measured local shapes and small token counts. Other
shapes use the existing linear method. The extension also includes two
SM75 CUTLASS Tensor Core candidates for reproducible comparisons; their
availability does not cause universal replacement of cuBLAS. Weights and
activations are not additionally quantized by these switches.

On the same per-channel INT8 W8A16 checkpoint and TP2 PP2 system, native
attention plus selective dense dispatch produced the following serving
results with a 2048-token prefill budget and MTP disabled:

| Measurement | Previous kernels | Native SM75 kernels |
| --- | ---: | ---: |
| 32K prefill, tokens/s | 2818 | 4178 |
| 64K prefill, tokens/s | 2736 | 3973 |
| 128K prefill, tokens/s | 2512 | 3505 |
| One-user decode, tokens/s | 52.4 | 56.3 |
| Five-user decode, total tokens/s | 187.8 | 197.9 |
| Five-user decode, mean per-user tokens/s | 37.6 | 39.6 |
| Five-user output rate including prefill, total tokens/s | 109.7 | 130.9 |

The long-context screen used one cold request per length, with 128 output
tokens. Decode used two repetitions of 8K input and 768 output tokens at
one/five users, after warmup. Decode-only rates use the common interval
after all prefills and before any request finishes; whole-request rates
include prefill. All retrieval checks passed with zero preemptions. The
fixed 8184-token teacher-forced diagnostic perplexity was 3.2732 before and
3.2751 after; this is a narrow numerical check, not broad model evaluation.
Power limits remained 200 W, memory utilization 0.90, CPU table offload
enabled, and prefix caching disabled. Single 128K requests do not qualify
five simultaneous 128K requests.

A dense-only diagnostic improved one-user decode from 52.4 to 54.4 tokens/s.
In a separate native prefill profile, sparse attention fell from about 42%
to 16% of summed GPU kernel time. Marlin expert GEMMs and gated delta-network
kernels now account for larger shares. Kernel durations can overlap across
streams, so these percentages are not fractions of unprofiled wall time.

With three-token native MTP enabled, the new kernels reached 86.2 tokens/s
for one user and 255.1 total tokens/s for five users (51.0 mean per user),
versus 82.2 and 237.9 total with the previous kernels and the same MTP
setting. The one-user repetitions were 84.1 and 88.4 tokens/s; acceptance
rate affects this result. Prefill reached 4020/3794/3309 tokens/s at
32K/64K/128K, versus 2641/2543/2322 previously. Five-user total output rate
including prefill improved from 122.4 to 151.3 tokens/s. All retrieval
checks passed, with zero preemptions and diagnostic perplexity 3.2710.
Reported cache capacity remained 860,088 hybrid-normalized tokens with
MTP3, versus 1,541,222 without MTP. This is an estimate, not a measured
simultaneous-context admission limit.

For this per-channel checkpoint, the measured serving profile is:

```bash
VLLM_RTX8000_NATIVE_QSA=1 VLLM_RTX8000_NATIVE_DENSE=1 \
MAX_BATCHED_TOKENS=2048 MAX_MODEL_LEN=131200 \
  bash examples/deployment/serve_flashnext_rtx8000.sh /path/to/per-channel-checkpoint \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Leave MTP disabled for the highest prefill rate; MTP3 won the measured
decode and whole-request output-rate comparisons. Native kernels remain
independent opt-ins rather than being applied to unqualified model layouts.

Build against the same PyTorch environment used by the server. A normal
source build includes the extension when SM75 is requested. For an existing
editable installation, the isolated incremental target avoids rebuilding
the other vLLM extensions:

```bash
cmake -S csrc/rtx8000 -B build/rtx8000 -G Ninja \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCUTLASS_INCLUDE_DIR=/path/to/cutlass/include \
  -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build build/rtx8000 -j 4
cmake --install build/rtx8000
```

The independent opt-in controls are `VLLM_RTX8000_NATIVE_QSA=1` and
`VLLM_RTX8000_NATIVE_DENSE=1`. A requested native extension must be built;
failure to import it is an error. The startup/runtime log reports native
attention activation and unsupported-layout fallbacks. Native decode's
32-way split merge is warmed before CUDA graph capture. The extension is
built for the installed PyTorch ABI and must be rebuilt after changing it.

`VLLM_RTX8000_CUSTOM_AR_MAX_MIB` controls the SM75 custom all-reduce
allocation and eligibility cutoff together. It defaults to 8 and accepts
1–256 MiB. The size test remains strictly less than the cutoff, so a 10 MiB
input requires a cutoff above 10 MiB. This is a buffer/dispatch limit, not
an SM75 hardware limit. An increased cutoff does not change peer checks,
group membership, dtype, or the reduction algorithm.

Pair-local tests on both RTX 8000 NVLink pairs verified exact FP16 integer
sums in eager execution and CUDA graphs from 5 KiB through 32 MiB. At
10 MiB, a 16 MiB custom buffer took about 303 us eagerly and 260 us in a
graph, versus default NCCL at 316 us and 313–314 us. Raising the buffer to
32 or 64 MiB did not materially improve the same 10 MiB operation. Splitting
it into 5 MiB custom reductions took about 313 us eagerly; it does not
reduce total data volume. These eager figures matter for this deployment's
uncaptured prefill.

NCCL with `NCCL_MIN_CTAS=16 NCCL_MAX_CTAS=16` took 278–289 us for the same
eager 10 MiB operation. Forced Ring/Simple and Tree/Simple did not beat it;
forced Ring/LL took 664–668 us. CTA settings also affect the process's
pipeline communicators, so serving comparisons are required before adopting
them globally. The full-model no-MTP comparison produced only 0.1–0.3%
prefill differences and unchanged decode rates, so automatic NCCL settings
are retained. LL128 was not forced. The larger custom cutoff is exposed
for experiments; its end-to-end serving benefit is not established.

The lossless architectural opportunity for reducing communication volume
is to keep token states sharded through compatible adjacent operations,
using reduce-scatter and postponing all-gather. The existing HyperConnection
sequence-parallel configuration explicitly requires PP1; TP2 PP2 needs
pipeline-state and metadata support before that path can be enabled.
Smaller chunks reduce individual message size, while compressed wire
formats change numerical precision and require separate quality evaluation.

Reproducible microbenchmarks live in `benchmarks/kernels/benchmark_rtx8000_qsa.py`,
`benchmark_rtx8000_dense.py`, and `benchmark_rtx8000_collectives.py`. Run
collectives with four ranks on the stated GPU/NUMA ordering. Run independent
kernel benchmarks only when those GPUs are idle; their timing must not
overlap model serving measurements.

Final qualification passed 13 targeted GPU tests, 54 sparse-attention
comparisons, 385 dense comparisons, and the 504 collective timings above.
Coverage includes production selection packing, empty/invalid rows, padded
cache pages, and page offsets beyond signed 32-bit indexing. The standalone
extension was rebuilt and installed against PyTorch 2.14.0+cu130; the full
wheel build and other GPU architectures were not rerun in this tuning pass.
Checkpoint revision, launch settings, source hashes, and measured rates are
recorded separately from the original FP8 qualification in
`examples/deployment/flashnext_rtx8000_provenance.json`.

#### Prefill graphs, GDN kernels and Marlin tuning

Eager prefill is the measured launch profile's explicit `FULL_DECODE_ONLY`
choice, not an SM75 hardware restriction. This Qwen4Exp implementation supports
breakable piecewise graphs without Torch compilation. QSA, the GDN attention
core, PLE prefetch/finalization, and short convolution remain eager breaks;
surrounding dense projections, Marlin experts, and reductions can be captured.
TP2 PP2 and CPU table offload do not inherently disable this path. Full
whole-prefill capture is a separate capability: QSA and GDN currently advertise
uniform-batch support, and GDN's full-capture metadata builder is decode-only.

A prefill experiment must set its capture sizes independently of the decode
request counts. To use the standard fine grid through 2048 tokens:

```json
{
  "mode": 0,
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "max_prefill_cudagraph_capture_tokens": 2048,
  "decode_cudagraph_capture_request_sizes": [1, 2, 3, 4, 5]
}
```

An explicit coarse list `[512, 1024, 2048]` was tested with the native kernels,
no MTP, and otherwise unchanged settings. Its trace confirmed graph execution
for Marlin/NCCL and eager execution for QSA/GDN. Single-request prefill rates
were 4166/3955/3491 tokens/s at 32K/64K/128K, within 0.5% of the eager profile.
Five-user time to the last first token rose from about 10 to 14 seconds; this
coarse list is not recommended. Small nonuniform batches are padded to its
minimum 512-token graph, so this result does not establish the performance of
a finer capture grid. Ordinary one-to-five-user decode uses separate full
decode graphs in both configurations.

The standard 31-size fine grid was then tested separately, with graph-padding
metrics enabled. Prefill rates were 4157/3954/3487 tokens/s; five-user time to
the last first token remained 14.0 seconds, and total output rate including
prefill was 115.5 tokens/s versus 130.9 for eager prefill. Runtime logs confirmed
2048-token graph replay without padding, plus small-tail padding such as
20 to 24 tokens. The fine grid therefore did not recover the concurrent
prefill regression; coarse padding does not explain most of it. The cause of
the mixed-prefill slowdown remains unresolved. Both graph variants passed all
three retrieval checks with no preemptions; diagnostic perplexities were
3.2725 and 3.2801 versus 3.2751 for the eager control. The tested default
remains eager prefill with full decode graphs. This does not rule out a later
graph implementation improvement or a graph/larger-custom-collective combination.

GDN's TP2 dimensions are eight key heads,
24 value heads, and K=V=128. The checkpoint uses FP32 recurrent state, while
chunk snapshots and dot operands are FP16. Both major profiled kernels use
255 registers/thread; output has 25% estimated occupancy, while the one-request
state recurrence launches only 96 CTAs and has 17% estimated occupancy.
The original warmup tunes at 64 tokens, and the tuning keys omit sequence
length and request count.

A bounded one-request, 2048-token synthetic kernel screen found:

| GDN kernel | T64-selected configuration | Candidate | Eager latency before/after |
| --- | --- | --- | ---: |
| State recurrence | BV32, 4 warps, 4 stages | BV16, 4 warps, 2 stages | 2.445 / 1.881 ms |
| Output | BK32/BV32, 8 warps, 2 stages | BK32/BV64, 8 warps, 3 stages | 2.758 / 1.820 ms |

All 50 existing/experimental configurations passed comparison with the
reference outputs on these synthetic inputs. The state candidate reduced
reported spill registers from 438 to 130 and doubled the grid to 192 CTAs.
The output candidate is already in the available tuning list. This initial
screen used approximate WY operands; the subsequent qualification below uses
the actual WY calculation and full GDN chain.

`VLLM_RTX8000_GDN_TILES=1` selects BV16/4-warps/2-stages for the state
recurrence and BK32/BV64/8-warps/2-stages for output. This opt-in dispatch is
restricted to SM75, the measured FP16 H24/Hg8/K128/V128 geometry, FP32 gates
and initial state, one request, and 512–2048 total scheduled tokens. Other
layouts and request counts retain the existing implementation. Output stages
2 and 3 were effectively tied in the initial screen; the qualified path uses 2.

Six public-entry correctness cases passed on RTX8000, covering T64/512/1024/2048,
five one-token requests, and packed lengths 17/63/64/65/319. Changed tiles
preserved the chunk states, updated values, final state, and output bit for bit.
The full-chain benchmark uses production FP32 beta and gates, real WY
intermediates, independent token recurrence and causal-output references, and
poisoned supplied output buffers. Its 70 rows passed with no failures:

| Scheduled tokens, one request | Full GDN chain, baseline | Tile override | Time reduction |
| ---: | ---: | ---: | ---: |
| 512 | 1.542 ms | 1.161 ms | 24.7% |
| 1024 | 2.981 ms | 2.135 ms | 28.4% |
| 2048 | 5.787 ms | 4.124 ms | 28.7% |

These are eager CUDA-event medians on GPU0, five samples of ten calls,
excluding startup/compilation. They are GDN component latencies, not serving
rates. Five-request cases intentionally retained the fallback. All changed
full-chain results were bit-identical to the original chain; the largest
absolute differences against independent FP32 token recurrence were 1.53e-5
for output and 6.46e-5 for final state. Reproduce with
`benchmarks/kernels/benchmark_rtx8000_gdn.py --modes baseline tiles`.

The earlier tile-only TP2 PP2 serving qualification, with the native QSA/dense kernels,
2048-token budget and no MTP, measured 4469/4226/3693 prefill tokens/s at
32K/64K/128K, versus 4178/3973/3505 before the GDN change. These are single
cold-context observations (7.0%/6.4%/5.4% higher rates). Two decode repetitions
gave 56.33 tokens/s for one user and 198.27 total for five simultaneous users
(39.65 mean per user), consistent with the previous decode rates. Five-user
time to the last first token fell from 10.00 to 9.44 seconds. All retrieval
checks passed, there were no preemptions, and the fixed 8184-token likelihood
diagnostic had perplexity 3.2686 versus 3.2751. This diagnostic does not prove
broad quality or bit-identical generated text; repeated greedy generations
were not bit-identical, as in earlier serving measurements.

`VLLM_RTX8000_NATIVE_GDN=1` replaces the GDN chunk-output calculation with
the optional `_rtx8000_C` extension for the same FP16 head geometry and FP32
gates. It supports packed requests, chunk tails and INT32/INT64 metadata;
unsupported layouts retain the existing path. This switch does not replace
the recurrent decode kernel. Combine it with `VLLM_RTX8000_GDN_TILES=1` to
retain the qualified state tiles.

The native output kernel computes raw QK once per query tile and key head,
sharing it across three value heads and all 128 value columns. Q-times-state
and per-head gating remain separate. Turing WMMA uses FP32 accumulation and
preserves the FP16 gated-score boundary before the PV product. Query16 uses
65 registers/thread and 27,904 shared bytes, with no spills and two resident
256-thread CTAs per SM. The included Query32 comparison uses 96 registers and
39,168 shared bytes, also without spills. Alternating-order comparisons on
GPU0 and GPU1 did not establish a consistent Query32 advantage at the primary
1024/2048 sizes, and some multi-request shapes regressed. Dispatch retains
Query16; occupancy alone is not the selection criterion.

Six native correctness cases passed, testing both variants, poisoned outputs,
packed tails, INT64 metadata, and graph replay with changed inputs. Checks
retain the 2e-3 absolute/relative element bound and additionally require
relative RMS error at most 1e-3 against the independent reference. The largest
observed native relative RMS was 2.46e-5 (0.00246%). Native memory checks
reported zero errors; race checks reported zero hazards, errors or warnings.
The full 150-row benchmark passed. At T2048, the public output entry point took
0.395 ms with native Query16 versus 1.835 ms for tuned Triton, while full GDN
with state tiles and native output took 2.706 ms versus 5.790 ms for the original chain.
These remain component measurements; graph results in the benchmark do not
mean that serving captures the prefill GDN chain.

Rebuild the optional native extension after pulling these source changes;
an older `_rtx8000_C` binary does not contain `gdn_o`. The standalone CMake
target now includes separately compiled Query16/Query32 CUDA units and their
binding. The tested binary used PyTorch 2.14.0+cu130 and CUDA 13.0.3; the full
vLLM wheel build was not rerun for this extension-only installation.

With both GDN switches enabled, the same no-MTP serving workload measured:

| Cold prefill context | Original GDN | State/output tiles | State tiles + native output |
| ---: | ---: | ---: | ---: |
| 32K | 4178 tokens/s | 4469 tokens/s | 4786 tokens/s |
| 64K | 3973 tokens/s | 4226 tokens/s | 4500 tokens/s |
| 128K | 3505 tokens/s | 3693 tokens/s | 3893 tokens/s |

Decode measurements below use 8192 input and 768 output tokens per request,
with two repetitions; they are separate from the long-context prefill cases.
Native output plus state tiles improved these prefill observations by
14.6%/13.3%/11.1% over original GDN. Decode remained 56.22 tokens/s for one
user and 197.27 total for five users (39.45 mean per user). Five-user time to
the last first token was 8.75 seconds; total output rate including prefill
was 136.37 tokens/s. All retrieval checks passed with no preemptions; the
likelihood diagnostic perplexity was 3.2640. Prompt hashes and all eight
likelihood windows match the control. The prefill trace confirmed 144 native
GDN output launches per rank; on rank0, output's share of summed kernel time
fell from 9.1% with tile tuning to 2.1%, while Marlin rose to 25.4%.

With three-token native MTP, the same GDN changes produced
4554/4248/3640 prefill tokens/s at 32K/64K/128K, gains of
13.3%/12.0%/10.0% over the previous MTP profile. Two decode repetitions gave
88.51 tokens/s for one user (86.94 and 90.08) and 255.84 total for five users
(51.17 mean per user). These decode differences should not be attributed to
the prefill GDN kernels; draft acceptance and generated sequences varied.
Five-user time to the last first token was 9.00 seconds, and total output
rate including prefill was 160.19 tokens/s. All retrieval checks passed,
preemptions remained zero, diagnostic perplexity was 3.2580, and reported
hybrid-normalized cache capacity stayed 860,088 tokens. MTP continues to
trade some prefill speed for faster decode. Neither configuration qualifies
five simultaneous 128K requests.

The RTX8000 launcher enables the qualified GDN changes with
`VLLM_RTX8000_GDN_TILES=1 VLLM_RTX8000_NATIVE_GDN=1`; either can be disabled
by explicitly setting it to `0`. The general vLLM defaults remain off.
Exact checkpoint,
source hashes, binary hash, settings and results are recorded in
`gdn_tiles_and_native_output_qualification` in the provenance JSON.

Marlin is already a native SM75 Tensor Core backend. At the measured 2048-token
prefill budget it uses an M48/N128/K64 tile with 251 registers/thread; decode
uses M8/N128/K64 with 99 registers. Both allocate 31,744 shared bytes and reach
25% estimated occupancy. Concrete experiments are model-specific M-tile selection, exact
shared-memory sizing, and activation/routed-output fusion. The computed
M8 scratch requirement is 21,248 bytes, and these GPUs report zero driver
shared-memory reservation, making three CTAs per SM a plausible experiment;
occupancy and speed must still be measured. Per-rank 2048-token intermediates
include 25 MiB of gate/up output, 12.5 MiB after activation, and 100 MiB before
the top-10 output sum. These are logical intermediates, not additive peak
VRAM: gate/up and down outputs reuse storage. Fusion must preserve split-K and
multi-expert reduction semantics. These observations motivated the captured-layer
experiments below.

The opt-in `VLLM_MARLIN_MOE_EXACT_SHARED_MEM=1` experiment replaces the
per-block shared-memory budget with the dispatcher's computed scratch bound
for SM75 FP16, per-channel INT8 (`uint8b128`), without zero points. CUDA's
occupancy API checks each candidate, including actual register granularity
and device reservations; cached results avoid repeated occupancy queries.
The existing lock and FP32 temporary workspace permit at most four CTAs per
SM, and the new path enforces that limit. Other formats retain their existing
dispatch. Despite the switch's name, the M8 bound remains conservative: it
uses M16 metadata and A-storage accounting. The 21,248-byte bound is safe,
but is not a claim of the smallest possible allocation.

`benchmarks/kernels/benchmark_rtx8000_marlin.py` captures packed weights,
actual routed activations and activation settings from an untimed worker.
It verifies the standard activation callback and rejects unsupported custom
callbacks. Replays test full MoE layers and individual gate/up or down GEMMs,
with M8/16/32/48/64 and supported existing N/K tiles. Sharing allocation alone
must preserve output bits on the same kernel/grid. Other candidates retain
the existing Marlin test's absolute 0.04 bound, with reference RMS, maximum
magnitude and normalized errors also reported. Candidate screening and
repeated finalist timings are separate; automatic resource selection is
marked as source-derived, while CUDA occupancy maxima are directly queried.

The isolated benchmark dispatcher reuses the installed compiled kernels in
a separate operator namespace. It does not replace `_moe_C` in the running
installation. Using the environment switch in normal serving requires a
core MoE extension built from the modified source. A single captured layer
does not qualify every layer's routing distribution or establish a serving
speedup; separate projection winners with different M tiles also cannot be
combined without accounting for route alignment and full-layer behavior.

The completed screen tested 2,530 candidate rows: 1,254 passed numerical
comparison and 1,276 were rejected as unsupported shape/resource choices
before timing, with no numerical failures. All 548 finalist rows passed
across GPU0 and GPU1, each using two reversed test orders and seven samples
of twenty calls. A separate ten-case compute-sanitizer memory check passed
with zero errors. Packed weights and scales from the base and MTP captures
were verified bit for bit; no checkpoint conversion was performed.

The inputs came from the first main-model MoE layer, TP rank0, with actual
routing captured at 512/1024/2048-token prefill and at decode/verification.
Five simultaneous requests produced multiple PP microbatches: M1 and M4
without MTP, and M4/M8/M12/M16 with three-token MTP. These matrix row counts
are not user counts. Each capture was replayed separately on each board;
the measured calls include both expert projections, activation, routing
alignment and the output sum, but exclude TP collectives and other layers.

| Captured full-layer call | Candidate | Less time, GPU0 / GPU1 |
| --- | --- | ---: |
| 512-token prefill, eager | M16/K64/N128, bounded allocation, 3 CTAs/SM | 3.48% / 4.30% |
| 1024-token prefill, eager | No consistent improvement | — |
| 2048-token prefill, eager | M32/K64/N128, bounded allocation, 2 CTAs/SM | 8.11% / 6.65% |
| Single-token decode, graph | M16, bounded allocation, automatic N/K choice | 8.22% / 8.78% |

The apparent 1024-token improvement was only 0.26–0.29% and changed sign
between test orders. MTP verifier improvements were smaller, approximately
1.5–3.7% across the observed microbatches. These results select useful
experimental candidates, not a universally optimal dispatch policy or
end-to-end serving rates. Both boards retained their 200 W caps; timed-run
snapshots showed 1455 MHz on GPU0 and 1380 MHz on GPU1 with 6500 MHz memory,
so absolute times are specific to the measured operating conditions.

The 2048-token winner changes M48 to M32 and uses 25,600 shared bytes.
The M16 decode candidate uses 21,248 bytes; its automatic policy requests
two CTAs for gate/up and three for down, while the 512-token candidate
explicitly requests three for both. These improvements combine tile and
dispatch changes; they do not isolate shared-memory allocation as the cause.
All 56 same-kernel/grid allocation comparisons on each GPU were bit-exact
when the aligned routes were held fixed. Across the finalists, maximum
absolute output difference from the production reference was 0.0004273
and maximum relative L2 was 0.001678; the acceptance tolerance was unchanged.

During this captured-layer experiment, the installed core MoE extension and
production Marlin defaults remained unchanged. The separate
`marlin_tile_and_shared_memory_qualification`
provenance record preserves both finalist runs, the build/input hashes,
resource audit and capture metadata. GPU execution was stopped after testing.

The RTX8000 launch profile enables `VLLM_RTX8000_MARLIN_TILES=1` and
defaults to a 2048-token scheduling budget. Revision 2 of the guarded policy
uses M32 for 1600 or 2048 input rows and M16 for one input row, with bounded
shared-memory allocation, K64/N128 tiles and occupancy-checked block counts.
It applies only to the measured SM75 FP16, per-channel INT8 expert geometry
(512 experts, top-10 routing, hidden width 2560, intermediate shard 320).
Other row counts and unsupported formats retain their original dispatch;
in particular, this does not adopt the experimental 512-token or MTP
verification choices. A one-row microbatch can also occur in a multi-user
request batch, so input rows must not be interpreted as user counts.

This policy does not require the broad experimental
`VLLM_MARLIN_MOE_EXACT_SHARED_MEM` switch; leave that switch disabled.
Set `VLLM_RTX8000_MARLIN_TILES=0` to restore the original Marlin policy.
The GDN switches and Marlin switch are independent, and MTP remains an
explicit serving option. Native QSA/dense activation in the example above
is unchanged.

The Marlin policy uses the installed `_rtx8000_marlin_C` companion. Revision 2
compiles its own guarded M16/M32 CUDA specializations and selector; it leaves
the original `_moe_C_stable_libtorch` operators intact. The generated kernel
header adds a CTA barrier before shared B storage becomes reduction scratch.
Generation checks the source-template hash and stops on an unexpected version.
Both the normal SM75 source build and incremental build use this generated
header and the same separately compiled specialization units. This is a CUDA
build, not the earlier host-dispatcher-only update:

```bash
cmake -S csrc/moe/marlin_moe_wna16/rtx8000 -B build/rtx8000-marlin -G Ninja \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DVLLM_MOE_LIBRARY="$PWD/vllm/_moe_C_stable_libtorch.abi3.so" \
  -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build build/rtx8000-marlin -j 4
cmake --install build/rtx8000-marlin
```

The companion and original MoE library must come from compatible source
and PyTorch builds. The Python loader requires `MARLIN_POLICY_REVISION=2`
and an operator schema matching the original. A missing or older companion,
or a differing schema, raises an error when an optimized shape is requested.
The companion also validates actual tensor dtypes, ranks, contiguous storage,
common CUDA device and routing/scale shapes before interpreting their pointers.

The earlier companion integration passed 17 targeted regression cases, including actual
`MarlinExperts.apply` execution for optimized and fallback row counts and
FakeTensor execution through both operator schemas. Eligibility excludes
unqualified activations, custom callbacks and LoRA. The original MoE library
and its 19 operator schemas were preserved; the companion adds one separately
named operator. That earlier companion reused the existing kernel selector;
those results do not validate the new guarded CUDA specializations.

The prior combined no-MTP serving profile measured 4916/4606/3981 prefill tokens/s
at 32K/64K/128K, versus 4786/4500/3893 with GDN improvements alone. Decode
measured 57.10 tokens/s for one user and 198.21 total for five users (39.64
mean per user), versus 56.22 and 197.27. All three retrieval checks passed;
diagnostic perplexity was 3.2722 versus 3.2640. These retain the same single
cold-prefill observations and two 8K-input/768-output decode repetitions,
so small rate differences are observations rather than confidence intervals.
The prefill trace confirmed 384 M32 Marlin GEMMs and 144 native GDN output
calls on every rank. Marlin's rank0 share of summed kernel time was 23.2%,
native QSA 19.4%, dense projections 15.5%, GDN state recurrence 8.1%, and
QSA indexing 5.9%.

With MTP3, the combined profile measured 4664/4339/3708 prefill tokens/s,
1.9–2.4% above the GDN-only MTP control. Decode measured 87.00 tokens/s
for one user versus 88.51 in the control, and 256.37 total for five users
(51.27 mean per user) versus 255.84. The single-user mean acceptance length
was 2.56–2.59 tokens per cycle versus 2.61–2.65 previously; generated
sequences also differed. These runs do not establish an MTP decode gain.
Retrieval passed at all three lengths, preemptions were zero, diagnostic
perplexity was 3.2698, and cache capacity remained 860,088 normalized tokens.
The corresponding no-MTP cache capacity also remained unchanged.

Both applied-profile runs used identical prompt-token hashes and the same
eight likelihood windows as their GDN-only controls. Their settings, runtime
binary hashes and results are preserved in the separate
`applied_gdn_and_marlin_policy` provenance record. Historical kernel-only
qualification records remain intact. These serving values precede the native
GDN state recurrence, extended dense/head plans, tuned QSA dispatch and guarded
Marlin revision described below; they must not be presented as measurements
of that newer combination.

### RTX8000 native state, verification and output-head components

The following results qualify individual kernels or one captured TP shard.
They are not per-user or aggregate serving token rates. The matrix row count
`M` is the number of scheduled tokens in that call, not the number of users;
pipeline microbatches and MTP verification can produce several different M
values for the same concurrent request count.

#### Native GDN prefill state

`VLLM_RTX8000_NATIVE_GDN_H=1` independently enables the native chunk-state
recurrence; its general default remains off and the qualified RTX8000 launcher
enables it. V16 is selected over V8. It keeps
the live initial, recurrent and final state in FP32, while preserving the
existing FP16 K/W/U operands,
chunk snapshots and updated-value rounding boundaries. Each CTA traverses a
request's chunks in order for its value-row tile. This does not fuse the
output kernel or replace the separate recurrent decode implementation.

Eligibility requires SM75, contiguous FP16 H24/Hg8/K128/V128 operands,
64-token chunks, FP32 scalar natural-log gates and an FP32 initial state.
Packed requests, partial chunks and zero-length requests are supported;
sequence indices and chunk offsets may independently be INT32 or INT64,
including the mixed widths produced by the real metadata helper. Other
layouts, vector gates, base-2 gates or omitted required state/output buffers
retain the existing path. The V16 kernel uses 116 registers/thread, 31,008
shared bytes and no spills, with two 128-thread CTAs resident per SM.

The synthetic GPU0 benchmark used real WY intermediates and native output
in both configurations. Its eager CUDA-event medians were:

| Scheduled tokens, one request | Tuned Triton state / native state | Full GDN chain before / after |
| ---: | ---: | ---: |
| 512 | 0.501 / 0.261 ms | 1.082 / 0.966 ms |
| 1024 | 0.823 / 0.368 ms | 1.402 / 1.003 ms |
| 2048 | 1.653 / 0.678 ms | 2.705 / 1.726 ms |

All 88 benchmark rows passed. Nine correctness cases covered both variants,
independent recurrence, tails, packed requests, mixed-width production
metadata and changed-input/metadata graph replay. Maximum direct relative
RMS error was 1.18e-5 against a 1e-3 limit; memory and race checks reported
zero errors or hazards. The production-metadata follow-up independently
confirmed dispatch and improvement at 2048 tokens. Small eager cases include
launch overhead; the graph measurements do not imply prefill graph serving.
The detailed record is `FlashNext-native-gdn-h-qualification.json`.

#### Native dense projections and vocabulary head

`VLLM_RTX8000_NATIVE_DENSE=1` enables the FP16 SM75 dense dispatcher, which
now covers nine additional measured shape/M points, including the TP2
output-head shard of 124,160 vocabulary rows. The general default is off;
the RTX8000 launcher enables it.
Variant 0 uses the existing FP32 vector reduction. Variant 8 uses Turing
WMMA with FP32 accumulation, an M16/N64/K128 tile and eight-element shared
memory skew. It consumes the existing weight layout without repacking or a
persistent second weight copy. The kernel implementation is shared through
a header, with a thin CUDA unit per specialization.

Actual activations and weights from TP rank2 were replayed on GPU0 and GPU1,
in forward and reverse orders, eagerly and through graphs, with both hot and
flushed caches. The selected plans and graph latency reductions with flushed
caches are:

| Local weight shape N × K | Input M | Selected variant | Less time, GPU0 / GPU1 |
| --- | ---: | ---: | ---: |
| 124160 × 2560, vocabulary head | 12 / 16 | 8 | 28.0% / 30.4–30.5% |
| 8192 × 2560 | 8 | 0 | 32.0% / 35.1% |
| 8192 × 2560 | 12 / 16 | 8 | 21.9–23.3% / 19.8–21.8% |
| 7296 × 2560 | 8 / 12 | 0 | 5.5–22.2% / 6.1–17.8% |
| 7296 × 2560 | 16 | 8 | 18.5% / 15.4% |
| 640 × 2560 | 8 | 0 | 20.7% / 24.0% |

For the head at M12/M16, measured time fell from approximately 1.52–1.60 ms
to 1.09–1.11 ms. Head M1/2/4/5 gains were below 1%, M8 regressed, and M3/M20
were unqualified, so these retain the standard linear path. Other unmeasured
shape/M combinations also retain their prior implementation. Registration
preserves the real embedding/head parameter and method; unsupported dtypes,
bias and noncontiguous inputs use the original method.

All 224 actual-input benchmark rows passed the independent FP32-product
reference, with maximum relative RMS 0.000291 against a 0.001 limit. Local
head argmax matched in every case on both boards; this does not establish
global generated-token identity. Six post-split regression cases passed,
including changed-input graph replay and real head/embedding fallbacks.
Torch-only native-kernel memory and race checks covered 64 cases per tool
with zero errors or hazards; they used the identical kernel body before its
compilation-unit split. See `RTX8000-Dense-qualification.json` for source,
binary and capture hashes and the separate component scopes.

#### QSA tiles and aligned-prefix scheduling

`VLLM_RTX8000_NATIVE_QSA=1` enables the native QSA path; its general default
is off and the RTX8000 launcher enables it. The tuned plan uses variant 4:
a 32-key tile, eight warps and register prefetch of V. It retains reuse across
the local attention heads.
The kernel uses 70 registers/thread and 29,056 shared bytes, allowing two
CTAs per SM. A convergent warp barrier before overwriting the shared row
maximum corrects a shared-memory hazard in both the original and tuned
kernels. Post-fix memory and race checks reported zero errors or hazards.

The measured split policy is 32 for M1/2/4/5, 16 for M8/12/16/20, four for
M448/512/1024, and one for M1600/2048. Other M values retain the prior
dispatch. The main post-fix screen passed 680 rows. At 64K context on GPU1,
M2048 eager latency fell from 12.862 to 11.321 ms; graph M12 fell from 0.180
to 0.110 ms. These combine the selected kernel and split policy, not just a
single instruction change.

Aligned hybrid prefix caching produces regular M1600 chunks at the 2048
scheduling budget and can produce M448 tails. A separate 64-row GPU0 screen
tested these observed row counts with synthetic valid QSA inputs. At 32K
and 128K context, M1600 fell from 9.31–9.45 to 8.26–8.40 ms (11.1–11.3%
less time); M448 fell from 3.00–3.05 to 2.31–2.33 ms (23.1–23.7%). Every
repeat improved, with no numerical failures. These event timings exclude
partial/LSE workspace allocation and compare the actual prior variant-0,
split-1 policy, not a separately optimized variant-0 split. The records are
`FlashNext-qsa-next-qualification.json` and
`FlashNext-qsa-actual-prefill-qualification.json`.

#### Greedy draft selection and guarded Marlin

`Qwen4ExpMTP.get_top_tokens` enables the existing
`use_local_argmax_reduction: true` option. It computes the same local head
projection, then exchanges each shard's winning score/token pair instead
of the entire vocabulary logits. It applies only to greedy draft selection;
the speculator rejects incompatible probabilistic drafting. A real TP2
regression matched full-logit argmax exactly on both ranks for batches of
one and five, cross-rank ties, poisoned padding, cropped vocabulary,
positive scaling, soft cap and random inputs. This test validates the shared
head/reduction implementation; it is not a serving speed measurement.

The Marlin revision-2 companion's CTA barrier prevents B readers from racing
with reduction-scratch writes. Qualification covered the full captured
M1/M1600/M2048 expert layer, with outputs bit-identical to the same-tile
control and zero memory/race errors. Forty timing rows passed across GPU0
and GPU1. At M1600, the guarded M32 full layer took 3.663/3.787 ms versus
3.958/4.075 ms for the prior M48 policy, reductions of 7.45%/7.07%. Against
an otherwise identical compiled control, the barrier cost approximately
0.2–0.4% in prefill; decode differences were order-sensitive. These results
cover one expert layer and one TP shard, excluding TP collectives. Other
shapes and formats still use the original library and are not covered by
this narrow safety correction. See `RTX8000-Marlin-safety-qualification.json`.

The installed revision-2 companion passed 19 targeted integration regressions,
including eight malformed direct-operator input cases. Real
`MarlinExperts.apply` replay at M1/M1600/M2048 recorded exactly two guarded
Marlin kernels and no stock Marlin kernel per shape, with passing output
checks. The original stock library remained unchanged.

The tested gate/up-plus-SiLU fusion is not enabled. It preserved the intended
rounding and avoided weight repacking, but the M2048 full layer was 4.71%/
4.01% slower on GPU0/GPU1. Its roughly 5% decode gate improvement also failed
to improve the full layer, which was about 1% slower. A stage-only win is
insufficient to select a serving policy; the fusion records remain separate
from the guarded unfused implementation.

### Combined native-kernel serving qualification

The combined native state, tuned QSA, extended dense/head, local draft argmax
and revision-2 Marlin profile was measured on the per-channel W8A16 checkpoint
with TP2/PP2. Cold comparisons explicitly disabled prefix caching and retained
the 2048-token budget, five-sequence limit, 90% memory budget and 200 W caps.
The benchmark implementation, timing prompt hashes and eight likelihood
windows are checked against the prior applied profile in the qualification
record. Profiling was performed after timed requests.

| Cold serving measurement | Prior no-MTP | New no-MTP | Prior MTP3 | New MTP3 |
| --- | ---: | ---: | ---: | ---: |
| 32K prefill, tokens/s | 4916 | 5168 | 4664 | 4919 |
| 64K prefill, tokens/s | 4606 | 4825 | 4339 | 4560 |
| 128K prefill, tokens/s | 3981 | 4140 | 3708 | 3870 |
| One-user decode, tokens/s | 57.10 | 57.18 | 87.00 | 87.84 |
| Five-user decode, total tokens/s | 198.21 | 198.63 | 256.37 | 263.53 |
| Five-user decode, mean tokens/s per user | 39.64 | 39.73 | 51.27 | 52.71 |

Prefill rates are single-request computed input tokens divided by server
prefill service time. Decode rates count actual delivered tokens in the
shared interval after all five prefills finish and before any request ends;
they exclude prefill. With MTP3, the five-request output rate including
prefill was 166.71 tokens/s, versus 160.05 previously. Each decode result is
the median of two 8K-input/768-output repetitions; each long-prefill point
is one request with 128 output tokens. These are measured observations,
not confidence intervals or a qualification of five simultaneous 128K
requests. The small single-user decode change does not establish a gain.

All six long-context retrieval checks passed and cache preemptions were zero.
The fixed 8184-token diagnostic perplexities were 3.2783 without MTP and
3.2711 with MTP3, versus 3.2722 and 3.2698 previously. Generated outputs and
MTP acceptance lengths vary, so these checks do not establish bitwise output
identity or broad model accuracy. Raw per-user rates and acceptance lengths
are retained in the qualification record.

The no-MTP trace verified native state/output, tuned QSA and guarded Marlin
on every rank. Native GDN state's summed GPU time fell 55.6% against the
matched prior trace. Its share of current summed kernel time is 3.75%; the
largest remaining shares are Marlin 24.44%, QSA 19.22%, dense projections
16.74% and NCCL 13.53%. These are summed GPU-work shares, not serving wall
time or isolated causal contributions to the observed speedup.

### RTX8000 225 W comparison

The same native binaries and prompt-token hashes were rerun at 225 W on all
four GPUs, with MTP3 and local draft argmax. The control is the preceding
completed 200 W run, not an interleaved randomized experiment.

| Measurement | 200 W | 225 W |
| --- | ---: | ---: |
| Cache-disabled 32K prefill, tokens/s | 4919 | 5075 |
| Cache-disabled 64K prefill, tokens/s | 4560 | 4691 |
| Cache-disabled 128K prefill, tokens/s | 3870 | 3975 |
| One-user decode, tokens/s | 87.84 | 88.41 |
| Five-user decode, total tokens/s | 263.53 | 271.27 |
| Five-user decode, mean per user | 52.71 | 54.25 |

With production aligned prefix caching enabled, cold prefill improved from
4724/4390/3740 to 4844/4503/3834 tokens/s at 32K/64K/128K, approximately
2.5%. Warm first-token latency was 0.419/0.855/1.128 seconds. Five requests
sharing an 8000-token prefix had 0.680-second median warm TTFT.
Retrieval and reset checks passed, with no cold-run cache preemptions.
Diagnostic perplexity was 3.2735 versus 3.2711 previously.

Peak sampled GPU temperatures were 68/66/63/62 degrees C. The hardware
thermal-slowdown flag was never active in the samples. Actual power samples
can exceed the configured cap momentarily; the cap is not an instantaneous
power-draw guarantee. Telemetry covers loading, warmup, timing and profiling,
so it does not establish phase-specific energy efficiency. Acceptance lengths
and generated text varied. One prefill observation per context and two decode
repetitions do not establish confidence intervals; the small one-user decode
change is not a demonstrated meaningful gain. The tested 225 W setting was
left applied after benchmark workers exited; reboot persistence is separate.

### LAN service deployment

The selected deployment continues at **200 W per GPU**. The 225 W results
above are retained as an experiment, not the selected production setting.
`examples/deployment/start_flashnext_lan.sh` runs the tested TP2/PP2,
per-channel W8A16, MTP3/local-argmax profile with prefix caching on
`192.168.0.51:8000`. It requires `VLLM_API_KEY`; development endpoints and
profiling are not enabled. The accompanying `flashnext.service` runs as
`jdillman`, reads a mode-0600 environment file at
`/home/jdillman/.config/flashnext/service.env`, and supervises all workers.
The API key is generated on the server and is never stored in this repository.

Before deployment, preserve the installed source, entire virtual environment,
toolchain, dependency inventory and native-library hashes in a private rollback
snapshot. Restore archives to their original absolute paths while the service
is stopped; the editable installation and interpreter contain absolute paths.
Keep future experiments in a separate installation. Systemd being active does
not imply model readiness: wait for `/health` and verify authenticated streaming.
Built-in API-key authentication protects inference routes, not every operational
route; this profile is intended for the trusted LAN only.

### RTX8000 benchmark controls and production features

The RTX8000 launcher now defaults to prefix caching. Set
`ENABLE_PREFIX_CACHING=0` explicitly for cold kernel/serving comparisons;
`ENABLE_PREFIX_CACHING=1` uses the model's aligned hybrid-state cache. Cold
and warm requests must be reported separately. A warm request's reused tokens
are not newly computed prefill tokens, so report its cache hits and TTFT rather
than dividing the entire prompt by the remaining prefill time.

The completed MTP3 cache qualification used untimed first-use warmups and
explicit cache resets before cold cases. Every cold/warm long-context
retrieval check passed, as did a reset followed by full recomputation:

| Input tokens | Cold computed prefill, tokens/s | Cold TTFT | Warm TTFT | Warm tokens computed / reused |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 4724 | 6.994 s | 0.423 s | 768 / 32000 |
| 65536 | 4390 | 15.040 s | 0.881 s | 1536 / 64000 |
| 131072 | 3740 | 35.262 s | 1.146 s | 1472 / 129600 |

Aligned caching rounds regular prefill steps to 1600 tokens despite the
2048-token budget. Cold prefill was 3.3–4.0% slower than this combination
with caching disabled; warm reuse avoids most of that work. The production
prefix trace confirms eight M1600 steps on every rank, native GDN state and
output, tuned QSA and guarded Marlin. Its FP16 hidden-state reductions use
the custom one-stage all-reduce: 392 calls on each first-stage GPU and 408
on each final-stage GPU, with no NCCL FP16 all-reduce in this capture.
The 7.8125 MiB payload fits the unchanged 8 MiB cutoff. NCCL remains for
other collectives and PP communication; larger mixed batches can still
exceed the cutoff.

Five 8192-token requests sharing the same 8000-token prefix computed 8960
tokens from an initially empty cache, then only 960 on repetition. Median
TTFT fell from 1.997 to 0.681 seconds (maximum 1.997 to 0.682 seconds).
This case benefits from reuse between requests even in the initially empty
batch; it is not a measurement of five unrelated cold prompts. Concurrent
prefill's computed-token count divided by summed request service times is
not aggregate system throughput and is not labeled as such here.

Cold/warm generated token sequences matched at 32K and 128K, but differed
at 64K and in the five-request case. Retrieval correctness is distinct from
bitwise text identity. Natural LRU eviction and five concurrent 128K
requests remain outside this qualification. Detailed source, build,
runtime and result records are in the `native_mtp_qsa_gdn_and_prefix`
provenance entry; previous records are retained separately.

To use the qualified long-context MTP3 configuration with the per-channel
checkpoint, retain the launcher's native kernels and prefix-cache defaults:

```bash
MAX_MODEL_LEN=131200 bash examples/deployment/serve_flashnext_rtx8000.sh \
  /path/to/Qwen3.8-Flash-Next-experts-W8A16-channel \
  --served-model-name Qwen3.8-Flash-Next-W8A16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
```

The launcher keeps MTP explicit so the same entry point can run a no-MTP
control. `VLLM_RTX8000_NATIVE_GDN_H=0` restores the previous state path;
other native switches remain independently controllable.

The profile has deliberately retained these controls:

| Setting | Benchmark behavior and reason |
| --- | --- |
| Prefix caching | Off only in explicit cold comparisons; on in the production launcher and cache qualification. |
| Torch compilation / prefill graphs | Compilation mode 0 and `FULL_DECODE_ONLY`; earlier coarse/fine prefill graph trials regressed concurrent prefill. Decode graphs remain enabled. |
| Speculative decoding | No-MTP control or fixed MTP3; dynamic draft depth and draft sparse-index reuse are not part of those comparisons. |
| Draft token selection | Qwen4ExpMTP now exposes the vocab-parallel greedy hook. Enable with `use_local_argmax_reduction: true`; it retains the local head projection and reduces only winning score/token pairs. |
| Dense INT8 activation quantization | Selective W8A8 remains off; the selected checkpoint uses per-channel W8A16 experts. |
| Communication | Eligible custom all-reduce stays enabled with its qualified 8 MiB cutoff. Larger cutoffs and forced NCCL settings were not adopted after the serving comparisons. |
| Resource limits | 2048-token scheduling budget, five sequences, 90% GPU memory utilization and existing 200 W caps are comparison controls, not claims of globally optimal settings. |
| Workload | Text-only, greedy requests using vLLM generation defaults. Multimodal, tool-use, structured-output and sampling workloads have not been qualified by these performance runs. |

Chunked prefill, continuous batching, asynchronous scheduling, NUMA binding,
CPU-offloaded ngram tables, hybrid cache management and decode CUDA graphs are
active. The launcher enables the qualified native dense and QSA paths explicitly;
unsupported shapes continue through the standard kernels. The standalone dense
candidate benchmark includes output-head shapes but does not itself enable an
unqualified plan in serving.

Other options are unavailable or unqualified, rather than intentionally disabled
for a favorable benchmark: the current ReplaySSM interface is not advertised by
Qwen4Exp, confidence-based adaptive verification is documented as DSpark-only
in this checkout (not a native-MTP switch), and the
draft-head FP8 compute path does not run natively on SM75. Prefix checkpoint
retention uses the upstream semantic-checkpoint default; periodic retention and
fine-grained shared-prefix checkpoints have not been tuned here.

## Data-parallel topologies (DeepSeek V4.1)

By default (`embedding_across_dp: false`), each DP replica keeps its own
TP-sharded copy of the tables — no cross-replica communication, but one host
copy per replica.

With `embedding_across_dp: true`, the hash heads are sharded across all
TP × DP ranks into a single table copy. Each step gathers the hash ids of
every co-located DP replica, each rank looks up the heads it owns, and the
rows are exchanged back. This trades per-step DP collectives for a much
smaller table footprint:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --engram-config '{"embedding_across_dp": true}'
```

`embedding_across_dp` is not supported with elastic expert parallelism yet.
Qwen4Exp PLE tables are ETP-sharded instead and honor only `cpu_offload`,
`disk_offload` and, on NVIDIA, `dp_shared_memory`.

## Sharing host tables across DP replicas

When the tables are CPU-offloaded and DP replicas are co-located on one node,
`dp_shared_memory` stores **one copy of each TP shard in `/dev/shm`**, shared
by all co-located replicas through `cudaHostRegister`. Each replica then
prefetches only its own tokens — no per-step Engram DP collectives — and host
memory drops from one table copy per replica to one per node.

`dp_shared_memory` defaults to enabled whenever the other settings allow it,
and falls back to per-replica (or DP-sharded) tables with a warning when:

- the DP replicas are not co-located on a single node,
- `/dev/shm` cannot hold the full tables — in containers, raise the limit
  with `--shm-size` or `--ipc=host`,
- expert parallelism is elastic, which is unsupported.

Requirements: `cpu_offload` enabled, `--data-parallel-size > 1`, and a shared
IPC namespace across the co-located replicas.

## Limitations

- CUDA-alike platforms only.
- DeepSeek V4.1 Engram does not support DBO or microbatching; disable
  `--enable-dbo` and set `--ubatch-size 0`.
- `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is
  unsupported with elastic expert parallelism.
