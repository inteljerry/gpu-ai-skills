# Coverage And Formulas

Use this reference when the user asks how the estimate is calculated,
why it differs from another calculator, or whether a specific model class
is covered.

## Coverage

| Model class | Behavior | Expected accuracy |
|---|---|---|
| Decoder-only LLMs such as Qwen, Llama, Mistral, and Gemma text models | Full estimate | About 5% |
| MoE models such as Qwen3-MoE, Mixtral, and DeepSeek-V3 | Full estimate, including shared experts | About 10% |
| Hybrid MoE models such as DeepSeek-V2/V3/V4 and Mistral-Large-3 | Counts dense and MoE layers separately via `first_k_dense_replace` | About 10% |
| MoE with a separate expert dtype such as DeepSeek-V4-Flash | Prices expert FFNs at `expert_dtype`, the rest at `quant_method` | About 5% |
| VLMs such as Qwen2-VL, Gemma-3 vision, LLaVA, and Nemotron-Omni | Adds vision tower and supports `text_config` or `llm_config` | Weights are exact; runtime caveats apply |
| Mistral `params.json` models | Reads `params.json` when `config.json` is absent | About 10% |
| Diffusion models | Refuses as a full estimate | Use empirical benchmarking |

The script reads standard Hugging Face `config.json`, Mistral
`params.json`, or a local JSON path. It detects diffusion repos from
`model_index.json` and exits with a routing message.

## Core Formula

```text
VRAM = weights + kv_cache(ctx, concurrency) + activations + framework
usable_vram = physical_vram * gpu_memory_utilization
fits = VRAM <= usable_vram
```

Weights are estimated from config dimensions:

```text
head_dim = config.head_dim or hidden_size / num_attention_heads
q_proj_dim = num_attention_heads * head_dim
kv_proj_dim = num_key_value_heads * head_dim

attention_per_layer =
    hidden * q_proj_dim
  + hidden * kv_proj_dim
  + hidden * kv_proj_dim
  + q_proj_dim * hidden

dense_ffn_per_layer = 3 * hidden * intermediate_size
moe_ffn_per_layer =
    num_experts * 3 * hidden * moe_intermediate_size
  + num_shared_experts * 3 * hidden * moe_intermediate_size

kv_cache =
  2 * num_layers * num_key_value_heads * head_dim
  * bytes_per_kv_dtype * ctx * concurrency
```

The script includes embeddings and untied LM head when applicable. For
hybrid MoE models, dense replacement layers use dense FFN dimensions and
remaining layers use MoE expert dimensions.

Activation memory is a bounded estimate:

```text
activations ~= 2 * concurrency * ctx * hidden_size * bytes_per_param + 512 MiB
```

Runtime framework overhead is a floor estimate:

| Runtime | Overhead |
|---|---:|
| `vllm` | About 2.0 GiB |
| `sglang` | About 1.5 GiB |
| `torch` | About 0.8 GiB |

## Bytes Per Parameter

| Quant | Bytes per parameter |
|---|---:|
| `bf16` / `fp16` | 2.00 |
| `fp8` / `int8` | 1.00 |
| `int4` | 0.55 |
| `int3` | 0.42 |
| `int2` | 0.30 |
| `mxfp4` | 0.55 |
| `fp4` | 0.55 |

`int4` includes typical scale and zero overhead for grouped
quantization with group size 128. KV dtype bytes are 2 for `bf16` and
`fp16`, and 1 for `fp8` or `int8`.

`fp4` covers NVFP4 and DeepSeek-V4's `expert_dtype: fp4`. The 0.55 figure
is deliberately conservative: DeepSeek-V4-Flash measures 0.531 effective
bytes per param (4 bits of weight plus one `ue8m0` scale byte per 32
weights), so the estimate runs ~4% heavy rather than light.

Any dtype narrower than 16-bit auto-pairs KV with `fp8`. That set is
derived from this table (`SUB16_QUANTS`), not hardcoded, so adding a row
here cannot leave a new dtype defaulting to `bf16` KV.

`quant_method` spellings are normalized before lookup:
`nvfp4`/`modelopt_fp4` to `fp4`, `awq`/`gptq` to `int4`.

## Important Modeling Details

Use explicit `head_dim` from config when present. Some models use a
larger head dimension than `hidden_size / num_attention_heads`; ignoring
that undercounts KV cache and Q/O projection parameters.

For MoE models, prefer `moe_intermediate_size`, `expert_hidden_dim`, or
the equivalent MoE-specific FFN field over dense `intermediate_size`.
Shared experts are always active in addition to routed experts.

## Mixed And Per-Component Precision

A checkpoint's weights are often not all one dtype. Two independent
config mechanisms express that, and a model may use both:

**1. `quantization_config.modules_to_not_convert`** (e.g.
openai/gpt-oss-20b) names components held at full precision while the
rest is quantized. The script recognizes embeddings, attention, and
router entries.

**2. A separate `expert_dtype`** (e.g. deepseek-ai/DeepSeek-V4-Flash)
gives the MoE expert FFNs their own dtype. DeepSeek-V4-Flash declares
`quantization_config.quant_method: fp8` *and* `expert_dtype: fp4`: only
the ~12 B non-expert params are fp8, while the 278 B of experts are 4-bit.

Reading `quant_method` alone and applying it uniformly prices 96% of that
model at double its real size -- 270.9 GiB of "weights" against 148.7 GiB
of actual safetensors -- which is enough to flip an 8-XPU verdict from
FITS to DOES NOT FIT. Always report which dtype landed on the experts.

Either way the script prints a component-level weight breakdown. With
`expert_dtype`, the rows are embeddings at `bf16`, non-expert weights at
`quant_method`, and expert FFNs at `expert_dtype`.

`expert_dtype` applies **whenever the requested weight dtype still matches
the config** -- auto-detected, or typed explicitly as the same dtype.
`--quant fp8` against a `quant_method: fp8` checkpoint keeps its fp4
experts, because that argument restates the config rather than overriding
it.

Only a *differing* `--quant` suppresses the split, since it posits a
uniform re-quantization the config no longer describes. That case prints
`hypothetical:` on the `Expert weights:` line along with the as-shipped
weight figure for comparison, so a re-quantization estimate cannot be
misread as the checkpoint's real size.

## Tested Model Families

Dense coverage includes Qwen2.5, Llama 3.1/3.3, Gemma-2, and
Nemotron-70B style configs.

MoE coverage includes Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral,
DeepSeek-V3/V4-style hybrid MoE, DeepSeek-V4-Flash (fp8 + fp4 experts),
and Mistral-Large-3 style `params.json` configs.

Multimodal coverage includes Qwen2-VL, Gemma vision configs, LLaVA-like
configs, and Nemotron-Omni style `llm_config` layouts.

## Validation

Weight estimates are checked against the real root-level safetensors byte
totals of each repo -- the only ground truth available without
downloading the models:

| Model | Estimate | On disk | Error |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct (bf16) | 14.19 GiB | 14.19 GiB | +0.0% |
| Qwen3-30B-A3B (bf16 MoE) | 56.85 GiB | 56.87 GiB | -0.0% |
| gpt-oss-20b (mxfp4 experts) | 13.13 GiB | 12.82 GiB | +2.4% |
| DeepSeek-V4-Flash (fp8 + fp4 experts) | 155.35 GiB | 148.66 GiB | +4.5% |

Reproduce a row by summing the repo's root-level `*.safetensors` sizes
from `https://huggingface.co/api/models/<id>?blobs=true` and comparing
against the script's `Weights` line at `--tp 1`. Count root level only:
some repos ship a duplicate copy in a subdirectory (gpt-oss-20b has
`original/`), which doubles a naive sum.

The regression suite lives at the repo root and covers dimension
parsing, KV ground truth, and verdicts:

```sh
python3 -m pytest tests/test_fit.py -q
```

It does not yet assert the mixed-precision paths above -- there is no
`expert_dtype` case in it. Add one there, using its pinned-revision
config fetch, rather than hand-computing a weight figure in an answer.
