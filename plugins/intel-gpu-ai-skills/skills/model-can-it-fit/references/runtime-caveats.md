# Runtime Caveats

Use this reference when choosing dtype/KV options, explaining VLM or
diffusion behavior, or deciding whether the user needs a benchmark
instead of a fit estimate.

## Quick Arc Pro B70 Reference

Regenerate this table for the current script and model set:

```sh
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
    --table --runtime vllm --device-vram-gb 32
```

`--device-vram-gb 32` is deliberate here: this table is pinned to the B70
so it regenerates identically from any host. Do not swap in the VRAM of
whatever card the regenerating machine happens to have -- that silently
retargets the table. For the user's actual hardware, run the script
per-model with a measured value instead (see the skill's "Measure VRAM
First").

Typical Arc Pro B70 (32 GB) planning outcomes:

| Model | bf16 / 8K / c=1 | bf16 / 4K / c=4 | int4 / 32K / c=4 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | fits | fits | fits |
| Qwen2.5-7B-Instruct | fits | fits | fits |
| Llama-3.1-8B-Instruct | fits | fits | fits |
| Qwen2.5-14B-Instruct | fits or tight | tight | fits |
| Qwen2.5-32B-Instruct | OOM from weights | OOM from weights | fits |
| GPT-OSS-120B | OOM from weights | OOM from weights | needs multi-XPU TP |

Treat this table as orientation only. Use the script for the user's
actual context, concurrency, runtime, and memory-utilization target.

## Dtype And KV Choices

`bf16` is the safest default on XPU.

For vLLM launch planning, set `--gpu-memory-utilization` to the same
value as the planned `vllm serve` command. The script default of `1.0`
means physical-fit only.

For `fp8`, pair with `--kv-cache-dtype fp8` when serving. The vLLM launch may
also need an attention backend that supports fp8 KV on XPU.

For `int4` AWQ, GPTQ, or AutoRound models, the script auto-pairs KV with
`fp8`. Validate live runtime logs for the expected int4 kernel path; if
the runtime falls back to a wider activation path, activation memory and
throughput may differ from the estimate.

`int3` and `int2` are AutoRound-oriented planning modes. Validate output
quality before trusting a deployment based on those sizes.

For `fp4` weights -- NVFP4 checkpoints, or MoE models whose config sets
`expert_dtype: fp4` such as DeepSeek-V4-Flash -- the memory estimate is
only half the question. Battlemage has no native fp4 datapath, so whether
those bytes stay 4-bit on device depends on the runtime's kernel support.
If the runtime upcasts expert weights to fp8 or bf16 at load, real weight
memory is 2x or 4x the estimate and a tight multi-XPU fit fails at engine
init. Say so on any fp4 verdict, and confirm the architecture is
registered before calling it deployable:

```sh
python3 -c "from vllm.model_executor.models.registry import ModelRegistry as M; \
print([a for a in M.get_supported_archs() if 'eepseek' in a])"
```

A FITS verdict for an fp4 model is a claim about bytes, not about kernels.

## VLM Caveats

The script includes the vision tower weights when `vision_config`,
vision-language architecture names, or VLM model types are present.

It does not fully model:

- image-token KV growth from resolution and image count
- vision encoder activation peaks
- processor-side memory spikes

For tight VLM fits, leave extra headroom. As a planning rule, add about
1-3 GiB per concurrent image-bearing request, then verify with
`torch-xpu-bench` at the user's image resolution.

## Diffusion Caveats

The script refuses diffusion pipelines when the root has
`model_index.json` but no top-level LLM config. Diffusion peak memory is
dominated by latent resolution, steps, scheduler, and pipeline component
activation peaks, so a config-only estimate is not reliable.

For a floor estimate, point the script at a component config such as the
UNet or transformer subdirectory. For the real answer, use
`torch-xpu-bench` with one run and read peak XPU memory.

## What This Skill Does Not Predict

- measured tokens/sec, TTFT, TPOT, or ITL
- runtime graph-capture or `torch.compile` buffers
- prefix-cache buffers when prefix caching is enabled
- pipeline-parallel sharding
- diffusion peak memory
- correctness or output quality after aggressive quantization

Use benchmark/profile skills for measured runtime behavior.

## External References

- apxml VRAM calculator, useful CUDA analogue: <https://apxml.com/tools/vram-calculator>
- vLLM XPU kernels: <https://github.com/vllm-project/vllm-xpu-kernels>
- Intel AutoRound: <https://github.com/intel/auto-round>
