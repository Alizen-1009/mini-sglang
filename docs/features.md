# Mini-SGLang 的功能特性

## 在线服务

Mini-SGLang 支持通过兼容 OpenAI 的 API Server 进行在线服务。它提供标准的 `/v1/chat/completions` 接口，可以无缝接入现有工具和客户端。若要查看详细的命令行参数和配置选项，请运行 `python -m minisgl --help`。

## 交互式 Shell 模式

为了便于演示和测试，项目提供了交互式 shell 模式。在该模式下，用户可以直接输入提示词，LLM 会实时生成回复。shell 会自动缓存聊天历史以维持上下文。如果想清空对话历史并开启新的会话，可以使用 `/reset` 命令。

示例：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务

为了在多张 GPU 上扩展性能，Mini-SGLang 支持张量并行（Tensor Parallelism, TP）。你可以通过 `--tp n` 参数启用分布式服务，其中 `n` 表示并行度，也就是使用的 GPU 数量。

## 支持的模型

目前框架支持以下稠密模型架构：

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series

## 分块 Prefill

默认启用了分块 Prefill（Chunked Prefill），这项技术最早由 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 提出。该特性会在 prefill 阶段将长提示词拆分成更小的块，从而显著降低峰值显存占用，并避免长上下文服务时出现内存溢出（OOM）。你可以通过 `--max-prefill-length n` 配置块大小。需要注意的是，不建议将 `n` 设置得过小（例如 128），因为这可能会显著降低性能。

## Page Size

你可以通过 `--page-size` 参数指定系统使用的 page size。

## Attention 后端

Mini-SGLang 集成了多个高性能 attention kernel，包括 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention)（`fa`）、[`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)（`fi`）以及 [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM)（`trtllm`）。它支持在 prefill 和 decode 阶段分别使用不同的后端，以最大化整体效率。例如，在 NVIDIA Hopper GPU 上，默认会使用 `FlashAttention 3` 处理 prefill，使用 `FlashInfer` 处理 decode。

你可以通过 `--attn` 参数指定后端。如果提供两个值（例如 `--attn fa,fi`），第一个表示 prefill 使用的后端，第二个表示 decode 使用的后端。需要注意的是，某些 attention 后端可能会覆盖用户指定的 page size（例如 `trtllm` 仅支持 16、32、64 的 page size）。

## CUDA Graph

为了尽可能减少 decode 阶段的 CPU launch 开销，Mini-SGLang 支持 CUDA graph 的捕获与重放。该特性默认开启。你可以通过 `--cuda-graph-max-bs n` 设置 CUDA graph 捕获所支持的最大 batch size。将 `n` 设置为 `0` 会关闭该特性。

## Radix Cache

Mini-SGLang 采用了 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，实现了基于 Radix Tree 的缓存机制来管理 Key-Value（KV）Cache。这样可以在不同请求共享前缀时复用 KV Cache，从而减少重复计算。该特性默认开启，但你也可以通过 `--cache naive` 切换为朴素的缓存管理策略。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*来自 [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/) 的 Radix Attention 示意图。*

## Overlap Scheduling

为了进一步降低 CPU 开销，Mini-SGLang 使用了 Overlap Scheduling，这是一项由 [NanoFlow](https://arxiv.org/abs/2408.12757) 提出的技术。它将 CPU 调度开销与 GPU 计算重叠执行，从而提升整体系统吞吐。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*来自 [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) 的 Overlap Scheduling 示意图。*
