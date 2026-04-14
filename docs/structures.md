# Mini-SGLang 的结构

## 系统架构

Mini-SGLang 被设计为一个分布式系统，用于高效处理大语言模型（LLM）推理。它由多个相互协作、彼此独立的进程组成。

### 核心组件

- **API Server**：用户入口。它提供兼容 OpenAI 的 API（例如 `/v1/chat/completions`），用于接收提示词并返回生成文本。
- **Tokenizer Worker**：将输入文本转换为模型能够理解的数字（tokens）。
- **Detokenizer Worker**：将模型生成的数字（tokens）再转换回人类可读的文本。
- **Scheduler Worker**：核心工作进程。在多 GPU 场景下，每张 GPU 都会对应一个 Scheduler Worker（称为一个 **TP Rank**）。它负责该 GPU 上的计算调度和资源分配。

### 数据流

这些组件之间使用 **ZeroMQ（ZMQ）** 传递控制消息，使用 **NCCL**（通过 `torch.distributed`）在 GPU 之间交换体量较大的张量数据。

![进程总览图](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期：**

1. **User** 向 **API Server** 发送请求。
2. **API Server** 将请求转发给 **Tokenizer**。
3. **Tokenizer** 将文本转换为 tokens，并发送给 **Scheduler（Rank 0）**。
4. **Scheduler（Rank 0）** 将请求广播给所有其他 Scheduler（如果启用了多 GPU）。
5. **所有 Scheduler** 对请求进行调度，并触发各自本地的 **Engine** 计算下一个 token。
6. **Scheduler（Rank 0）** 收集输出 token，并将其发送给 **Detokenizer**。
7. **Detokenizer** 将 token 转换为文本，并发回 **API Server**。
8. **API Server** 将结果以流式方式返回给 **User**。

## 代码组织（`minisgl` 包）

源码位于 `python/minisgl`。下面是面向开发者的模块划分说明：

- `minisgl.core`：提供核心数据类 `Req` 和 `Batch`，用于表示请求状态；提供 `Context` 类，用于保存推理上下文的全局状态；提供 `SamplingParams` 类，用于保存用户传入的采样参数。
- `minisgl.distributed`：提供张量并行中的 all-reduce 和 all-gather 接口，以及保存 TP 信息的 `DistributedInfo` 数据类。
- `minisgl.layers`：实现构建支持 TP 的 LLM 所需的基础模块，包括 linear、layernorm、embedding、RoPE 等。它们共享定义在 `minisgl.layers.base` 中的基础类。
- `minisgl.models`：实现 LLM 模型，包括 Llama 和 Qwen3，同时也定义了从 Hugging Face 加载权重和对权重进行分片的工具。
- `minisgl.attention`：提供 attention Backend 的接口，并实现 `flashattention` 和 `flashinfer` 等后端。它们由 `AttentionLayer` 调用，并使用保存在 `Context` 中的元数据。
- `minisgl.kvcache`：提供 KVCache 池和 KVCache 管理器的接口，并实现 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`：提供一组工具函数，包括日志初始化以及对 zmq 的封装。
- `minisgl.engine`：实现 `Engine` 类。它是单进程中的一个 TP worker，负责管理模型、上下文、KVCache、attention backend 以及 CUDA graph replay。
- `minisgl.message`：定义 `api_server`、`tokenizer`、`detokenizer` 和 `scheduler` 之间通过 zmq 交换的消息。所有消息类型都支持自动序列化和反序列化。
- `minisgl.scheduler`：实现 `Scheduler` 类。它运行在每个 TP worker 进程中，并管理对应的 `Engine`。其中 rank 0 的 scheduler 负责接收来自 tokenizer 的消息、与其他 TP worker 上的 scheduler 通信，并将消息发送给 detokenizer。
- `minisgl.server`：定义命令行参数以及 `launch_server`，用于启动 Mini-SGLang 的所有子进程；同时还在 `minisgl.server.api_server` 中实现了一个作为前端的 FastAPI 服务，提供如 `/v1/chat/completions` 这样的接口。
- `minisgl.tokenizer`：实现 `tokenize_worker` 函数，用于处理 tokenize 和 detokenize 请求。
- `minisgl.llm`：提供 `LLM` 类，作为一个 Python 接口，便于用户与 Mini-SGLang 系统交互。
- `minisgl.kernel`：实现自定义 CUDA kernel，并通过 `tvm-ffi` 提供 Python 绑定和 JIT 接口支持。
- `minisgl.benchmark`：基准测试工具。
