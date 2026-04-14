# Mini-SGLang 学习清单

## 学习目标
- [ ] 先搞清楚这个项目的整体请求链路：`API/Tokenizer -> Scheduler -> Engine -> Model -> KV Cache -> Detokenizer`
- [ ] 再搞清楚它和“普通单卡推理脚本”最大的区别：`调度`、`KV Cache 管理`、`Chunked Prefill`、`Overlap Scheduling`
- [ ] 最后再去看性能相关细节：`Attention Backend`、`CUDA Graph`、`Kernel`

## 第 0 阶段：建立全局地图
- [ ] 阅读 [README.md](/Users/alizen/study_files/LongRoad/mini-sglang/README.md:11)，先确认这是一个“可读的高性能 SGLang 参考实现”
- [ ] 阅读 [docs/structures.md](/Users/alizen/study_files/LongRoad/mini-sglang/docs/structures.md:3)，把系统角色和请求生命周期先看明白
- [ ] 阅读 [docs/features.md](/Users/alizen/study_files/LongRoad/mini-sglang/docs/features.md:29)，先记住四个关键词：`Radix Cache`、`Chunked Prefill`、`Overlap Scheduling`、`TP`
- [ ] 看完后，自己能口头讲清楚“一次请求从 prompt 到输出 token 的 8 个步骤”

## 第 1 阶段：从最小主线切入
- [ ] 先读 [python/minisgl/llm/llm.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/llm/llm.py:28)
- [ ] 重点看 `generate()`、`offline_receive_msg()`、`offline_send_result()`
- [ ] 理解为什么它适合作为源码入口：去掉了多进程和服务层，只保留核心调度链路
- [ ] 看完后，自己能回答：`LLM` 为什么直接继承 `Scheduler`？

## 第 2 阶段：吃透核心数据结构
- [ ] 阅读 [python/minisgl/core.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/core.py:15)
- [ ] 重点理解 `SamplingParams`
- [ ] 重点理解 `Req` 的几个字段：`cached_len`、`device_len`、`max_device_len`
- [ ] 重点理解 `Batch` 为什么要区分 `prefill` 和 `decode`
- [ ] 重点理解 `Context` 为什么是全局状态入口
- [ ] 看完后，自己能画出 `Req` 在一次生成中的状态变化

## 第 3 阶段：主攻 Scheduler
- [ ] 阅读 [python/minisgl/scheduler/scheduler.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/scheduler/scheduler.py:45)
- [ ] 第一遍只盯 4 个函数：`_process_one_msg()`、`_schedule_next_batch()`、`_forward()`、`_process_last_data()`
- [ ] 第二遍再看 `overlap_loop()` 和 `normal_loop()` 的区别
- [ ] 搞清楚 scheduler 到底做了什么：收请求、组 batch、分配 cache、调用 engine、处理结果、释放资源
- [ ] 看完后，自己能回答：为什么这个项目里 `Scheduler` 才是真正的核心？

## 第 4 阶段：理解 Prefill/Decode 调度策略
- [ ] 阅读 [python/minisgl/scheduler/prefill.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/scheduler/prefill.py:116)
- [ ] 理解 `PrefillAdder` 如何做预算控制
- [ ] 理解 `ChunkedReq` 的作用，以及为什么 chunked prefill 不直接进入 decode
- [ ] 阅读 [python/minisgl/scheduler/decode.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/scheduler/decode.py:8)
- [ ] 看完后，自己能回答：为什么长 prompt 要拆 chunk，收益是什么，代价是什么？

## 第 5 阶段：理解 KV Cache 和 Page 管理
- [ ] 阅读 [python/minisgl/scheduler/cache.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/scheduler/cache.py:15)
- [ ] 阅读 [python/minisgl/kvcache/radix_cache.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/kvcache/radix_cache.py:101)
- [ ] 理解 `free_slots`、`page_size`、`page_table` 分别负责什么
- [ ] 理解 `match_prefix -> lock -> allocate -> insert_prefix -> evict` 这一整条链
- [ ] 看完后，自己能回答：`page table` 和 `radix cache` 的职责边界是什么？

## 第 6 阶段：理解 Engine 是怎么真正跑起来的
- [ ] 阅读 [python/minisgl/engine/engine.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/engine/engine.py:29)
- [ ] 重点看模型初始化、KV Cache 初始化、page table 初始化、attention backend 初始化
- [ ] 重点看 `forward_batch()`
- [ ] 理解 `GraphRunner` 在这里扮演什么角色，不必一开始深入实现
- [ ] 看完后，自己能回答：一次 batch 真正进入模型前，Engine 做了哪些准备？

## 第 7 阶段：理解模型层如何接入框架
- [ ] 先选一个模型实现阅读，推荐 [python/minisgl/models/qwen3.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/models/qwen3.py:18)
- [ ] 配合阅读 [python/minisgl/layers/attention.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/layers/attention.py:18)
- [ ] 配合阅读 [python/minisgl/attention/__init__.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/attention/__init__.py:52)
- [ ] 理解模型 forward 为什么不显式传很多参数，而是从 `Context` 中取 batch/meta
- [ ] 看完后，自己能回答：这个项目是怎么把“模型结构”和“服务调度”解耦的？

## 第 8 阶段：回头看服务封装
- [ ] 阅读 [python/minisgl/server/launch.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/server/launch.py:40)
- [ ] 阅读 [python/minisgl/server/api_server.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/server/api_server.py:100)
- [ ] 阅读 [python/minisgl/tokenizer/server.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/tokenizer/server.py:30)
- [ ] 阅读 [python/minisgl/server/args.py](/Users/alizen/study_files/LongRoad/mini-sglang/python/minisgl/server/args.py:54)
- [ ] 搞清楚多进程、ZMQ、tokenize/detokenize worker、rank0 广播分别在做什么
- [ ] 看完后，自己能回答：如果去掉 FastAPI，这个系统还能不能跑？为什么？

## 第 9 阶段：用测试反向验证理解
- [ ] 阅读 [tests/core/test_scheduler.py](/Users/alizen/study_files/LongRoad/mini-sglang/tests/core/test_scheduler.py:26)，用它理解最小端到端链路
- [ ] 阅读 [tests/core/test_cache_allocate.py](/Users/alizen/study_files/LongRoad/mini-sglang/tests/core/test_cache_allocate.py:58)，用它验证自己对 page 对齐和 eviction 的理解
- [ ] 如果你后面想深挖 kernel，再看 `tests/kernel/*`

## 建议节奏
- [ ] 第 1 天：`README + docs + llm.py + core.py`
- [ ] 第 2 天：`scheduler.py` 主线
- [ ] 第 3 天：`prefill.py + decode.py + cache.py`
- [ ] 第 4 天：`radix_cache.py + tests/core/test_cache_allocate.py`
- [ ] 第 5 天：`engine.py`
- [ ] 第 6 天：`models/qwen3.py + layers/attention.py + attention backend`
- [ ] 第 7 天：`server/* + tokenizer/* + tests/core/test_scheduler.py`

## 学习时要持续追问自己的 5 个问题
- [ ] 请求状态在哪里流转？
- [ ] 哪些数据在 CPU，哪些在 GPU？
- [ ] 哪一步决定 batch 组成？
- [ ] 哪一步决定 cache 复用与释放？
- [ ] 哪些优化属于“调度层”，哪些属于“算子层”？
