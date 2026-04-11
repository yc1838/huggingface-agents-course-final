# [SOTA Roadmap] GAIA Agent 极致形态演进计划

为了将当前的 GAIA Agent 从“能跑通”升级到“能打榜（SOTA）”的究极完全体，我们制定了以下四个维度的演进计划。

## 升级维度 1：Executor 自纠错内循环 (Inner Code Loop)
**目标：** 解决大模型写错代码（如语法错误、漏掉 import）导致的昂贵重算。
- **机制：** 在 Executor 执行 `run_python` 时，如果返回 Traceback，则调用廉价模型（Gemini Flash）进行修复，并原地重试（限 3 次）。
- **收益：** 减少 70% 的全局 Planner 重负载，大幅降低 API 成本和等待时间。

## 升级维度 2：多专家路由架构 (Multi-Agent Routing)
**目标：** 解决 Prompt 稀释导致的指令遵循能力下降。
- **机制：** 引进 `Router` 节点，根据任务领域将 State 分发给专用的 Specialist Executor（如 `Code_Specialist`、`Research_Specialist`）。
- **收益：** 每个节点的 Prompt 更短、更精准，大幅提升复杂任务（如音视频分析、学术检索）的成功率。

## 升级维度 3：长文档“降维打压” (Data Sandbox)
**目标：** 解决 10MB+ 文件导致的 Context Overflow。
- **机制：** 在 `FILE_SPECIALIST` 指令中增加强制约束，禁止读取全文。Executor 必须通过 Python 脚本进行 `sum()`、`grep()` 或提取摘要。
- **收益：** 保持 `Working Memory` 的极端整洁，防止幻觉生成。

## 升级维度 4：并行执行能力 (Parallel Fan-out)
**目标：** 提升多任务、多实体检索的效率。
- **机制：** 引入 LangGraph 的并发执行模式，使 Orchestrator 能同时下发多个独立子任务并在收尾处由 Reflector 聚合。
- **收益：** 响应速度提升 2~3 倍，适合处理 Level 3 中需要对比多份文献的场景。

