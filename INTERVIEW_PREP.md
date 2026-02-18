# RAG Agent 面试速答手册

## 1. 30 秒项目介绍
我实现了一个生产化思路的 RAG Agent：离线做多格式解析、语义+滑窗切块、向量入库；在线做双路检索（语义+元数据）并通过归一化、RRF 和重排融合；Agent 通过工具调用完成检索、摘要、政策抽取、OCR 和数据库读写；最后记录 trace、token/cost、延迟和 groundedness。

## 2. 架构拆解（按请求路径讲）
1. Ingest：Parser -> Chunker(500-1200 token + overlap) -> Embedder -> VectorStore。
2. Retrieve：metadata route + semantic route 并行召回，先扩召回再融合，优化 Recall@5。
3. Fusion：分数归一化 + RRF + reranker，输出 Top-K 证据块。
4. Agent：LangChain function calling，工具注册用 Pydantic 校验。
5. Answer：强制引用 chunk_id，缺证据就拒答。
6. Observability：trace_id 关联问题、答案、工具调用、延迟、token、cost、groundedness。

## 3. 关键设计题怎么答
- 为什么语义分块+滑窗：语义分块保证语义完整，滑窗 overlap 保证跨段上下文不断裂。
- 为什么双路检索：语义检索抗同义改写，元数据过滤保证域内精确性。
- 为什么 RRF：对不同检索路由的分值尺度不敏感，融合稳定。
- 为什么先扩召回再截断：先拿更多候选减少漏召回，再融合到 Top5。

## 4. 指标题怎么答
- Recall@5：通过 route oversampling（例如 final_k 的 4 倍候选）+ RRF 提升。
- Groundedness>95%：系统提示词强制逐条引用；答案句子与证据片段做重叠校验。
- E2E < 8s：限制 agent 迭代数、减少无效工具调用、记录并监控 P95 延迟。

## 5. 你可以直接说的取舍
- 为了可复现测试，默认用了 deterministic embedder + in-memory store。
- 生产可替换为 OpenAI Embedding + FAISS/Pinecone。
- Reranker 先用轻量关键词版本，后续可替换 cross-encoder。

## 6. 高频追问与短答
1. 如何避免幻觉：只允许基于 internal_search 的证据回答，且强制 citation。
2. 如何做工具参数安全：Pydantic v2 args_schema 做强校验。
3. 如何验证工具被正确选择：集成测试中用 Mock Executor 断言 `internal_search` 被调用。
4. 如何看系统健康：`/metrics` 看 total requests、avg/p95 latency、avg groundedness、token/cost。
5. 如何排查坏答案：`/traces/{trace_id}` 看工具输入输出、来源片段、引用是否匹配。

## 7. Demo 话术（2 分钟）
1. 调 `/ingest` 导入一份政策文档。
2. 调 `/query` 提问政策问题，展示返回 `citations`。
3. 调 `/traces/{trace_id}` 展示工具链路和 source snippets。
4. 调 `/metrics` 展示延迟和 groundedness。

## 8. 你必须记住的局限（诚实加分）
- groundedness 目前是启发式自动打分，不是人工标注金标准。
- OCR 真实效果依赖具体 Vision 模型与图像质量。
- 线上 SLA 需要压测与限流熔断策略配合。
