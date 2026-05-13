# 第一阶段性能优化计划

## 背景

当前系统在 5 到 7 个任务并发执行时，页面响应会明显变慢，并且容易触发前端 30 秒 API 超时。已有数据库规模约几十万行、SQLite 文件约 6GB。这个规模本身不应导致系统几乎不可用，主要瓶颈来自运行期的访问模式：

- 执行器持续向 SQLite 逐条写入 execution 记录。
- 后台监控线程、任务详情页、任务列表页、成本页会在运行中反复触发统计查询。
- 多个统计/报告接口会全量读取某个 run 的 execution 记录，并反序列化大 JSON 字段。
- FastAPI 路由是 `async def`，但内部执行同步 SQLAlchemy/SQLite 和 CPU 统计，慢请求会阻塞事件循环。

第一阶段目标是在不拆大字段表、不做大规模历史数据迁移的前提下，优先消除热路径全量扫描和高频重算，降低 SQLite 锁竞争，恢复多任务运行时的页面可用性。

## 不在第一阶段做的事情

第一阶段不做 `executions` 大字段拆表。

原因：

- 需要迁移已有 6GB SQLite 数据，迁移耗时、磁盘占用和锁表风险较高。
- `RunRecord`、导出、报告、分析等路径当前都依赖完整记录对象，拆表会扩大改动面。
- 当前最大瓶颈是运行中自动重算和全量反序列化，先优化这些路径收益更高、风险更低。

## 现状热点

### 后端热点

1. `Storage.fetch_run_records(run_id)` 会执行 `WHERE run_id = ? ORDER BY id ASC`，然后 `.all()` 拉取全部 execution 行，并反序列化 `usage_json`、`request_params_json`、`reasoning_json`、`content_json`、`extra_json`。

   主要调用点：

   - `TaskService._update_progress_snapshot`
   - `TaskService.get_stats`
   - `TaskService.get_quick_report`
   - `AnalysisService.get_summary`
   - `AnalysisService.get_timeseries`
   - `AnalysisService.compare_executors`
   - `AnalysisService.detect_anomalies`
   - 导出接口

2. 运行中监控线程每 2 秒轮询一次；每 5 次执行一次完整进度刷新，约每 10 秒全量读取当前 run 的 records。

3. `get_total_cost()` 会读取全部 runs，并对全部 executions 做 `SUM(total_cost)`。任务列表、仪表盘、成本页都会调用它。

4. 执行器每完成一条记录就 `insert_record()`，每条记录单独 `commit()`。多进程、多线程并发写 SQLite 时会放大写锁竞争。

5. SQLite PRAGMA 只在初始化阶段的一条连接上执行，没有通过 SQLAlchemy connect event 保证每个连接都设置 `busy_timeout` 等参数。

6. 部分查询缺少贴合访问模式的组合索引，例如 `WHERE run_id = ? ORDER BY id`、错误分页、成本汇总。

### 前端热点

1. 任务详情页首屏会并发加载：

   - task
   - progress
   - report
   - errors

   其中 report 是重查询。

2. 任务详情页在任务运行中：

   - 每 5 秒刷新 task + progress。
   - 每 20 秒刷新 report + errors。

   report 当前每次都会重新计算，不走缓存。

3. 任务列表页和仪表盘加载时会请求总成本，总成本接口当前可能扫描全库 executions。

## 第一阶段目标

1. 运行中页面核心接口稳定在秒级以内返回，避免 30 秒超时。
2. 运行中不再自动触发完整报告重算。
3. progress 接口不再依赖全量 `fetch_run_records()`。
4. 总成本接口不再扫描全量 executions。
5. 降低 SQLite 锁等待和写入争用。
6. 保留现有产品能力：任务列表、任务详情、实时进度、错误列表、报告、导出、成本页都继续可用。

## 优化方案

### 1. 前端停止运行中自动刷新 report

调整范围：

- `frontend/src/pages/TaskDetail.tsx`

建议行为：

- 进入详情页时：
  - 对 `completed`、`failed`、`cancelled` 任务加载 report。
  - 对 `running`、`paused`、`pending`、`scheduled` 任务不自动加载 report，除非已有缓存快照或用户点击“刷新报告”。
- 运行中定时器：
  - 保留每 5 秒刷新 task + progress。
  - 保留每 20 秒刷新 errors，或改为 30 秒。
  - 移除每 20 秒 `loadReport()`。
- 页面文案从“每次请求都会重新计算”改成“运行中展示轻量实时进度；完整报告可手动刷新或任务完成后生成”。

收益：

- 立即减少运行中最重的 API 请求。
- 对产品功能影响小，完整报告仍然可手动查看。

牺牲：

- 运行中的报告图表不再自动实时精确刷新。

风险控制：

- 用户仍可以点击“刷新报告”主动计算。
- 任务完成后自动或首次进入详情时加载最终报告。

### 2. progress 改成轻量 SQL 聚合

调整范围：

- `src/llmperf/records/storage.py`
- `src/llmperf/web/services/task_service.py`
- 相关测试：`tests/test_task_execution_controls.py`

现状：

- `_update_progress_snapshot_lightweight()` 已经使用 `get_run_counts()` 做总体聚合。
- `_update_progress_snapshot()` 仍然会全量读取 records 并计算 executor 明细。
- `get_progress()` 在缓存失效时会调用完整 `_update_progress_snapshot()`。

建议：

- 新增 `Storage.get_executor_progress_summary(run_id)`，只查询必要字段并在 SQL 层聚合：
  - `executor_id`
  - `COUNT(*)`
  - `SUM(status = 200)`
  - `SUM(total_cost)`
  - `AVG(qtokens)`
  - `AVG(atokens)`
  - `MIN(created_at)`
  - `MAX(created_at)`
- `TaskService._update_progress_snapshot()` 改为使用 SQL 聚合结果构建 executor progress。
- 对运行中的 progress，不计算需要完整 `action_times` 分位数的指标，或先填充 `0`/最近快照值。
- 对 completed/failed/cancelled 任务，可以保留最终完整刷新并缓存，或也使用聚合结果。

收益：

- progress 从“读取并反序列化 N 条完整记录”变成“返回 executor 数量级的聚合行”。
- 任务列表批量进度接口会明显变快。

牺牲：

- 运行中 executor 级 `avg_ttft`、`p95_ttft`、`avg_total_time`、`token_per_second` 等复杂指标可能不再实时精确。

风险控制：

- UI 中运行中优先显示 completed/total/success/error/cost/status。
- 完整报告仍保留复杂指标。
- 保留 `tests/test_task_execution_controls.py::test_batch_progress_endpoint_returns_multiple_task_snapshots` 和 `test_get_progress_returns_detached_snapshot` 的行为。

### 3. report/stats 增加缓存和节流

调整范围：

- `src/llmperf/web/services/task_service.py`
- `src/llmperf/web/routers/tasks.py`

建议：

- `get_stats(run_id)`：
  - 已完成任务继续使用 `_completed_stats_cache`。
  - 运行中任务增加短 TTL 缓存，例如 30 到 60 秒。
  - 缓存 key 包含 `run_id` 和当前 execution 总数，避免长时间返回过旧数据。
- `get_quick_report(run_id)`：
  - 运行中使用 report TTL 缓存。
  - completed/failed/cancelled 后缓存最终报告。
  - 任务重跑、恢复、删除、改配置时调用 `_clear_task_caches(run_id)`。
- 路由注释和 Cache-Control 不再宣称“每次都重算”。

收益：

- 即使用户频繁刷新或多个浏览器打开同一任务，也不会重复做全量统计。

牺牲：

- 运行中 report 可能最多延迟 30 到 60 秒。

风险控制：

- 手动刷新可以支持 `force=true` 查询参数，必要时绕过缓存。
- 任务完成时强制刷新一次最终快照。

### 4. 总成本接口改为读 runs 聚合

调整范围：

- `src/llmperf/records/storage.py`
- `src/llmperf/web/routers/pricing.py`
- 可能新增一次性回填脚本或启动时懒回填

现状：

- `get_total_cost()` 同时读 `runs.total_cost` 和 `SUM(executions.total_cost)`，取最大值。

建议：

- 默认只执行：
  - `SELECT COALESCE(SUM(total_cost), 0), COUNT(*) FROM runs`
- 提供单独维护/修复入口用于历史数据回填：
  - 按 run_id 聚合 executions，更新 runs.total_cost。
  - 只手动执行，不放在页面请求链路里。
- 任务完成时已经会更新 run cost，保留该逻辑。

收益：

- 仪表盘、任务列表、成本页不再因为总成本统计扫描 executions 全表。

牺牲：

- 如果历史 run 的 `runs.total_cost` 不完整，总成本初期可能偏低，直到回填完成。

风险控制：

- 提供回填命令或脚本。
- 页面可标注“按任务汇总成本统计”。
- 回填前后做抽样校验。

### 5. 增加低风险索引

调整范围：

- `src/llmperf/records/db.py`

建议新增：

```sql
CREATE INDEX IF NOT EXISTS idx_exec_run_id_id
ON executions(run_id, id);

CREATE INDEX IF NOT EXISTS idx_exec_run_error_id
ON executions(run_id, id DESC)
WHERE status != 200;

CREATE INDEX IF NOT EXISTS idx_exec_status_created_provider_model
ON executions(status, created_at, provider, model);

CREATE INDEX IF NOT EXISTS idx_exec_provider_model_created
ON executions(provider, model, created_at);

CREATE INDEX IF NOT EXISTS idx_runs_created_at
ON runs(created_at DESC);
```

收益：

- 优化 run 内分页/顺序读取、错误列表、成本汇总和 provider/model 查询。

牺牲：

- 数据库文件会变大。
- 写入时需要维护更多索引，单条写入略慢。

风险控制：

- 第一阶段只加确实匹配现有查询的索引。
- 在大库上建索引前先备份数据库。
- 选择低峰期建索引。

### 6. SQLite 连接级 PRAGMA

调整范围：

- `src/llmperf/records/db.py`

建议：

- 使用 SQLAlchemy event 在每个 DBAPI connection 上执行：
  - `PRAGMA busy_timeout=30000`
  - `PRAGMA journal_mode=WAL`
  - `PRAGMA synchronous=NORMAL`
  - 可评估 `PRAGMA temp_store=MEMORY`
- `create_engine()` 增加 `connect_args={"timeout": 30}`。

收益：

- 锁等待行为更稳定。
- 避免只有初始化连接设置了 PRAGMA，后续 session 连接没有设置。

牺牲：

- 无明显产品牺牲。

风险控制：

- 保留现有 `_init_pragmas()`，但以 connect event 为准。
- 测试 Windows 和 Linux 下启动兼容性。

### 7. 写入批量化作为第一阶段后半段

调整范围：

- `src/llmperf/executors/base.py`
- `src/llmperf/records/storage.py`

建议分两步：

1. 低风险版：
   - 增加 `Storage.insert_records(records)`。
   - executor 内部积累小批量，例如 20 到 100 条或 1 秒 flush。
   - 任务结束时强制 flush。

2. 稳定版：
   - 设计单 writer 队列。
   - 多线程/多进程将记录发送给 writer，由 writer 批量写 SQLite。

第一阶段建议先做低风险版，单 writer 队列可作为后续独立阶段。

收益：

- 大幅减少 commit 次数和 SQLite 写锁争用。

牺牲：

- 进度可见性可能延迟 1 秒左右。
- 异常退出时可能丢失最后一小批未 flush 记录。

风险控制：

- 批量大小和 flush 间隔通过环境变量控制。
- 默认保守，例如 batch size 20、flush interval 1 秒。
- 每条记录仍然是完整 execution 行，不改变 schema。
- 任务取消、失败、完成都必须 flush。

## 建议实施顺序

### Milestone 1：立即降低页面压力

1. 前端停止运行中自动刷新 report。
2. report/stats 增加运行中 TTL 缓存。
3. 总成本接口改为只读 runs 聚合。

预期收益：

- 任务详情页和任务列表页超时概率明显下降。
- 不涉及数据库 schema 迁移。

### Milestone 2：降低监控和 progress 压力

1. 新增 executor 级 SQL 聚合。
2. progress 完整刷新不再全量 `fetch_run_records()`。
3. 调整监控线程：运行中只做轻量刷新；最终状态再计算完整快照。

预期收益：

- 5 到 7 个运行任务下，后台监控不再周期性扫大 run。

### Milestone 3：数据库访问稳定性

1. 增加连接级 PRAGMA。
2. 增加组合索引。
3. 增加历史总成本回填脚本。

预期收益：

- 查询更稳定，锁等待更可控。

### Milestone 4：写入批量化

1. 增加 `insert_records()`。
2. executor 小批量 flush。
3. 增加取消/异常/完成 flush 保证。

预期收益：

- 多进程/多线程写 SQLite 时吞吐和稳定性提升。

## 产品影响

保留能力：

- 创建任务、启动、暂停、恢复、取消、重跑。
- 任务列表和任务详情实时进度。
- 错误列表。
- 完整报告。
- CSV/JSONL/HTML 导出。
- 成本统计和价格管理。

变化：

- 运行中报告不再自动高频重算。
- 运行中 executor 复杂指标可能延迟更新或展示最近快照。
- 总成本依赖 `runs.total_cost`，历史数据需要回填后完全准确。
- 批量写入启用后，进度最多有 1 秒左右延迟。

不变：

- execution 原始记录仍写入同一张表。
- 导出仍可读取完整内容。
- 不需要迁移大字段。

## 验收标准

功能验收：

- 任务创建、启动、暂停、恢复、取消、重跑正常。
- 任务列表能显示状态、成本、进度。
- 任务详情能显示实时 progress 和错误列表。
- 完成任务后能生成完整 report。
- CSV/JSONL 导出内容不丢失。
- 价格管理和成本页可用。

性能验收：

- 5 到 7 个任务并发运行时：
  - `/api/tasks` P95 < 1 秒。
  - `/api/tasks/progress/batch` P95 < 1 秒。
  - `/api/tasks/{run_id}/progress` P95 < 1 秒。
  - `/api/tasks/{run_id}/errors` P95 < 2 秒。
  - 页面不出现 30 秒超时。
- 运行中不应出现周期性 CPU/DB 峰值，尤其不应每 10 到 20 秒出现明显卡顿。

数据库验收：

- `EXPLAIN QUERY PLAN` 确认关键查询使用新增索引。
- WAL 模式正常启用。
- 多进程写入下 `database is locked` 日志明显减少。

## 回归测试建议

后端：

```bash
pytest tests/test_task_execution_controls.py
pytest tests/test_web_api.py
pytest tests/test_pricing_linkage.py
pytest tests/test_analysis_history_export.py
```

前端：

```bash
cd frontend
npm run build
```

手工测试：

1. 创建 5 到 7 个 mock 或低成本任务并发运行。
2. 打开任务列表页观察进度刷新。
3. 打开一个运行中任务详情页，观察 progress/errors 刷新。
4. 手动点击“刷新报告”，确认仍可生成阶段性报告。
5. 等任务完成后刷新详情页，确认最终报告生成。
6. 导出 CSV/JSONL，确认内容完整。

## 回滚策略

前端改动：

- 恢复运行中 `loadReport()` 定时器即可回滚。

缓存改动：

- 通过环境变量关闭 report/stats TTL 缓存，或将 TTL 设置为 0。

总成本改动：

- 保留旧的全量 `SUM(executions.total_cost)` 作为可选修复接口，不放在页面默认路径。

索引改动：

- 索引新增通常可保留。
- 如写入下降明显，可按索引名逐个 drop。

批量写入：

- 通过环境变量将 batch size 设置为 1，恢复逐条提交行为。

## 观测指标

建议补充日志或简单指标：

- 每个 API 请求耗时，至少覆盖 `/api/tasks`、`/progress`、`/progress/batch`、`/report`、`/pricing/cost/total`。
- `fetch_run_records()` 调用次数和耗时。
- `insert_record()` / `insert_records()` commit 次数和耗时。
- SQLite locked retry 次数。
- report/stats 缓存命中率。

这些指标可以先用日志实现，不必第一阶段引入完整监控系统。

## 第一阶段推荐取舍

第一阶段优先牺牲“运行中完整报告自动实时精确刷新”，换取页面可用性和执行稳定性。

这个取舍是合理的：用户在任务运行中最需要的是进度、错误、成本和状态；完整报告和复杂分位数指标在任务完成后查看更有价值。通过保留手动刷新和完成后最终报告，产品能力不丢失，但系统压力会显著下降。
