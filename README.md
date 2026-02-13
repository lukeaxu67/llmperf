# LLMPerf：大模型接口性能评测工具（YAML 驱动）

LLMPerf 是一个以 **YAML 配置** 驱动的 LLM 性能评测工具，支持：

- **数据集**：`DatasetSource`（JSONL / 生成器等）+ `DatasetIterator`（变异链 + round/case index 标记；终止边界由 Executor 裁决）
- **执行**：多进程（按 executor 独立进程）+ 每个 executor 内并发
- **记录**：SQLite 持久化（每条调用记录包含请求参数、计费、时序片段等）
- **产物**：执行完成自动导出 Excel（汇总 + 明细分 sheet），也可通过分析命令重放导出
- **持续监控**：支持 24/7 长期监控，通过 `interval_seconds` + `max_total_seconds` 配置实现
- **深度分析**：稳定性波动性分析、服务健康度分析、资源行为推断、限流检测、综合报告

---

## 目录结构

```text
template/            # 示例任务配置
resource/            # 示例数据集
pricings/            # 示例价格表
src/                 # 源码
```

---

## 安装

要求：Python >= 3.12

推荐使用虚拟环境后安装本项目（`src/` 布局）：

```bash
pip install -e .
```

Excel 导出依赖：`numpy`、`pandas`、`openpyxl`（已在 `pyproject.toml` 依赖中声明）。

---

## 默认数据库位置

默认 SQLite 路径为：`~/llmperf/db/perf.sqlite`，如需覆盖，在任务 YAML 中设置 `db_path`。

---

## 任务配置

参考 `template/v1.yaml`。核心字段：

- `info`：任务名称（也用于 Excel 文件命名）
- `db_path`：可选，覆盖默认 DB 路径
- `dataset.source`：数据集来源（例如 JSONL）
- `dataset.iterator`：iterator 策略（mutation_chain + 可选终止边界；终止判定由 Executor 执行）
- `executors`：评测对象列表（同一任务内可以同 `provider+model` 但不同 `param`）
- `pricing`：可留空，推荐走命令行 `--pricing-file` 注入

一个最小配置示例（JSONL + mock executor）：

```yaml
info: "默认示例任务"

dataset:
  source:
    type: "jsonl"
    name: "demo"
    config:
      path: "resource/demo.jsonl"
      encoding: "utf-8"
  iterator:
    mutation_chain: ["identity"]
    max_rounds: 1

executors:
  - id: "debug-001"
    name: "Mock链路"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-001"
    rate:
      interval_seconds: 10

  - id: "debug-002"
    name: "Mock链路"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-001"
    param:
      temperature: 0.7
      extra_body:
        enable_thining: true

  - id: "debug-003"
    name: "Mock链路"
    after: ["debug-001", "debug-002"]
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-001"
    param:
      max_tokens: 500
```

---

## 价格表

价格表建议放在 repo 根目录同级的 `pricings/` 下，便于多个任务复用、随时维护刷新。

格式示例（见 `pricings/20251227.yaml`）：

```yaml
pricing:
  - provider: openai
    model: gpt-4o
    unit: per_1m
    input_price: 15.00
    output_price: 60.00
    cache_input_discount: 0.20
    cache_output_discount: 0.00
    currency: USD
```

---

## 执行示例

执行命令（推荐显式指定价格表文件）：

```bash
llmperf-run -c template/0.1.0.yaml --pricing-file pricings/20251227.yaml
```

执行完成后将：

1) 把本次任务的 `任务配置`/`价格表` 内容快照写入 `runs` 表
2) 把每条调用写入 `executions` 表（包含 `provider/model/request_params` 等）
3) 自动在当前目录生成 Excel：`<任务名称>-<执行时间>.xlsx`

---

## 持续监控

通过配置 `interval_seconds`（请求间隔）+ 足够大的数据集（`count`）即可实现长期监控：

### 配置说明

- `dataset.source.config.generator.parameters.count`：生成足够的测试数据（如 60 条 = 60 次请求）
- `executor.rate.interval_seconds`：每次请求之间的间隔（秒）
- 数据集会循环使用，配合 `interval_seconds` 实现持续监控

### 24小时监控配置示例

```yaml
info: "24小时持续监控 - OpenAI GPT-4o"

dataset:
  source:
    type: "generator"
    name: "simple_questions"
    config:
      generator:
        class_name: "simple_questions"
        parameters:
          count: 60  # 生成60个问题，配合 interval_seconds 实现监控
  iterator:
    mutation_chain: ["identity"]

executors:
  - id: "openai-gpt4o-monitor"
    name: "OpenAI GPT-4o 监控"
    type: "openai"
    impl: "chat"
    concurrency: 1
    model: "gpt-4o"
    param:
      max_tokens: 400
    rate:
      interval_seconds: 60  # 每60秒执行一次

pricing:
  - provider: openai
    model: gpt-4o
    unit: per_1m
    input_price: 2.50
    output_price: 10.00
    currency: USD
```

### 启动监控

```bash
llmperf-run -c template/monitoring.yaml
```

执行后将：
1. 按配置的间隔持续发起请求
2. 所有记录自动保存到 SQLite 数据库
3. 执行完成后自动生成 Excel 报告
4. 可使用分析命令对收集的数据进行深度分析

---

## 分析示例

分析命令统一为：

```bash
llmperf-analyze --type <分析类型> --config <配置文件>
```

### 基础分析类型

- **`summary`**：基于指定 `run_id` 重放导出 Excel
- **`history`**：查看历史任务（按 run 展开 provider/model/params 的条数、成功/失败与成本，并给出时间范围）
- **`export`**：导出 run 的执行结果为 JSONL

#### summary 示例

新建 `summary.yaml`：

```yaml
task_name: "任务导出"
run_id: "<RUN_ID>"
```

运行：

```bash
llmperf-analyze --type summary --config summary.yaml
```

会输出 `output_path`（生成的 xlsx 文件路径）。

---

### 深度分析类型

#### stability - 稳定性与波动性分析

分析延迟分布、昼夜差异、滑动窗口稳定性、异常值检测：

```yaml
type: stability
query:
  db_path: "~/llmperf/db/perf.sqlite"
  provider: "openai"      # 可选：筛选特定提供商
  model: "gpt-4o"         # 可选：筛选特定模型
  # run_ids:             # 可选：指定 run_id 列表
  start_ts: 1704067200000  # 可选：开始时间戳（毫秒）
  end_ts: 1704153600000    # 可选：结束时间戳（毫秒）
segment_size: 6          # 时间分段大小（小时），默认 6
window_minutes: 30        # 滑动窗口大小（分钟），默认 30
outlier_threshold: 2.0  # 异常值阈值（P95 的倍数）
z_score_threshold: 2.0  # 异常检测 Z 分数阈值
```

**输出指标**：
- TTFT 分布（P50/P90/P95/P99/均值/标准差/CV）
- 异常值比例（> 2x P95）
- 时间分段统计（0-6/6-12/12-18/18-24 小时）
- 日夜对比分析
- 滑动窗口稳定性
- 输出长度分布

#### health - 服务健康度分析

分析错误率、错误类型、截断模式、输出长度分布：

```yaml
type: health
query:
  db_path: "~/llmperf/db/perf.sqlite"
  provider: "openai"
  model: "gpt-4o"
```

**输出指标**：
- 总体错误率
- 错误分类（timeout/5xx/4xx/网络/其他）
- 截断检测（接近 max_tokens 的响应）
- 输出长度统计
- 时间段错误率分析

#### resource - 资源行为分析

分析 TTFT 与生成速率相关性、请求间自相关：

```yaml
type: resource
query:
  db_path: "~/llmperf/db/perf.sqlite"
  provider: "openai"
  model: "gpt-4o"
```

**输出指标**：
- TTFT vs 吞吐量相关系数
- TTFT vs 总耗时相关系数
- TTFT 自相关（lag 1/5/10/30）
- 吞吐量自相关
- 高 TTFT 分析（排队 vs 资源竞争）

#### ratelimit - 限流检测

检测分钟级限流、固定节奏波动、周期性模式：

```yaml
type: ratelimit
query:
  db_path: "~/llmperf/db/perf.sqlite"
  provider: "openai"
  model: "gpt-4o"
```

**输出指标**：
- 分钟级模式分析
- 固定间隔慢速检测
- 周期性检测（1/5/10/15/60 分钟）
- 请求计数限流检测

#### monitoring_report - 综合监控报告

生成包含所有分析维度的 HTML 报告：

```yaml
type: monitoring_report
query:
  db_path: "~/llmperf/db/perf.sqlite"
  provider: "openai"
  model: "gpt-4o"
output_dir: "./reports"      # 报告输出目录
report_name: "24h_monitoring" # 报告名称
include_html: true          # 是否生成 HTML 报告
include_excel: true         # 是否生成 Excel 报告
```

**输出内容**：
1. 服务稳定性报告
   - 延迟分布
   - 波动指标
   - 错误率趋势

2. 调度行为报告
   - 自相关分析
   - 日夜差异
   - 批次震荡模式

3. 模型输出稳定性
   - 输出长度趋势
   - 拒答比例

4. 执行摘要
   - 关键洞察
   - 优化建议

---

## 测试示例

### Mock 测试（每秒执行一次，执行1分钟）

配置文件：`template/test_mock.yaml`

```bash
llmperf-run -c template/test_mock.yaml
```

预期结果：
- 60 条请求（60 秒 ÷ 1 秒间隔）
- 100% 成功率
- 平均 TTFT 约 10ms（mock 数据）
- 吞吐量约 3000 tok/s

分析结果：

```bash
# 创建分析配置
cat > analyze_test.yaml << EOF
type: stability
query:
  run_ids:
    - "<RUN_ID>"  # 替换为实际的 run_id
segment_size: 6
window_minutes: 5
EOF

# 运行分析
llmperf-analyze --type stability --config stability.yaml
```

---

## 常见问题

### Q: 如何实现24小时持续监控？

A: 配置足够的数据量（如 `count: 1440`）+ 请求间隔（`interval_seconds: 60`），工具会自动按间隔执行所有请求。

### Q: 数据存储在哪里？

A: 默认存储在 `~/llmperf/db/perf.sqlite`，可通过配置中的 `db_path` 字段覆盖。

### Q: 如何分析历史数据？

A: 使用分析命令，通过 `start_ts` 和 `end_ts` 指定时间范围，或使用 `run_ids` 指定特定的 run。

### Q: 支持哪些 LLM 提供商？

A: OpenAI、通义千问、智谱、DeepSeek、讯飞星火、腾讯混元、火山方舟、Moonshot、Mock 等。

---

## 开发

项目结构：

```
src/llmperf/
├── analysis/          # 分析模块
│   ├── statistics.py           # 统计分析工具
│   ├── timeseries.py          # 时序分析工具
│   └── impl/                 # 分析实现
│       ├── stability.py        # 稳定性分析
│       ├── health.py           # 健康度分析
│       ├── resource.py         # 资源行为分析
│       ├── ratelimit.py        # 限流检测
│       └── monitoring_report.py # 综合报告
├── cli/              # 命令行工具
├── config/           # 配置加载
├── datasets/         # 数据集
│   └── generators/
│       └── simple_questions_generator.py  # 简单问题生成器
├── executors/       # 执行器
├── providers/        # 提供商实现
│   └── mock.py               # Mock 提供商（用于测试）
├── records/          # 数据记录
└── runner.py         # 运行入口
```

---

## License

MIT
