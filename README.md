# LLMPerf：大模型接口性能评测工具（YAML 驱动）

LLMPerf 是一个以 **YAML 配置** 驱动的 LLM 性能评测工具，支持：

- 数据集：`DatasetSource`（JSONL / 生成器等）+ `DatasetIterator`（变异链 + round/case index 标记；终止边界由 Executor 裁决）
- 执行：多进程（按 executor 独立进程）+ 每个 executor 内并发
- 记录：SQLite 持久化（每条调用记录包含请求参数、计费、时序片段等）
- 产物：执行完成自动导出 Excel（汇总 + 明细分 sheet），也可通过分析命令重放导出

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

## 分析示例

分析命令统一为：

```bash
llmperf-analyze --type <分析类型> --config <所需的配置文件>
```

目前内置类型：

- [x]`summary`：基于指定 `run_id` 重放导出 Excel

### summary 示例

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
