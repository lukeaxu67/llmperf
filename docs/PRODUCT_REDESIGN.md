# LLMPerf 产品重构方案

## 一、业务分析与指标体系

### 1.1 核心业务场景

| 场景 | 描述 | 关键指标 |
|------|------|----------|
| 基准测试 | 对比不同模型性能 | TTFT, TPS, 成本 |
| 压力测试 | 测试系统极限 | QPS, 成功率, 延迟 |
| 回归测试 | 监控版本变化 | 历史对比, 变化率 |
| 成本分析 | 优化API成本 | 单次成本, 缓存率 |
| 稳定性监控 | 长期运行监控 | 错误率, 超时率 |

### 1.2 LLM性能指标体系

#### 延迟指标 (Latency)
```
TTFT (Time To First Token)     - 首Token延迟，用户感知关键指标
TPOT (Time Per Output Token)   - 每Token生成时间
Total Latency                  - 总响应时间
Inter-Token Latency            - Token间延迟，反映生成流畅度
```

#### 吞吐指标 (Throughput)
```
TPS (Tokens Per Second)        - 每秒生成Token数
CPS (Chars Per Second)         - 每秒生成字符数
RPS (Requests Per Second)      - 每秒请求数
```

#### 质量指标 (Quality)
```
Success Rate                   - 成功率
Error Rate                     - 错误率
Timeout Rate                   - 超时率
Truncation Rate                - 截断率
Complete Response Rate         - 完整响应率
```

#### 成本指标 (Cost)
```
Cost Per Request               - 单次请求成本
Cost Per 1K Tokens             - 每千Token成本
Cache Hit Rate                 - 缓存命中率
Cache Savings                  - 缓存节省费用
```

#### 稳定性指标 (Stability)
```
Latency Variance               - 延迟抖动
Error Distribution             - 错误分布
Cold Start Latency             - 冷启动延迟
Sustained Performance         - 持续性能
```

### 1.3 分析报告类型

#### 快速报告 (Quick Report)
任务完成自动生成，1秒内呈现。

**内容:**
- 整体评分 (S/A/B/C/D/F)
- 核心指标卡片
- 异常提示
- 优化建议

#### 详细报告 (Detailed Report)

**内容:**
- 执行概览
- 性能详情
  - 延迟分布图
  - 吞吐趋势图
  - 错误分布
- 执行器对比
- Token使用分析
- 成本分析
- 异常检测
- 优化建议

#### 对比报告 (Comparison Report)

**内容:**
- 多任务横向对比
- 历史趋势对比
- 配置差异分析

## 二、用户体验设计

### 2.1 核心理念

**一站式体验**: 创建 = 执行 = 分析 = 报告

用户只需:
1. 选择/配置任务
2. 点击运行
3. 等待完成
4. 直接看报告

### 2.2 页面结构

```
首页 (Dashboard)
├── 快速创建任务入口
├── 最近任务卡片 (带评分)
├── 系统状态概览
└── 快捷操作

任务列表 (Tasks)
├── 任务卡片列表
│   ├── 状态/进度
│   ├── 评分预览
│   └── 快速操作
└── 筛选/搜索

创建任务 (New Task)
├── 模板选择
├── YAML编辑
├── 配置验证
└── 一键运行

任务详情 (Task Detail) ⭐ 重新设计
├── 执行状态
├── 实时进度
├── 快速报告 (自动生成)
│   ├── 评分卡片
│   ├── 核心指标
│   ├── 图表
│   └── 建议
├── 详细分析 (Tab切换)
│   ├── 概览
│   ├── 延迟分析
│   ├── 吞吐分析
│   ├── 成本分析
│   ├── 错误分析
│   └── 原始数据
└── 导出选项

对比分析 (Compare)
├── 选择任务
├── 横向对比
└── 历史趋势

设置 (Settings)
├── API密钥
├── 通知配置
└── 模板管理
```

### 2.3 关键交互

#### 任务完成后的自动报告

```
┌────────────────────────────────────────────────────────────┐
│  🎉 任务完成                                               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │  A  │  │TTFT │  │ TPS │  │ 成本 │                       │
│  │ 85分│  │235ms│  │ 42  │  │¥2.31│                       │
│  └─────┘  └─────┘  └─────┘  └─────┘                       │
│                                                            │
│  📊 性能概览                                               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  [延迟分布图]           [吞吐趋势图]                  │ │
│  │                                                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ⚠️ 发现2个问题:                                          │
│  • P95延迟较高 (1.2s)，建议优化prompt                     │
│  • 错误率5%高于基准，主要错误: 429 Rate Limit              │
│                                                            │
│  💡 优化建议:                                              │
│  • 降低并发数从10到5可减少429错误                          │
│  • 启用prompt缓存可节省约30%成本                           │
│                                                            │
│  [查看详细报告] [导出PDF] [分享]                           │
└────────────────────────────────────────────────────────────┘
```

## 三、数据库设计

### 3.1 表结构

```sql
-- 测试运行表
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    status TEXT,  -- pending/running/completed/failed/cancelled
    config_path TEXT,
    config_content TEXT,
    created_at INTEGER,
    started_at INTEGER,
    completed_at INTEGER,

    -- 聚合指标 (自动更新)
    total_requests INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0,
    avg_ttft_ms REAL,
    p50_ttft_ms REAL,
    p95_ttft_ms REAL,
    p99_ttft_ms REAL,
    avg_tps REAL,
    total_tokens INTEGER DEFAULT 0,

    -- 报告相关
    report_generated INTEGER DEFAULT 0,
    report_data TEXT,  -- JSON
    score INTEGER,     -- 0-100
    grade TEXT,        -- S/A/B/C/D/F

    -- 元数据
    tags TEXT,         -- JSON array
    metadata TEXT      -- JSON
);

-- 执行记录表 (每次API调用)
CREATE TABLE executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    executor_id TEXT,
    dataset_row_id TEXT,

    -- 请求信息
    provider TEXT,
    model TEXT,
    request_params TEXT,  -- JSON

    -- 响应信息
    status INTEGER,
    error_code TEXT,
    error_message TEXT,

    -- Token统计
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,

    -- 时间统计
    ttft_ms INTEGER,      -- 首Token延迟
    total_ms INTEGER,     -- 总时间
    generation_ms INTEGER,

    -- 成本
    input_cost REAL,
    output_cost REAL,
    cache_cost REAL,
    total_cost REAL,
    currency TEXT,

    -- 内容
    reasoning_tokens INTEGER,
    content_tokens INTEGER,
    reasoning_chars INTEGER,
    content_chars INTEGER,

    -- 性能
    tps REAL,             -- tokens/second
    cps REAL,             -- chars/second

    created_at INTEGER,

    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- 聚合指标表 (按run + executor)
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    executor_id TEXT,
    metric_type TEXT,     -- latency/throughput/cost/error

    -- 统计值
    count INTEGER,
    sum REAL,
    mean REAL,
    std_dev REAL,
    min REAL,
    max REAL,
    p50 REAL,
    p90 REAL,
    p95 REAL,
    p99 REAL,

    -- 时间窗口
    window_start INTEGER,
    window_end INTEGER,

    created_at INTEGER,

    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- 异常记录表
CREATE TABLE anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    executor_id TEXT,
    execution_id INTEGER,

    anomaly_type TEXT,    -- latency_spike/error_burst/cost_anomaly
    severity TEXT,        -- low/medium/high/critical
    description TEXT,

    -- 异常指标
    metric_name TEXT,
    expected_value REAL,
    actual_value REAL,
    deviation REAL,       -- 偏离程度

    -- 上下文
    context TEXT,         -- JSON

    created_at INTEGER,

    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- 报告模板表
CREATE TABLE report_templates (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    template_type TEXT,   -- quick/detailed/comparison
    sections TEXT,        -- JSON: 报告包含的section
    is_default INTEGER,
    created_at INTEGER
);

-- 历史对比表
CREATE TABLE comparisons (
    id TEXT PRIMARY KEY,
    name TEXT,
    run_ids TEXT,         -- JSON array
    comparison_type TEXT, -- model/config/time
    result TEXT,          -- JSON
    created_at INTEGER
);
```

### 3.2 索引

```sql
CREATE INDEX idx_executions_run_id ON executions(run_id);
CREATE INDEX idx_executions_executor ON executions(run_id, executor_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_metrics_run_type ON metrics(run_id, metric_type);
CREATE INDEX idx_anomalies_run ON anomalies(run_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at DESC);
```

## 四、评分系统

### 4.1 评分维度

```
总分 = TTFT分 × 0.25 + 吞吐分 × 0.25 + 成功率分 × 0.25 + 成本分 × 0.25
```

### 4.2 各维度评分标准

**TTFT评分 (越低越好)**
```
S: < 200ms    → 100分
A: 200-500ms  → 90分
B: 500-1000ms → 75分
C: 1-2s       → 60分
D: 2-5s       → 40分
F: > 5s       → 20分
```

**吞吐评分 (越高越好)**
```
S: > 100 tps  → 100分
A: 50-100 tps → 90分
B: 20-50 tps  → 75分
C: 10-20 tps  → 60分
D: 5-10 tps   → 40分
F: < 5 tps    → 20分
```

**成功率评分**
```
S: 99.9%+     → 100分
A: 99-99.9%   → 95分
B: 95-99%     → 80分
C: 90-95%     → 60分
D: 80-90%     → 40分
F: < 80%      → 20分
```

**成本效率评分 (相对基准)**
```
S: 节省30%+   → 100分
A: 节省10-30% → 90分
B: 基准水平   → 75分
C: 高于基准20% → 60分
D: 高于基准50% → 40分
F: 高于基准100%→ 20分
```

### 4.3 等级映射

```
90-100: S (卓越)
80-89:  A (优秀)
70-79:  B (良好)
60-69:  C (一般)
50-59:  D (较差)
0-49:   F (失败)
```

## 五、报告生成

### 5.1 快速报告结构

```typescript
interface QuickReport {
  // 基本信息
  runId: string
  taskName: string
  completedAt: Date
  duration: number

  // 评分
  score: number        // 0-100
  grade: 'S' | 'A' | 'B' | 'C' | 'D' | 'F'
  dimensionScores: {
    latency: number
    throughput: number
    successRate: number
    cost: number
  }

  // 核心指标
  metrics: {
    totalRequests: number
    successRate: number
    avgTTFT: number
    p95TTFT: number
    avgTPS: number
    totalCost: number
    cacheHitRate: number
  }

  // 执行器对比
  executorSummary: Array<{
    id: string
    requests: number
    successRate: number
    avgTTFT: number
    avgTPS: number
    cost: number
  }>

  // 异常提示
  alerts: Array<{
    type: string
    severity: 'info' | 'warning' | 'error'
    message: string
    suggestion?: string
  }>

  // 优化建议
  recommendations: Array<{
    category: 'performance' | 'cost' | 'reliability'
    title: string
    description: string
    impact: 'high' | 'medium' | 'low'
  }>
}
```

### 5.2 详细报告结构

```typescript
interface DetailedReport extends QuickReport {
  // 延迟分析
  latencyAnalysis: {
    distribution: HistogramData
    percentiles: { p50, p90, p95, p99 }
    timeseries: TimeSeriesData
    byExecutor: Map<string, LatencyStats>
    anomalies: Anomaly[]
  }

  // 吞吐分析
  throughputAnalysis: {
    distribution: HistogramData
    timeseries: TimeSeriesData
    byExecutor: Map<string, ThroughputStats>
    tokenBreakdown: {
      input: number
      output: number
      cached: number
    }
  }

  // 成本分析
  costAnalysis: {
    total: number
    breakdown: {
      input: number
      output: number
      cache: number
    }
    byExecutor: Map<string, CostStats>
    timeseries: TimeSeriesData
    savings: {
      cached: number
      potentialSavings: number
    }
  }

  // 错误分析
  errorAnalysis: {
    total: number
    rate: number
    byType: Map<string, number>
    byExecutor: Map<string, ErrorStats>
    timeseries: TimeSeriesData
    recentErrors: ErrorRecord[]
  }

  // 质量分析
  qualityAnalysis: {
    truncationRate: number
    incompleteRate: number
    avgResponseLength: number
    responseLengthDistribution: HistogramData
  }
}
```

## 六、实施计划

### Phase 1: 数据库重构 (1天)
- 新建表结构
- 数据迁移
- 更新ORM

### Phase 2: 后端API重构 (2天)
- 任务执行完成自动生成报告
- 新的分析API
- 报告API

### Phase 3: 前端重构 (3天)
- 任务详情页重构
- 报告组件
- 图表可视化

### Phase 4: 测试与优化 (1天)
- 单元测试
- 集成测试
- 性能优化
