// Extended report types for detailed analysis

export interface DetailedReport {
  run_id: string
  task_name: string
  task_type: 'benchmark' | 'monitoring'
  completed_at: string | null
  duration_seconds: number
  score: number
  grade: string
  dimension_scores: {
    latency: number
    throughput: number
    success_rate: number
    cost: number
  }
  metrics: {
    total_requests: number
    success_rate: number
    avg_ttft: number
    p50_ttft: number
    p90_ttft: number
    p95_ttft: number
    p99_ttft: number
    avg_tps: number
    total_cost: number
    currency: string
    total_input_tokens: number
    total_output_tokens: number
  }
  executor_summary: Array<{
    id: string
    requests: number
    success_rate: number
    avg_ttft: number
    p95_ttft: number
    avg_tps: number
    cost: number
    avg_output_tokens: number
  }>
  alerts: Array<{
    type: string
    severity: 'info' | 'warning' | 'error'
    message: string
    suggestion?: string
  }>
  recommendations: Array<{
    category: 'performance' | 'cost' | 'reliability'
    title: string
    description: string
    impact: 'high' | 'medium' | 'low'
  }>

  // Extended detailed data
  latency_analysis?: LatencyAnalysis
  token_analysis?: TokenAnalysis
  time_series?: TimeSeriesData
  time_frame_analysis?: TimeFrameAnalysis
  error_analysis?: ErrorAnalysis
  cost_analysis?: CostAnalysis
}

export interface LatencyAnalysis {
  values: number[]
  p50: number
  p90: number
  p95: number
  p99: number
  mean: number
  std: number
  cv: number
  by_executor?: Record<string, {
    values: number[]
    p50: number
    p95: number
    p99: number
  }>
}

export interface TokenAnalysis {
  output_tokens: number[]
  avg_tokens: number
  p50: number
  p90: number
  by_executor?: Record<string, {
    values: number[]
    avg: number
  }>
}

export interface TimeSeriesData {
  timeline: number[]
  ttft: number[]
  tps: number[]
  success_rate?: number[]
  cost_accumulated?: number[]
}

export interface TimeFrameAnalysis {
  time_frames: Array<{
    name: string
    start_hour: number
    end_hour: number
    metrics: {
      avg_ttft: number
      avg_total_time: number
      avg_tps: number
      success_rate: number
      request_count: number
    }
  }>
  insights: string[]
}

export interface ErrorAnalysis {
  total_errors: number
  error_rate: number
  by_type: Record<string, {
    count: number
    rate: number
    examples: string[]
  }>
  by_time?: Array<{
    time: number
    error_count: number
    error_rate: number
  }>
}

export interface CostAnalysis {
  total_cost: number
  currency: string
  by_executor: Array<{
    executor: string
    cost: number
    input_tokens: number
    output_tokens: number
    request_count: number
    avg_cost_per_request: number
  }>
  cost_trend?: Array<{
    time: number
    accumulated_cost: number
  }>
}
