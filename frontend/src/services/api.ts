import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    let message = '请求失败'

    if (error.response?.data?.detail) {
      const detail = error.response.data.detail
      // Handle FastAPI validation errors (array of error objects)
      if (Array.isArray(detail)) {
        message = detail.map((e: any) => e.msg || JSON.stringify(e)).join('; ')
      } else if (typeof detail === 'string') {
        message = detail
      } else {
        message = JSON.stringify(detail)
      }
    } else if (error.message) {
      message = error.message
    }

    return Promise.reject(new Error(message))
  }
)

export default api

// Types
export interface Task {
  run_id: string
  status: 'scheduled' | 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
  config_path?: string
  task_name: string
  task_type?: 'benchmark' | 'monitoring'
  created_at: string
  scheduled_at?: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface ExecutorProgress {
  id: string
  name: string
  provider?: string
  model?: string | null
  after: string[]
  order: number
  status: string
  completed: number
  total: number
  progress_percent: number
  success_count: number
  error_count: number
  success_rate: number
  avg_input_tokens: number
  avg_output_tokens: number
  avg_ttft: number
  p95_ttft: number
  avg_total_time: number
  avg_token_per_second: number
  avg_token_per_second_with_calltime: number
  cost: number
  avg_cost_per_request: number
  score: number
  conclusion: string
}

export interface TaskTopologyNode {
  id: string
  name: string
  kind?: 'boundary' | 'executor'
  status?: string
  level?: number
  progress_percent?: number
  model?: string | null
  provider?: string
}

export interface TaskTopology {
  nodes: TaskTopologyNode[]
  edges: Array<{ source: string; target: string }>
  layers: Array<{ level: number; node_ids: string[] }>
}

export interface TaskProgress {
  run_id: string
  status: string
  progress_percent: number
  completed: number
  total: number
  elapsed_seconds: number
  eta_seconds?: number
  success_count: number
  error_count: number
  current_cost: number
  currency: string
  current_rate?: number
  concurrency?: number
  paused_at?: string | null
  dataset_total_per_executor?: number
  executors?: ExecutorProgress[]
  topology?: TaskTopology
}

export interface TaskStats {
  run_id: string
  total_requests: number
  success_count: number
  error_count: number
  success_rate: number
  total_cost: number
  currency: string
  avg_first_resp_time: number
  p50_first_resp_time: number
  p95_first_resp_time: number
  p99_first_resp_time: number
  avg_last_resp_time: number
  p95_last_resp_time: number
  avg_char_per_second: number
  avg_token_throughput: number
  avg_token_per_second: number
  avg_token_per_second_with_calltime: number
  avg_input_tokens: number
  avg_output_tokens: number
  total_input_tokens: number
  total_output_tokens: number
}

export interface Dataset {
  id: string
  name: string
  description?: string
  file_type: string
  file_path: string
  row_count?: number
  record_count?: number
  file_size?: number
  size?: number
  columns?: string[]
  created_at?: number
  updated_at?: number
  format?: string
  encoding?: string
}

export interface ConfigTemplate {
  name: string
  path: string
  description: string
}

export interface AnalysisSummary {
  run_id: string
  total_requests: number
  success_count: number
  error_count: number
  success_rate: number
  total_cost: number
  currency: string
  avg_first_resp_time: number
  p50_first_resp_time: number
  p95_first_resp_time: number
  p99_first_resp_time: number
  avg_char_per_second: number
  avg_token_throughput: number
  by_executor: Record<string, any>
}

// Pricing types
export interface PricingRecord {
  id: number
  provider: string
  model: string
  input_price: number
  output_price: number
  cache_read_price: number
  cache_write_price: number
  effective_at: number
  created_at: number
  note: string
}

export interface CostSummary {
  provider: string
  model: string
  request_count: number
  total_input_tokens: number
  total_output_tokens: number
  total_cost: number
}

export interface TotalCost {
  total_cost: number
  run_count: number
  currency: string
}

// API Functions
export const taskApi = {
  list: (params?: { status?: string; limit?: number; offset?: number }) =>
    api.get<{ tasks: Task[]; total: number }>('/tasks', { params }),

  get: (runId: string) =>
    api.get<Task>(`/tasks/${runId}`),

  create: (data: { config_path?: string; config_content?: string; run_id?: string; auto_start?: boolean; task_type?: 'benchmark' | 'monitoring'; scheduled_at?: string }) =>
    api.post<Task>('/tasks', data),

  getProgress: (runId: string) =>
    api.get<TaskProgress>(`/tasks/${runId}/progress`),

  getStats: (runId: string) =>
    api.get<TaskStats>(`/tasks/${runId}/stats`),

  getReport: (runId: string) =>
    api.get(`/tasks/${runId}/report`),

  cancel: (runId: string) =>
    api.post(`/tasks/${runId}/cancel`),

  start: (runId: string) =>
    api.post(`/tasks/${runId}/start`),

  pause: (runId: string) =>
    api.post(`/tasks/${runId}/pause`),

  resume: (runId: string) =>
    api.post(`/tasks/${runId}/resume`),

  stop: (runId: string) =>
    api.post(`/tasks/${runId}/stop`),

  retry: (runId: string) =>
    api.post(`/tasks/${runId}/retry`),

  rerun: (runId: string, data?: { auto_start?: boolean; scheduled_at?: string }) =>
    api.post(`/tasks/${runId}/rerun`, data),

  getConfig: (runId: string) =>
    api.get<{ run_id: string; config_content: string }>(`/tasks/${runId}/config`),

  delete: (runId: string) =>
    api.delete(`/tasks/${runId}`),

  export: (runId: string, format: 'jsonl' | 'csv' | 'html') =>
    api.post(`/tasks/${runId}/export`, { format }),

  exportResults: (runId: string, format: 'jsonl' | 'csv') =>
    api.get(`/tasks/${runId}/export`, { params: { format }, responseType: 'blob' }),
}

export const pricingApi = {
  list: (params?: { provider?: string; model?: string; limit?: number }) =>
    api.get<{ items: PricingRecord[]; total: number }>('/pricing', { params }),

  add: (data: {
    provider: string
    model: string
    input_price: number
    output_price: number
    cache_read_price?: number
    cache_write_price?: number
    effective_at?: number
    note?: string
  }) => api.post<PricingRecord>('/pricing', data),

  get: (id: number) =>
    api.get<PricingRecord>(`/pricing/${id}`),

  delete: (id: number) =>
    api.delete(`/pricing/${id}`),

  getProviders: () =>
    api.get('/pricing/providers'),

  getHistory: (params?: { provider?: string; days?: number }) =>
    api.get('/pricing/history/chart', { params }),

  getCostSummary: (days?: number) =>
    api.get<CostSummary[]>('/pricing/cost/summary', { params: { days } }),

  getTotalCost: () =>
    api.get<TotalCost>('/pricing/cost/total'),

  getCurrentPrice: (provider: string, model: string) =>
    api.get(`/pricing/current`, { params: { provider, model } }),
}

export const datasetApi = {
  list: () =>
    api.get<{ datasets: Dataset[]; total: number }>('/datasets'),

  get: (datasetId: string) =>
    api.get<Dataset>(`/datasets/${datasetId}`),

  preview: (datasetId: string, limit: number = 10) =>
    api.get(`/datasets/${datasetId}/preview`, { params: { limit } }),

  validate: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/datasets/validate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  upload: (file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
  },

  delete: (datasetId: string) =>
    api.delete(`/datasets/${datasetId}`),
}

export const configApi = {
  listTemplates: () =>
    api.get<ConfigTemplate[]>('/config/templates'),

  getTemplate: (name: string) =>
    api.get(`/config/templates/${name}`),

  validate: (configContent: string) =>
    api.post('/config/validate', configContent, {
      headers: { 'Content-Type': 'text/plain' },
    }),

  getRuntime: () =>
    api.get('/config/runtime'),

  getPricing: () =>
    api.get('/config/pricing'),
}

// Test run types
export interface TestRunRequest {
  config_content: string
}

export interface TestRunResponse {
  success: boolean
  duration_ms: number
  first_token_ms: number
  tokens_per_second: number
  response: string
  error: string
  results?: Array<{
    executor_id: string
    executor_name: string
    provider: string
    model: string
    success: boolean
    duration_ms: number
    first_token_ms: number
    tokens_per_second: number
    status_code?: number
    error_type?: string
    error_detail?: Record<string, any>
    response: string
    error: string
  }>
}

// Extend taskApi with testRun
export const testRunApi = {
  run: (data: TestRunRequest) =>
    api.post<TestRunResponse>('/tasks/test-run', data, { timeout: 300000 }),
}

// Extended report types
export interface DetailedReport {
  run_id: string
  task_name: string
  task_type?: 'benchmark' | 'monitoring'
  status: string
  is_partial: boolean
  completed_at: string | null
  generated_at?: string | null
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
    success_count?: number
    error_count?: number
    success_rate: number
    avg_ttft: number
    p50_ttft: number
    p95_ttft: number
    avg_total_time?: number
    p95_total_time?: number
    avg_tps: number
    avg_tps_with_ttft?: number
    avg_input_tokens?: number
    avg_output_tokens?: number
    total_cost: number
    currency: string
    total_input_tokens: number
    total_output_tokens: number
  }
  executor_summary: ExecutorProgress[]
  topology?: TaskTopology
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
  latency_analysis?: {
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
      p50?: number
      p90?: number
      p95?: number
      p99?: number
      mean?: number
      std?: number
      cv?: number
    }>
  }
  token_analysis?: {
    output_tokens: number[]
    avg_tokens: number
    p50: number
    p90: number
    by_executor?: Record<string, {
      values: number[]
      avg: number
      p50?: number
      p90?: number
    }>
  }
  time_series?: {
    timeline: number[]
    ttft: number[]
    tps: number[]
    success_rate?: number[]
    cost_accumulated?: number[]
  }
  time_frame_analysis?: {
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
  error_analysis?: {
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
  cost_analysis?: {
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
}

// Extended report API
export const reportApi = {
  getDetailed: (runId: string) =>
    api.get<DetailedReport>(`/tasks/${runId}/detailed-report`),

  getLatencyAnalysis: (runId: string) =>
    api.get<DetailedReport['latency_analysis']>(`/tasks/${runId}/latency-analysis`),

  getTokenAnalysis: (runId: string) =>
    api.get<DetailedReport['token_analysis']>(`/tasks/${runId}/token-analysis`),

  getTimeSeries: (runId: string) =>
    api.get<DetailedReport['time_series']>(`/tasks/${runId}/time-series`),

  getTimeFrameAnalysis: (runId: string) =>
    api.get<DetailedReport['time_frame_analysis']>(`/tasks/${runId}/time-frame-analysis`),

  getErrorAnalysis: (runId: string) =>
    api.get<DetailedReport['error_analysis']>(`/tasks/${runId}/error-analysis`),

  getCostAnalysis: (runId: string) =>
    api.get<DetailedReport['cost_analysis']>(`/tasks/${runId}/cost-analysis`),
}
