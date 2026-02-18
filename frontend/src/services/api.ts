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
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  config_path?: string
  task_name: string
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
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
  avg_char_per_second: number
  avg_token_throughput: number
}

export interface Dataset {
  name: string
  path: string
  size: number
  record_count: number
  format: string
  encoding: string
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

  create: (data: { config_path?: string; config_content?: string; run_id?: string; auto_start?: boolean }) =>
    api.post<Task>('/tasks', data),

  getProgress: (runId: string) =>
    api.get<TaskProgress>(`/tasks/${runId}/progress`),

  getStats: (runId: string) =>
    api.get<TaskStats>(`/tasks/${runId}/stats`),

  getReport: (runId: string) =>
    api.get(`/tasks/${runId}/report`),

  cancel: (runId: string) =>
    api.post(`/tasks/${runId}/cancel`),

  retry: (runId: string) =>
    api.post(`/tasks/${runId}/retry`),

  delete: (runId: string) =>
    api.delete(`/tasks/${runId}`),

  export: (runId: string, format: string) =>
    api.post(`/tasks/${runId}/export`, { format }),
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
}

export const datasetApi = {
  list: () =>
    api.get<Dataset[]>('/datasets'),

  get: (name: string, limit?: number) =>
    api.get(`/datasets/${name}`, { params: { limit } }),

  validate: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/datasets/validate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  upload: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  delete: (name: string) =>
    api.delete(`/datasets/${name}`),
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
