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
    const message = error.response?.data?.detail || error.message || '请求失败'
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

// API Functions
export const taskApi = {
  list: (params?: { status?: string; limit?: number; offset?: number }) =>
    api.get<{ tasks: Task[]; total: number }>('/tasks', { params }),

  get: (runId: string) =>
    api.get<Task>(`/tasks/${runId}`),

  create: (data: { config_path?: string; config_content?: string; run_id?: string }) =>
    api.post<Task>('/tasks', data),

  getProgress: (runId: string) =>
    api.get<TaskProgress>(`/tasks/${runId}/progress`),

  getStats: (runId: string) =>
    api.get<TaskStats>(`/tasks/${runId}/stats`),

  cancel: (runId: string) =>
    api.post(`/tasks/${runId}/cancel`),

  retry: (runId: string) =>
    api.post(`/tasks/${runId}/retry`),

  delete: (runId: string) =>
    api.delete(`/tasks/${runId}`),
}

export const analysisApi = {
  getSummary: (runId: string) =>
    api.get<AnalysisSummary>(`/analysis/${runId}/summary`),

  getTimeseries: (runId: string, metric: string, interval?: string) =>
    api.get(`/analysis/${runId}/timeseries`, { params: { metric, interval } }),

  compare: (runId: string) =>
    api.get(`/analysis/${runId}/compare`),

  getAnomalies: (runId: string, sensitivity?: number) =>
    api.get(`/analysis/${runId}/anomalies`, { params: { sensitivity } }),

  export: (runId: string, format: string) =>
    api.post(`/analysis/${runId}/export`, { format }),

  getHistory: (params?: { limit?: number; days?: number }) =>
    api.get('/analysis/history', { params }),
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
