/**
 * Type definitions for task configuration
 * Aligned with backend models in src/llmperf/config/models.py
 */

// Rate limiting configuration
export interface RateConfig {
  qps?: number | null
  interval_seconds?: number | null
}

// Executor configuration
export interface ExecutorConfig {
  id: string
  name: string
  type: ExecutorType
  impl: string
  after?: string[]
  concurrency: number
  api_url?: string | null
  api_key?: string | null
  model?: string | null
  param?: Record<string, any>
  rate?: RateConfig | null
  // Price status (computed, not in YAML)
  hasPrice?: boolean
}

// Supported executor types
export type ExecutorType =
  | 'openai'
  | 'qianwen'
  | 'zhipu'
  | 'deepseek'
  | 'spark'
  | 'hunyuan'
  | 'huoshan'
  | 'moonshot'
  | 'mock'

// All available executor types for selection
export const EXECUTOR_TYPES: { value: ExecutorType; label: string }[] = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'qianwen', label: 'Qianwen (通义千问)' },
  { value: 'zhipu', label: 'Zhipu (智谱)' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'spark', label: 'Spark (讯飞星火)' },
  { value: 'hunyuan', label: 'Hunyuan (腾讯混元)' },
  { value: 'huoshan', label: 'Huoshan (字节火山)' },
  { value: 'moonshot', label: 'Moonshot (月之暗面)' },
  { value: 'mock', label: 'Mock (测试用)' },
]

// Implementation types
export type ExecutorImpl = 'chat' | 'completion'

export const EXECUTOR_IMPLS: { value: ExecutorImpl; label: string }[] = [
  { value: 'chat', label: 'Chat API' },
  { value: 'completion', label: 'Completion API' },
]

// Dataset iterator options
export interface IteratorConfig {
  mutation_chain: string[]
  max_rounds: number | null
  max_total_seconds: number | null
}

// Mutation methods
export type MutationMethod = 'identity' | 'rdmprefix' | 'rmdupspaces' | 'rmcomments'

export const MUTATION_METHODS: { value: MutationMethod; label: string; description: string }[] = [
  { value: 'identity', label: 'Identity', description: '不进行任何变换' },
  { value: 'rdmprefix', label: 'Remove Prefix', description: '移除prompt前缀' },
  { value: 'rmdupspaces', label: 'Remove Duplicate Spaces', description: '移除重复空格' },
  { value: 'rmcomments', label: 'Remove Comments', description: '移除注释' },
]

// Dataset source configuration
export interface DatasetSourceConfig {
  type: string
  name: string | null
  config: Record<string, any>
}

// Dataset configuration
export interface DatasetConfig {
  source: DatasetSourceConfig
  iterator: IteratorConfig | null
}

// Pricing entry
export interface PricingEntry {
  provider: string
  model: string
  unit: 'per_1k' | 'per_1m'
  input_price: number
  output_price: number
  cache_input_discount?: number
  cache_output_discount?: number
  currency: string
}

// Multiprocess configuration
export interface MultiprocessConfig {
  per_executor: boolean
  max_workers: number | null
}

// Full run configuration
export interface RunConfig {
  info: string
  db_path?: string | null
  dataset: DatasetConfig
  executors: ExecutorConfig[]
  pricing?: PricingEntry[]
  multiprocess?: MultiprocessConfig | null
}

// Form state for the task creation wizard
export interface TaskFormState {
  // Step 1: Dataset selection
  taskDescription: string
  selectedDataset: string | null
  selectedDatasetPath?: string | null
  selectedDatasetType?: string | null

  // Step 2: Execution settings
  iteratorConfig: IteratorConfig

  // Step 3: Executors
  executors: ExecutorConfig[]

  // Metadata
  currentStep: number
  taskType?: 'benchmark' | 'monitoring'
}

// Default form state
export const DEFAULT_ITERATOR_CONFIG: IteratorConfig = {
  mutation_chain: ['identity'],
  max_rounds: 1,
  max_total_seconds: null,
}

export const DEFAULT_EXECUTOR_CONFIG: Partial<ExecutorConfig> = {
  impl: 'chat',
  concurrency: 1,
  after: [],
  param: {},
  rate: null,
}

// Test run response
export interface TestRunResponse {
  success: boolean
  duration_ms: number
  first_token_ms: number
  tokens_per_second: number
  response: string
  error: string
}

// Helper function to generate unique executor ID
export function generateExecutorId(): string {
  return `executor-${Date.now().toString(36)}-${Math.random().toString(36).substr(2, 5)}`
}

// Helper function to create a new executor with defaults
export function createNewExecutor(type: ExecutorType = 'mock'): ExecutorConfig {
  return {
    id: generateExecutorId(),
    name: `New ${type} Executor`,
    type,
    impl: 'chat',
    concurrency: 1,
    after: [],
    param: {},
    rate: null,
    model: type === 'mock' ? 'mock-model' : null,
    api_url: null,
    api_key: null,
  }
}
