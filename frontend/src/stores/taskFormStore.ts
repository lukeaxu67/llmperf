/**
 * Task form state management store using Zustand
 */

import { create } from 'zustand'
import {
  TaskFormState,
  ExecutorConfig,
  IteratorConfig,
  DEFAULT_ITERATOR_CONFIG,
  createNewExecutor,
  ExecutorType,
} from '@/types/taskConfig'
import { generateYaml, parseYaml } from '@/utils/yamlGenerator'

type TaskType = 'benchmark' | 'monitoring'

interface TaskFormStore extends TaskFormState {
  // Task type
  taskType: TaskType
  setTaskType: (type: TaskType) => void

  // Step navigation
  setStep: (step: number) => void
  nextStep: () => void
  prevStep: () => void

  // Task description
  setTaskDescription: (description: string) => void

  // Dataset selection
  setSelectedDataset: (dataset: string | null) => void

  // Iterator config
  setIteratorConfig: (config: Partial<IteratorConfig>) => void
  setMutationChain: (chain: string[]) => void
  setMaxRounds: (rounds: number | null) => void
  setMaxTotalSeconds: (seconds: number | null) => void

  // Executors
  addExecutor: (type?: ExecutorType) => void
  updateExecutor: (id: string, updates: Partial<ExecutorConfig>) => void
  removeExecutor: (id: string) => void
  duplicateExecutor: (id: string) => void
  reorderExecutors: (fromIndex: number, toIndex: number) => void
  setExecutorPriceStatus: (id: string, hasPrice: boolean) => void

  // YAML operations
  generateYamlContent: () => string
  loadFromYaml: (yamlContent: string) => void

  // Form state
  reset: () => void
  isValid: () => { valid: boolean; errors: string[] }
}

const initialState: TaskFormState = {
  taskDescription: '',
  selectedDataset: null,
  iteratorConfig: { ...DEFAULT_ITERATOR_CONFIG },
  executors: [],
  currentStep: 0,
}

export const useTaskFormStore = create<TaskFormStore>((set, get) => ({
  ...initialState,
  taskType: 'benchmark',

  // Task type
  setTaskType: (type) => set({ taskType: type }),

  // Step navigation
  setStep: (step) => set({ currentStep: step }),

  nextStep: () => {
    const { currentStep } = get()
    if (currentStep < 3) {
      set({ currentStep: currentStep + 1 })
    }
  },

  prevStep: () => {
    const { currentStep } = get()
    if (currentStep > 0) {
      set({ currentStep: currentStep - 1 })
    }
  },

  // Task description
  setTaskDescription: (description) => set({ taskDescription: description }),

  // Dataset selection
  setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),

  // Iterator config
  setIteratorConfig: (config) =>
    set((state) => ({
      iteratorConfig: { ...state.iteratorConfig, ...config },
    })),

  setMutationChain: (chain) =>
    set((state) => ({
      iteratorConfig: { ...state.iteratorConfig, mutation_chain: chain },
    })),

  setMaxRounds: (rounds) =>
    set((state) => ({
      iteratorConfig: { ...state.iteratorConfig, max_rounds: rounds },
    })),

  setMaxTotalSeconds: (seconds) =>
    set((state) => ({
      iteratorConfig: { ...state.iteratorConfig, max_total_seconds: seconds },
    })),

  // Executors
  addExecutor: (type = 'mock') => {
    const newExecutor = createNewExecutor(type)
    set((state) => ({
      executors: [...state.executors, newExecutor],
    }))
    return newExecutor.id
  },

  updateExecutor: (id, updates) =>
    set((state) => ({
      executors: state.executors.map((exec) =>
        exec.id === id ? { ...exec, ...updates } : exec
      ),
    })),

  removeExecutor: (id) =>
    set((state) => ({
      executors: state.executors.filter((exec) => exec.id !== id),
    })),

  duplicateExecutor: (id) => {
    const { executors } = get()
    const executor = executors.find((e) => e.id === id)
    if (executor) {
      const newExecutor: ExecutorConfig = {
        ...executor,
        id: `executor-${Date.now().toString(36)}`,
        name: `${executor.name} (Copy)`,
      }
      set({ executors: [...executors, newExecutor] })
    }
  },

  reorderExecutors: (fromIndex, toIndex) =>
    set((state) => {
      const newExecutors = [...state.executors]
      const [removed] = newExecutors.splice(fromIndex, 1)
      newExecutors.splice(toIndex, 0, removed)
      return { executors: newExecutors }
    }),

  setExecutorPriceStatus: (id, hasPrice) =>
    set((state) => ({
      executors: state.executors.map((exec) =>
        exec.id === id ? { ...exec, hasPrice } : exec
      ),
    })),

  // YAML operations
  generateYamlContent: () => {
    const state = get()
    return generateYaml({
      taskDescription: state.taskDescription,
      selectedDataset: state.selectedDataset,
      iteratorConfig: state.iteratorConfig,
      executors: state.executors,
      currentStep: state.currentStep,
      taskType: state.taskType,
    })
  },

  loadFromYaml: (yamlContent) => {
    try {
      const parsed = parseYaml(yamlContent)
      set(parsed)
    } catch (error) {
      console.error('Failed to parse YAML:', error)
      throw error
    }
  },

  // Form state
  reset: () => set({ ...initialState, taskType: 'benchmark' }),

  isValid: () => {
    const state = get()
    const errors: string[] = []

    // Step 1: Dataset validation
    if (!state.selectedDataset) {
      errors.push('请选择一个数据集')
    }

    // Step 2: Iterator config validation
    if (state.iteratorConfig.max_rounds !== null && state.iteratorConfig.max_rounds < 1) {
      errors.push('执行轮数必须大于 0')
    }

    // Step 3: Executors validation
    if (state.executors.length === 0) {
      errors.push('至少需要配置一个执行器')
    }

    for (const exec of state.executors) {
      if (!exec.id) {
        errors.push('执行器 ID 不能为空')
      }
      if (!exec.name) {
        errors.push(`执行器 ${exec.id} 的名称不能为空`)
      }
      if (!exec.type) {
        errors.push(`执行器 ${exec.id} 的类型不能为空`)
      }
      if (exec.type !== 'mock') {
        if (!exec.model) {
          errors.push(`执行器 ${exec.id} 需要指定模型名称`)
        }
        if (!exec.api_url) {
          errors.push(`执行器 ${exec.id} 需要配置 API URL`)
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    }
  },
}))

export default useTaskFormStore
