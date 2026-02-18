import { create } from 'zustand'
import { Task, TaskProgress, Dataset, ConfigTemplate } from '@/services/api'
import { taskApi, datasetApi, configApi } from '@/services/api'

interface AppState {
  // Tasks
  tasks: Task[]
  totalTasks: number
  loadingTasks: boolean
  fetchTasks: (params?: { status?: string; limit?: number; offset?: number }) => Promise<void>

  // Current task
  currentTask: Task | null
  currentTaskProgress: TaskProgress | null
  fetchTask: (runId: string) => Promise<void>
  fetchTaskProgress: (runId: string) => Promise<void>
  clearCurrentTask: () => void

  // Datasets
  datasets: Dataset[]
  loadingDatasets: boolean
  fetchDatasets: () => Promise<void>

  // Templates
  templates: ConfigTemplate[]
  loadingTemplates: boolean
  fetchTemplates: () => Promise<void>

  // UI State
  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  tasks: [],
  totalTasks: 0,
  loadingTasks: false,
  datasets: [],
  loadingDatasets: false,
  templates: [],
  loadingTemplates: false,
  currentTask: null,
  currentTaskProgress: null,
  sidebarCollapsed: false,

  // Actions
  fetchTasks: async (params) => {
    set({ loadingTasks: true })
    try {
      const response = await taskApi.list(params) as any
      set({
        tasks: response.tasks || [],
        totalTasks: response.total || 0,
        loadingTasks: false,
      })
    } catch (error) {
      set({ loadingTasks: false })
      throw error
    }
  },

  fetchTask: async (runId: string) => {
    try {
      const task = await taskApi.get(runId) as any
      set({ currentTask: task })
    } catch (error) {
      set({ currentTask: null })
      throw error
    }
  },

  fetchTaskProgress: async (runId: string) => {
    try {
      const progress = await taskApi.getProgress(runId) as any
      set({ currentTaskProgress: progress })
    } catch (error) {
      set({ currentTaskProgress: null })
    }
  },

  clearCurrentTask: () => {
    set({ currentTask: null, currentTaskProgress: null })
  },

  fetchDatasets: async () => {
    set({ loadingDatasets: true })
    try {
      const datasets = await datasetApi.list() as any
      set({ datasets: datasets || [], loadingDatasets: false })
    } catch (error) {
      set({ loadingDatasets: false })
      throw error
    }
  },

  fetchTemplates: async () => {
    set({ loadingTemplates: true })
    try {
      const templates = await configApi.listTemplates() as any
      set({ templates: templates || [], loadingTemplates: false })
    } catch (error) {
      set({ loadingTemplates: false })
      throw error
    }
  },

  toggleSidebar: () => {
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }))
  },
}))
