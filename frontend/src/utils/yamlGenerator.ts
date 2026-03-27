/**
 * YAML generation and parsing utilities for task configuration
 */

import yaml from 'js-yaml'
import {
  TaskFormState,
  RunConfig,
  ExecutorConfig,
  IteratorConfig,
  DEFAULT_ITERATOR_CONFIG,
} from '@/types/taskConfig'

/**
 * Generate YAML from form state
 */
export function generateYaml(state: TaskFormState): string {
  const datasetType = state.selectedDatasetType
    || state.selectedDatasetPath?.split('.').pop()
    || 'jsonl'
  const datasetPath = state.selectedDatasetPath
    || `data/datasets/${state.selectedDataset || 'default'}.${datasetType}`
  const config: RunConfig = {
    info: state.taskDescription || 'Untitled Task',
    dataset: {
      source: {
        type: datasetType,
        name: state.selectedDataset || 'default',
        config: {
          path: datasetPath,
        },
      },
      iterator: state.iteratorConfig || DEFAULT_ITERATOR_CONFIG,
    },
    executors: state.executors.map((exec) => ({
      id: exec.id,
      name: exec.name,
      type: exec.type,
      impl: exec.impl,
      concurrency: exec.concurrency,
      ...(exec.after && exec.after.length > 0 && { after: exec.after }),
      ...(exec.model && { model: exec.model }),
      ...(exec.api_url && { api_url: exec.api_url }),
      ...(exec.api_key && { api_key: exec.api_key }),
      ...(exec.param && Object.keys(exec.param).length > 0 && { param: exec.param }),
      ...(exec.rate && { rate: exec.rate }),
    })),
  }

  return yaml.dump(config, {
    indent: 2,
    lineWidth: -1,
    noRefs: true,
    sortKeys: false,
    quotingType: '"',
    forceQuotes: false,
  })
}

/**
 * Parse YAML content to form state
 */
export function parseYaml(content: string): TaskFormState {
  try {
    const config = yaml.load(content) as RunConfig

    // Extract dataset name from source config
    let datasetName = config.dataset?.source?.name
    const datasetPath = config.dataset?.source?.config?.path as string | undefined
    let datasetType = config.dataset?.source?.type || null
    if (!datasetName && config.dataset?.source?.config?.path) {
      // Try to extract from path
      const path = config.dataset.source.config.path as string
      const normalizedPath = path.replace(/\\/g, '/')
      const match = normalizedPath.match(/\/([^/]+)\.([^.]+)$/)
      if (match) {
        datasetName = match[1]
        datasetType = datasetType || match[2]
      }
    }

    // Parse iterator config
    const iteratorConfig: IteratorConfig = config.dataset?.iterator || {
      ...DEFAULT_ITERATOR_CONFIG,
    }

    // Parse executors
    const executors: ExecutorConfig[] = (config.executors || []).map((exec) => ({
      id: exec.id,
      name: exec.name,
      type: exec.type as any,
      impl: exec.impl || 'chat',
      concurrency: exec.concurrency || 1,
      after: exec.after || [],
      model: exec.model || null,
      api_url: exec.api_url || null,
      api_key: exec.api_key || null,
      param: exec.param || {},
      rate: exec.rate || null,
    }))

    return {
      taskDescription: config.info || '',
      selectedDataset: datasetName || null,
      selectedDatasetPath: datasetPath || null,
      selectedDatasetType: datasetType || null,
      iteratorConfig,
      executors,
      currentStep: 0,
    }
  } catch (error) {
    console.error('Failed to parse YAML:', error)
    throw new Error('Invalid YAML format')
  }
}

/**
 * Validate YAML content
 */
export function validateYaml(content: string): { valid: boolean; error?: string } {
  try {
    const config = yaml.load(content) as RunConfig

    // Check required fields
    if (!config.dataset) {
      return { valid: false, error: 'Missing required field: dataset' }
    }

    if (!config.dataset.source) {
      return { valid: false, error: 'Missing required field: dataset.source' }
    }

    if (!config.executors || config.executors.length === 0) {
      return { valid: false, error: 'At least one executor is required' }
    }

    // Validate executors
    for (const exec of config.executors) {
      if (!exec.id) {
        return { valid: false, error: 'Executor missing required field: id' }
      }
      if (!exec.type) {
        return { valid: false, error: `Executor ${exec.id} missing required field: type` }
      }
    }

    return { valid: true }
  } catch (error: any) {
    return { valid: false, error: error.message || 'Invalid YAML syntax' }
  }
}

/**
 * Export YAML to file download
 */
export function exportYaml(content: string, filename: string = 'task-config.yaml'): void {
  const blob = new Blob([content], { type: 'text/yaml' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Read YAML from file upload
 */
export function readYamlFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target?.result as string
      resolve(content)
    }
    reader.onerror = () => {
      reject(new Error('Failed to read file'))
    }
    reader.readAsText(file)
  })
}
