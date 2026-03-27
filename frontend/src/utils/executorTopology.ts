import { ExecutorConfig } from '@/types/taskConfig'
import { ExecutorProgress, TaskTopology } from '@/services/api'

export function buildTopologyFromExecutors(executors: ExecutorConfig[]): TaskTopology {
  const levelMap = new Map<string, number>()
  const executorMap = new Map(executors.map((executor) => [executor.id, executor]))

  const visit = (executorId: string, seen: Set<string>): number => {
    if (levelMap.has(executorId)) {
      return levelMap.get(executorId) || 0
    }
    if (seen.has(executorId)) {
      return 0
    }
    seen.add(executorId)
    const executor = executorMap.get(executorId)
    const level = executor?.after?.length
      ? 1 + Math.max(...executor.after.filter((dep) => executorMap.has(dep)).map((dep) => visit(dep, seen)), 0)
      : 0
    seen.delete(executorId)
    levelMap.set(executorId, level)
    return level
  }

  executors.forEach((executor) => visit(executor.id, new Set()))

  const layersMap = new Map<number, string[]>()
  const edges: Array<{ source: string; target: string }> = []
  const downstream = new Map<string, number>(executors.map((executor) => [executor.id, 0]))

  executors.forEach((executor) => {
    const level = levelMap.get(executor.id) || 0
    layersMap.set(level, [...(layersMap.get(level) || []), executor.id])
    if (executor.after?.length) {
      executor.after.forEach((dep) => {
        edges.push({ source: dep, target: executor.id })
        downstream.set(dep, (downstream.get(dep) || 0) + 1)
      })
    } else {
      edges.push({ source: '__start__', target: executor.id })
    }
  })

  executors.forEach((executor) => {
    if ((downstream.get(executor.id) || 0) === 0) {
      edges.push({ source: executor.id, target: '__end__' })
    }
  })

  const nodes = [
    { id: '__start__', name: 'Start', kind: 'boundary' as const, status: 'completed', level: -1, progress_percent: 100 },
    ...executors.map((executor) => ({
      id: executor.id,
      name: executor.name,
      kind: 'executor' as const,
      status: 'pending',
      level: levelMap.get(executor.id) || 0,
      model: executor.model,
      provider: executor.type,
      progress_percent: 0,
    })),
    {
      id: '__end__',
      name: 'End',
      kind: 'boundary' as const,
      status: 'pending',
      level: Math.max(...Array.from(levelMap.values()), 0) + 1,
      progress_percent: 0,
    },
  ]

  return {
    nodes,
    edges,
    layers: Array.from(layersMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([level, node_ids]) => ({ level, node_ids })),
  }
}

export function mergeTopologyProgress(
  topology: TaskTopology | undefined,
  executors: ExecutorProgress[] | undefined,
): TaskTopology {
  if (!topology) {
    return { nodes: [], edges: [], layers: [] }
  }
  const progressMap = new Map((executors || []).map((executor) => [executor.id, executor]))
  return {
    ...topology,
    nodes: topology.nodes.map((node) => {
      const progress = progressMap.get(node.id)
      return progress
        ? {
            ...node,
            name: progress.name,
            status: progress.status,
            progress_percent: progress.progress_percent,
            model: progress.model,
            provider: progress.provider,
          }
        : node
    }),
  }
}
