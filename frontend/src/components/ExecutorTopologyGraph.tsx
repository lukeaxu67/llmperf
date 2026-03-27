import { Card, Progress, Tag, Typography } from 'antd'
import { TaskTopology } from '@/services/api'

const { Text } = Typography

const NODE_WIDTH = 220
const NODE_HEIGHT = 112
const H_GAP = 90
const V_GAP = 36
const PADDING = 32

const STATUS_COLORS: Record<string, string> = {
  completed: '#52c41a',
  running: '#1677ff',
  paused: '#faad14',
  blocked: '#bfbfbf',
  pending: '#d9d9d9',
  failed: '#ff4d4f',
  cancelled: '#ff7875',
}

interface ExecutorTopologyGraphProps {
  topology: TaskTopology
  height?: number
}

export default function ExecutorTopologyGraph({
  topology,
  height,
}: ExecutorTopologyGraphProps) {
  if (!topology?.nodes?.length) {
    return null
  }

  const layers = new Map<number, typeof topology.nodes>()
  topology.nodes.forEach((node) => {
    const level = node.level ?? 0
    layers.set(level, [...(layers.get(level) || []), node])
  })
  const orderedLevels = Array.from(layers.keys()).sort((a, b) => a - b)
  const maxRows = Math.max(...Array.from(layers.values()).map((nodes) => nodes.length), 1)
  const graphHeight = height || Math.max(260, maxRows * (NODE_HEIGHT + V_GAP) + PADDING * 2)
  const graphWidth = orderedLevels.length * (NODE_WIDTH + H_GAP) + PADDING * 2

  const positions = new Map<string, { x: number; y: number }>()
  orderedLevels.forEach((level, layerIndex) => {
    const nodes = layers.get(level) || []
    const layerHeight = nodes.length * NODE_HEIGHT + Math.max(0, nodes.length - 1) * V_GAP
    const offsetY = (graphHeight - layerHeight) / 2
    nodes.forEach((node, nodeIndex) => {
      positions.set(node.id, {
        x: PADDING + layerIndex * (NODE_WIDTH + H_GAP),
        y: offsetY + nodeIndex * (NODE_HEIGHT + V_GAP),
      })
    })
  })

  return (
    <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
      <div style={{ position: 'relative', width: graphWidth, height: graphHeight }}>
        <svg
          width={graphWidth}
          height={graphHeight}
          style={{ position: 'absolute', inset: 0 }}
        >
          {topology.edges.map((edge, index) => {
            const source = positions.get(edge.source)
            const target = positions.get(edge.target)
            if (!source || !target) {
              return null
            }
            const startX = source.x + NODE_WIDTH
            const startY = source.y + NODE_HEIGHT / 2
            const endX = target.x
            const endY = target.y + NODE_HEIGHT / 2
            const midX = (startX + endX) / 2
            return (
              <path
                key={`${edge.source}-${edge.target}-${index}`}
                d={`M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`}
                fill="none"
                stroke="#bfbfbf"
                strokeWidth="2"
              />
            )
          })}
        </svg>

        {topology.nodes.map((node) => {
          const position = positions.get(node.id)
          if (!position) {
            return null
          }
          const color = STATUS_COLORS[node.status || 'pending'] || STATUS_COLORS.pending
          const isBoundary = node.kind === 'boundary'
          return (
            <Card
              key={node.id}
              size="small"
              style={{
                position: 'absolute',
                left: position.x,
                top: position.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
                borderColor: color,
                boxShadow: `0 10px 24px ${color}22`,
              }}
              styles={{ body: { padding: 14 } }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <div>
                  <Text strong>{node.name}</Text>
                  {!isBoundary && node.model && (
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>{node.model}</Text>
                    </div>
                  )}
                </div>
                <Tag color={color}>{node.status || 'pending'}</Tag>
              </div>
              {!isBoundary && (
                <>
                  <Progress
                    percent={Math.round(node.progress_percent || 0)}
                    size="small"
                    strokeColor={color}
                    style={{ marginTop: 12, marginBottom: 8 }}
                  />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {node.provider || 'executor'}
                  </Text>
                </>
              )}
            </Card>
          )
        })}
      </div>
    </div>
  )
}
