import { useState } from 'react'
import { Button, Card, Empty, Space, Tag, Typography } from 'antd'

const { Paragraph, Text } = Typography

type MessageRecord = {
  role?: string
  content?: unknown
}

type PreviewRecord = Record<string, unknown> | string | number | boolean | null

interface DatasetPreviewListProps {
  records: PreviewRecord[]
  emptyDescription?: string
}

const roleMeta: Record<string, { label: string; color: string }> = {
  system: { label: 'System', color: 'purple' },
  user: { label: 'User', color: 'blue' },
  assistant: { label: 'Assistant', color: 'green' },
  tool: { label: 'Tool', color: 'gold' },
}

const DEFAULT_VISIBLE_MESSAGES = 4
const MESSAGE_MAX_HEIGHT = 160

function stringifyValue(value: unknown): string {
  if (value == null) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function normalizeMessages(record: PreviewRecord): Array<{ role: string; content: string }> {
  if (record == null || typeof record !== 'object' || Array.isArray(record)) {
    return [{ role: 'sample', content: stringifyValue(record) }]
  }

  const objectRecord = record as Record<string, unknown>
  const messages = objectRecord.messages
  if (Array.isArray(messages) && messages.length > 0) {
    return messages.map((message) => {
      const item = (message ?? {}) as MessageRecord
      return {
        role: item.role || 'message',
        content: stringifyValue(item.content),
      }
    })
  }

  if (typeof objectRecord.prompt === 'string' || typeof objectRecord.response === 'string') {
    return [
      {
        role: 'user',
        content: stringifyValue(objectRecord.prompt),
      },
      {
        role: 'assistant',
        content: stringifyValue(objectRecord.response),
      },
    ].filter((item) => item.content)
  }

  if (typeof objectRecord.content === 'string') {
    return [{ role: 'content', content: objectRecord.content }]
  }

  return [{ role: 'sample', content: stringifyValue(objectRecord) }]
}

function recordTitle(record: PreviewRecord, index: number): string {
  if (record && typeof record === 'object' && !Array.isArray(record)) {
    const id = (record as Record<string, unknown>).id
    if (typeof id === 'string' || typeof id === 'number') {
      return `样本 ${index + 1} / ${id}`
    }
  }
  return `样本 ${index + 1}`
}

export default function DatasetPreviewList({
  records,
  emptyDescription = '暂无可展示的预览数据',
}: DatasetPreviewListProps) {
  const [expandedIndexes, setExpandedIndexes] = useState<Set<number>>(new Set())

  if (!records.length) {
    return <Empty description={emptyDescription} />
  }

  const toggleExpanded = (index: number) => {
    setExpandedIndexes((previous) => {
      const next = new Set(previous)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {records.map((record, index) => {
        const normalizedMessages = normalizeMessages(record)
        const expanded = expandedIndexes.has(index)
        const visibleMessages = expanded
          ? normalizedMessages
          : normalizedMessages.slice(0, DEFAULT_VISIBLE_MESSAGES)
        const hiddenCount = Math.max(0, normalizedMessages.length - visibleMessages.length)

        return (
          <Card
            key={`dataset-preview-${index}`}
            size="small"
            title={<Text strong>{recordTitle(record, index)}</Text>}
            extra={(
              normalizedMessages.length > DEFAULT_VISIBLE_MESSAGES
                ? (
                  <Button type="link" size="small" onClick={() => toggleExpanded(index)}>
                    {expanded ? '收起多轮内容' : `展开全部 ${normalizedMessages.length} 轮`}
                  </Button>
                )
                : null
            )}
            styles={{
              body: {
                padding: 16,
                background: '#fafcff',
              },
            }}
          >
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              {visibleMessages.map((message, messageIndex) => {
                const meta = roleMeta[message.role] || {
                  label: message.role.toUpperCase(),
                  color: 'default',
                }
                return (
                  <div key={`dataset-preview-message-${index}-${messageIndex}`}>
                    <Tag color={meta.color}>{meta.label}</Tag>
                    <Paragraph
                      style={{
                        marginBottom: 0,
                        marginTop: 8,
                        padding: 12,
                        borderRadius: 10,
                        background: '#ffffff',
                        border: '1px solid #eaeef5',
                        whiteSpace: 'pre-wrap',
                        fontFamily: 'Consolas, Monaco, monospace',
                        maxHeight: MESSAGE_MAX_HEIGHT,
                        overflowY: 'auto',
                      }}
                    >
                      {message.content || '-'}
                    </Paragraph>
                  </div>
                )
              })}
              {hiddenCount > 0 && (
                <Text type="secondary">还有 {hiddenCount} 轮未展示，点击右上角可展开。</Text>
              )}
            </Space>
          </Card>
        )
      })}
    </Space>
  )
}
