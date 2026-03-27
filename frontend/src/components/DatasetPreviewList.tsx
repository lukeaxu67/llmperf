import { Card, Empty, Space, Tag, Typography } from 'antd'

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
      return `样本 ${index + 1} · ${id}`
    }
  }
  return `样本 ${index + 1}`
}

export default function DatasetPreviewList({
  records,
  emptyDescription = '暂无可展示的预览数据',
}: DatasetPreviewListProps) {
  if (!records.length) {
    return <Empty description={emptyDescription} />
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {records.map((record, index) => (
        <Card
          key={`dataset-preview-${index}`}
          size="small"
          title={<Text strong>{recordTitle(record, index)}</Text>}
          styles={{
            body: {
              padding: 16,
              background: '#fafcff',
            },
          }}
        >
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            {normalizeMessages(record).map((message, messageIndex) => {
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
                    }}
                  >
                    {message.content || '-'}
                  </Paragraph>
                </div>
              )
            })}
          </Space>
        </Card>
      ))}
    </Space>
  )
}
