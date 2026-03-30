import { Progress, Space, Typography } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined, FileTextOutlined } from '@ant-design/icons'

const { Text } = Typography

interface TaskProgressBarProps {
  percent: number
  successCount: number
  errorCount: number
  total: number
  size?: 'small' | 'default'
}

export default function TaskProgressBar({
  percent,
  successCount,
  errorCount,
  total,
  size = 'default',
}: TaskProgressBarProps) {
  const isSmall = size === 'small'

  return (
    <div>
      <Progress
        percent={Math.round(percent)}
        size={size}
        strokeColor={{
          '0%': '#52c41a',
          '100%': '#73d13d',
        }}
        trailColor="var(--color-border-secondary)"
        format={(p) => (
          <span style={{ fontSize: isSmall ? 12 : 14, fontWeight: 500 }}>
            {p}%
          </span>
        )}
      />
      <Space size={isSmall ? 'small' : 'large'} style={{ marginTop: 4 }}>
        <Text
          style={{
            color: '#52c41a',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            fontSize: isSmall ? 12 : 14,
          }}
        >
          <CheckCircleOutlined />
          成功: {successCount}
        </Text>
        <Text
          style={{
            color: '#ff4d4f',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            fontSize: isSmall ? 12 : 14,
          }}
        >
          <CloseCircleOutlined />
          失败: {errorCount}
        </Text>
        <Text
          type="secondary"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            fontSize: isSmall ? 12 : 14,
          }}
        >
          <FileTextOutlined />
          总计: {total}
        </Text>
      </Space>
    </div>
  )
}
