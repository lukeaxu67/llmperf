import { Progress, Space, Typography } from 'antd'

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
  return (
    <div>
      <Progress
        percent={Math.round(percent)}
        size={size}
        strokeColor={{
          '0%': '#52c41a',
          '100%': '#1677ff',
        }}
        format={(p) => (
          <span style={{ fontSize: size === 'small' ? 12 : 14 }}>
            {p}%
          </span>
        )}
      />
      <Space size="large" style={{ marginTop: 4 }}>
        <Text type="success">成功: {successCount}</Text>
        <Text type="danger">失败: {errorCount}</Text>
        <Text type="secondary">总计: {total}</Text>
      </Space>
    </div>
  )
}
