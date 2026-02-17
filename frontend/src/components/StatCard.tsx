import { Card, Statistic, Tooltip } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'
import type { ReactNode } from 'react'

interface StatCardProps {
  title: string
  value: number | string
  suffix?: string
  prefix?: ReactNode
  tooltip?: string
  loading?: boolean
  trend?: {
    value: number
    isUp: boolean
  }
  color?: string
}

export default function StatCard({
  title,
  value,
  suffix,
  prefix,
  tooltip,
  loading,
  color = '#1677ff',
}: StatCardProps) {
  return (
    <Card
      className="stat-card"
      loading={loading}
      styles={{
        body: { padding: 20 },
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ color: '#666', marginBottom: 8, fontSize: 14 }}>
            {title}
            {tooltip && (
              <Tooltip title={tooltip}>
                <InfoCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
            )}
          </div>
          <Statistic
            value={value}
            suffix={suffix}
            prefix={prefix}
            valueStyle={{ color, fontWeight: 600, fontSize: 28 }}
          />
        </div>
      </div>
    </Card>
  )
}
