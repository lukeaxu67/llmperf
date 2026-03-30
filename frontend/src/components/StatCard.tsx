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
  variant?: 'default' | 'success' | 'warning' | 'error'
}

export default function StatCard({
  title,
  value,
  suffix,
  prefix,
  tooltip,
  loading,
  color,
  variant = 'default',
}: StatCardProps) {
  const variantColors = {
    default: 'var(--color-primary)',
    success: 'var(--color-success)',
    warning: 'var(--color-warning)',
    error: 'var(--color-error)',
  }

  const finalColor = color || variantColors[variant]

  return (
    <Card
      className={`stat-card ${variant !== 'default' ? variant : ''}`}
      loading={loading}
      styles={{
        body: { padding: 20 },
      }}
      style={{
        height: '100%',
        borderRadius: 12,
        border: '1px solid var(--color-border-secondary)',
        background: 'var(--color-bg-container)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <div
            style={{
              color: 'var(--color-text-secondary)',
              marginBottom: 8,
              fontSize: 14,
              fontWeight: 500,
            }}
          >
            {title}
            {tooltip && (
              <Tooltip title={tooltip}>
                <InfoCircleOutlined
                  style={{
                    marginLeft: 6,
                    color: 'var(--color-text-tertiary)',
                    fontSize: 12,
                  }}
                />
              </Tooltip>
            )}
          </div>
          <Statistic
            value={value}
            suffix={suffix}
            prefix={prefix}
            valueStyle={{
              color: finalColor,
              fontWeight: 600,
              fontSize: 28,
            }}
          />
        </div>
        {prefix && (
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              background: `linear-gradient(135deg, ${finalColor}15 0%, ${finalColor}08 100%)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: finalColor,
              fontSize: 20,
            }}
          >
            {prefix}
          </div>
        )}
      </div>
    </Card>
  )
}
