import { Tag } from 'antd'
import {
  ClockCircleOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons'

type TaskStatus = 'scheduled' | 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'

interface StatusTagProps {
  status: TaskStatus
  style?: React.CSSProperties
}

interface StatusConfig {
  color: string
  background: string
  border: string
  icon: React.ReactNode
  text: string
}

const statusConfig: Record<TaskStatus, StatusConfig> = {
  scheduled: {
    color: '#1677ff',
    background: '#e6f4ff',
    border: '#91caff',
    icon: <ClockCircleOutlined />,
    text: '已定时',
  },
  pending: {
    color: '#faad14',
    background: '#fffbe6',
    border: '#ffe58f',
    icon: <ClockCircleOutlined />,
    text: '等待中',
  },
  running: {
    color: '#1677ff',
    background: '#e6f4ff',
    border: '#91caff',
    icon: <LoadingOutlined spin />,
    text: '运行中',
  },
  paused: {
    color: '#fa8c16',
    background: '#fff7e6',
    border: '#ffd591',
    icon: <PauseCircleOutlined />,
    text: '已暂停',
  },
  completed: {
    color: '#52c41a',
    background: '#f6ffed',
    border: '#b7eb8f',
    icon: <CheckCircleOutlined />,
    text: '已完成',
  },
  failed: {
    color: '#ff4d4f',
    background: '#fff2f0',
    border: '#ffa39e',
    icon: <CloseCircleOutlined />,
    text: '失败',
  },
  cancelled: {
    color: '#666666',
    background: '#f5f5f5',
    border: '#d9d9d9',
    icon: <StopOutlined />,
    text: '已取消',
  },
}

export default function StatusTag({ status, style }: StatusTagProps) {
  const config = statusConfig[status] || statusConfig.pending

  return (
    <Tag
      icon={config.icon}
      style={{
        color: config.color,
        background: config.background,
        borderColor: config.border,
        borderRadius: 6,
        padding: '2px 10px',
        fontSize: 13,
        fontWeight: 500,
        border: `1px solid ${config.border}`,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        ...style,
      }}
    >
      {config.text}
    </Tag>
  )
}
