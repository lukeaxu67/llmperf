import { Tag } from 'antd'
import {
  ClockCircleOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
} from '@ant-design/icons'

type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

interface StatusTagProps {
  status: TaskStatus
}

const statusConfig: Record<TaskStatus, { color: string; icon: React.ReactNode; text: string }> = {
  pending: { color: 'warning', icon: <ClockCircleOutlined />, text: '等待中' },
  running: { color: 'processing', icon: <LoadingOutlined spin />, text: '运行中' },
  completed: { color: 'success', icon: <CheckCircleOutlined />, text: '已完成' },
  failed: { color: 'error', icon: <CloseCircleOutlined />, text: '失败' },
  cancelled: { color: 'default', icon: <StopOutlined />, text: '已取消' },
}

export default function StatusTag({ status }: StatusTagProps) {
  const config = statusConfig[status] || statusConfig.pending

  return (
    <Tag color={config.color} icon={config.icon}>
      {config.text}
    </Tag>
  )
}
