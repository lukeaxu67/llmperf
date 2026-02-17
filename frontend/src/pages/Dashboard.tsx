import { useEffect, useState } from 'react'
import { Row, Col, Card, Table, Typography, Space, Tag, Empty, Spin } from 'antd'
import {
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DollarOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import { Line, Pie, Column } from '@ant-design/plots'
import dayjs from 'dayjs'
import StatCard from '@/components/StatCard'
import StatusTag from '@/components/StatusTag'
import { taskApi, analysisApi, Task } from '@/services/api'

const { Title, Text } = Typography

interface DashboardStats {
  totalTasks: number
  runningTasks: number
  completedTasks: number
  failedTasks: number
  totalCost: number
  avgLatency: number
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<DashboardStats>({
    totalTasks: 0,
    runningTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    totalCost: 0,
    avgLatency: 0,
  })
  const [recentTasks, setRecentTasks] = useState<Task[]>([])
  const [historyRuns, setHistoryRuns] = useState<any[]>([])

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    setLoading(true)
    try {
      // Fetch tasks
      const tasksRes = await taskApi.list({ limit: 100 }) as any
      const tasks = tasksRes.tasks || []
      setRecentTasks(tasks.slice(0, 5))

      // Calculate stats
      const runningTasks = tasks.filter((t: Task) => t.status === 'running').length
      const completedTasks = tasks.filter((t: Task) => t.status === 'completed').length
      const failedTasks = tasks.filter((t: Task) => t.status === 'failed').length

      setStats({
        totalTasks: tasks.length,
        runningTasks,
        completedTasks,
        failedTasks,
        totalCost: 0,
        avgLatency: 0,
      })

      // Fetch history
      try {
        const historyRes = await analysisApi.getHistory({ limit: 30 }) as any
        setHistoryRuns(historyRes.runs || [])
      } catch (e) {
        // Ignore history errors
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const taskColumns = [
    {
      title: '任务名称',
      dataIndex: 'task_name',
      key: 'task_name',
      ellipsis: true,
    },
    {
      title: 'Run ID',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (id: string) => <Text code copyable={{ text: id }}>{id.slice(0, 8)}...</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => <StatusTag status={status as any} />,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => dayjs(time).format('MM-DD HH:mm'),
    },
  ]

  // Task status distribution chart config
  const statusPieConfig = {
    appendPadding: 10,
    data: [
      { type: '运行中', value: stats.runningTasks },
      { type: '已完成', value: stats.completedTasks },
      { type: '失败', value: stats.failedTasks },
      { type: '其他', value: Math.max(0, stats.totalTasks - stats.runningTasks - stats.completedTasks - stats.failedTasks) },
    ],
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    innerRadius: 0.6,
    color: ['#1677ff', '#52c41a', '#ff4d4f', '#d9d9d9'],
    label: {
      type: 'inner',
      offset: '-50%',
      content: '{value}',
      style: {
        textAlign: 'center',
        fontSize: 12,
      },
    },
    legend: {
      position: 'bottom' as const,
    },
    interactions: [{ type: 'element-selected' }, { type: 'element-active' }],
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>系统概览</Title>

      {/* Stats Cards */}
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="总任务数"
            value={stats.totalTasks}
            prefix={<RocketOutlined />}
            color="#1677ff"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="运行中"
            value={stats.runningTasks}
            prefix={<ClockCircleOutlined />}
            color="#faad14"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="已完成"
            value={stats.completedTasks}
            prefix={<CheckCircleOutlined />}
            color="#52c41a"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="失败任务"
            value={stats.failedTasks}
            prefix={<CloseCircleOutlined />}
            color="#ff4d4f"
          />
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="任务状态分布" className="chart-container">
            {stats.totalTasks > 0 ? (
              <Pie {...statusPieConfig} />
            ) : (
              <Empty description="暂无数据" style={{ padding: 40 }} />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="最近任务" extra={<a href="/tasks">查看全部</a>}>
            {recentTasks.length > 0 ? (
              <Table
                columns={taskColumns}
                dataSource={recentTasks}
                rowKey="run_id"
                pagination={false}
                size="small"
              />
            ) : (
              <Empty description="暂无任务" style={{ padding: 40 }} />
            )}
          </Card>
        </Col>
      </Row>

      {/* Quick Actions */}
      <Card title="快捷操作" style={{ marginTop: 24 }}>
        <Space size="large">
          <a href="/tasks/create">
            <Tag color="blue" style={{ padding: '8px 16px', fontSize: 14 }}>
              <RocketOutlined /> 创建任务
            </Tag>
          </a>
          <a href="/datasets">
            <Tag color="green" style={{ padding: '8px 16px', fontSize: 14 }}>
              <DollarOutlined /> 管理数据集
            </Tag>
          </a>
          <a href="/analysis">
            <Tag color="purple" style={{ padding: '8px 16px', fontSize: 14 }}>
              <CheckCircleOutlined /> 查看分析
            </Tag>
          </a>
        </Space>
      </Card>
    </div>
  )
}
