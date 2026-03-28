import { useEffect, useState } from 'react'
import { Card, Col, Empty, Row, Space, Spin, Statistic, Table, Tag, Typography } from 'antd'
import {
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DollarOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import { Pie } from '@ant-design/plots'
import dayjs from 'dayjs'
import StatCard from '@/components/StatCard'
import StatusTag from '@/components/StatusTag'
import { pricingApi, taskApi, Task } from '@/services/api'

const { Title, Text } = Typography

interface DashboardStats {
  totalTasks: number
  scheduledTasks: number
  runningTasks: number
  completedTasks: number
  failedTasks: number
  cancelledTasks: number
  totalCost: number
  avgCostPerTask: number
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<DashboardStats>({
    totalTasks: 0,
    scheduledTasks: 0,
    runningTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    cancelledTasks: 0,
    totalCost: 0,
    avgCostPerTask: 0,
  })
  const [recentTasks, setRecentTasks] = useState<Task[]>([])

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    setLoading(true)
    try {
      const [tasksRes, costRes] = await Promise.all([
        taskApi.list({ limit: 100 }) as any,
        pricingApi.getTotalCost() as any,
      ])

      const tasks = tasksRes.tasks || []
      const totalTasks = tasksRes.total || tasks.length
      const scheduledTasks = tasks.filter((t: Task) => t.status === 'scheduled').length
      const runningTasks = tasks.filter((t: Task) => t.status === 'running').length
      const completedTasks = tasks.filter((t: Task) => t.status === 'completed').length
      const failedTasks = tasks.filter((t: Task) => t.status === 'failed').length
      const cancelledTasks = tasks.filter((t: Task) => t.status === 'cancelled').length
      const totalCost = costRes?.total_cost || 0
      const runCount = costRes?.run_count || tasks.length

      setRecentTasks(tasks.slice(0, 5))
      setStats({
        totalTasks,
        scheduledTasks,
        runningTasks,
        completedTasks,
        failedTasks,
        cancelledTasks,
        totalCost,
        avgCostPerTask: runCount > 0 ? totalCost / runCount : 0,
      })
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
      render: (name: string, record: Task) => (
        record.status === 'completed'
          ? <a href={`/tasks/${record.run_id}`}>{name || '未命名任务'}</a>
          : (name || '未命名任务')
      ),
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

  const statusPieConfig = {
    appendPadding: 12,
    data: [
      { type: '已定时', value: stats.scheduledTasks },
      { type: '运行中', value: stats.runningTasks },
      { type: '已完成', value: stats.completedTasks },
      { type: '失败', value: stats.failedTasks },
      { type: '已取消', value: stats.cancelledTasks },
      {
        type: '其他',
        value: Math.max(
          0,
          stats.totalTasks - stats.scheduledTasks - stats.runningTasks - stats.completedTasks - stats.failedTasks - stats.cancelledTasks,
        ),
      },
    ],
    angleField: 'value',
    colorField: 'type',
    radius: 0.68,
    innerRadius: 0.46,
    color: ['#2f54eb', '#1677ff', '#52c41a', '#ff4d4f', '#fa8c16', '#d9d9d9'],
    label: {
      type: 'inner',
      offset: '-50%',
      content: '{value}',
      style: {
        textAlign: 'center' as const,
        fontSize: 12,
      },
    },
    legend: {
      position: 'bottom' as const,
    },
    tooltip: {
      customContent: (_title: string, items?: Array<{ data?: { type?: string; value?: number } }>) => {
        const datum = items?.[0]?.data
        if (!datum) {
          return ''
        }
        return `<div style="padding:8px 12px;">${datum.type ?? '状态'}: ${datum.value ?? 0}</div>`
      },
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

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="累计总费用"
            value={stats.totalCost.toFixed(4)}
            prefix={<DollarOutlined />}
            suffix="元"
            color="#cf1322"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="任务总数"
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
            title="平均任务费用"
            value={stats.avgCostPerTask.toFixed(4)}
            prefix="¥"
            color="#722ed1"
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="已完成"
              value={stats.completedTasks}
              valueStyle={{ color: '#52c41a', fontSize: 20 }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="失败"
              value={stats.failedTasks}
              valueStyle={{ color: '#ff4d4f', fontSize: 20 }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12}>
          <Card size="small" style={{ height: '100%' }}>
            <Space size="large">
              <a href="/tasks/create">
                <Tag color="blue" style={{ padding: '8px 16px', fontSize: 14 }}>
                  <RocketOutlined /> 创建任务
                </Tag>
              </a>
              <a href="/pricing">
                <Tag color="red" style={{ padding: '8px 16px', fontSize: 14 }}>
                  <DollarOutlined /> 成本监控
                </Tag>
              </a>
              <a href="/datasets">
                <Tag color="green" style={{ padding: '8px 16px', fontSize: 14 }}>
                  数据管理
                </Tag>
              </a>
            </Space>
          </Card>
        </Col>
      </Row>

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
    </div>
  )
}
