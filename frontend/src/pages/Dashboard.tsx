import { useEffect, useState } from 'react'
import { Card, Col, Empty, Row, Space, Spin, Table, Tag, Typography } from 'antd'
import {
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DollarOutlined,
  ClockCircleOutlined,
  ArrowRightOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
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
          ? <a href={`/tasks/${record.run_id}`} style={{ fontWeight: 500 }}>{name || '未命名任务'}</a>
          : <Text>{name || '未命名任务'}</Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => <StatusTag status={status as any} />,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      render: (time: string) => <Text type="secondary">{dayjs(time).format('MM-DD HH:mm')}</Text>,
    },
  ]

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <Title level={3} style={{ margin: 0, marginBottom: 4 }}>
            系统概览
          </Title>
          <Text type="secondary">查看任务执行状态和成本统计</Text>
        </div>
      </div>

      {/* Main Stats */}
      <Row gutter={[16, 16]} className="stagger-animation">
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="累计总费用"
            value={stats.totalCost.toFixed(4)}
            prefix={<DollarOutlined />}
            suffix="元"
            variant="error"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="任务总数"
            value={stats.totalTasks}
            prefix={<RocketOutlined />}
            variant="default"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="运行中"
            value={stats.runningTasks}
            prefix={<ClockCircleOutlined />}
            variant="warning"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="平均任务费用"
            value={stats.avgCostPerTask.toFixed(4)}
            prefix={<ThunderboltOutlined />}
            color="#722ed1"
          />
        </Col>
      </Row>

      {/* Secondary Stats */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={12} sm={6}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
            }}
          >
            <Space>
              <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 18 }} />
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>已完成</Text>
                <div style={{ fontWeight: 600, color: '#52c41a', fontSize: 20 }}>
                  {stats.completedTasks}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
            }}
          >
            <Space>
              <CloseCircleOutlined style={{ color: '#ff4d4f', fontSize: 18 }} />
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>失败</Text>
                <div style={{ fontWeight: 600, color: '#ff4d4f', fontSize: 20 }}>
                  {stats.failedTasks}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
        <Col xs={24} sm={12}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
              height: '100%',
            }}
          >
            <Space size="middle">
              <a href="/tasks/create">
                <Tag
                  color="blue"
                  style={{
                    padding: '8px 16px',
                    fontSize: 13,
                    borderRadius: 6,
                    cursor: 'pointer',
                  }}
                >
                  <RocketOutlined style={{ marginRight: 6 }} />
                  创建任务
                  <ArrowRightOutlined style={{ marginLeft: 6 }} />
                </Tag>
              </a>
              <a href="/pricing">
                <Tag
                  color="red"
                  style={{
                    padding: '8px 16px',
                    fontSize: 13,
                    borderRadius: 6,
                    cursor: 'pointer',
                  }}
                >
                  <DollarOutlined style={{ marginRight: 6 }} />
                  成本监控
                  <ArrowRightOutlined style={{ marginLeft: 6 }} />
                </Tag>
              </a>
              <a href="/datasets">
                <Tag
                  color="green"
                  style={{
                    padding: '8px 16px',
                    fontSize: 13,
                    borderRadius: 6,
                    cursor: 'pointer',
                  }}
                >
                  <RocketOutlined style={{ marginRight: 6 }} />
                  数据管理
                  <ArrowRightOutlined style={{ marginLeft: 6 }} />
                </Tag>
              </a>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Recent Tasks */}
      <Card
        title={<Text strong>最近任务</Text>}
        extra={<a href="/tasks" style={{ fontSize: 13 }}>查看全部 <ArrowRightOutlined /></a>}
        style={{
          marginTop: 24,
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        {recentTasks.length > 0 ? (
          <Table
            columns={taskColumns}
            dataSource={recentTasks}
            rowKey="run_id"
            pagination={false}
            size="small"
            showHeader={false}
          />
        ) : (
          <Empty description="暂无任务" style={{ padding: 40 }} />
        )}
      </Card>
    </div>
  )
}
