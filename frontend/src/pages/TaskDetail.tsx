import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Descriptions,
  Button,
  Space,
  Tag,
  Typography,
  Progress,
  Statistic,
  Row,
  Col,
  Divider,
  Popconfirm,
  message,
  Spin,
  Empty,
  Tabs,
  Table,
} from 'antd'
import {
  ArrowLeftOutlined,
  StopOutlined,
  RedoOutlined,
  DownloadOutlined,
  LineChartOutlined,
  FileTextOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, Task, TaskProgress, TaskStats } from '@/services/api'

const { Title, Text } = Typography

export default function TaskDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [task, setTask] = useState<Task | null>(null)
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const [stats, setStats] = useState<TaskStats | null>(null)

  useEffect(() => {
    if (id) {
      fetchTaskData(id)
    }
  }, [id])

  // Poll for progress if running
  useEffect(() => {
    if (!id || task?.status !== 'running') return

    const interval = setInterval(() => {
      fetchProgress(id)
    }, 2000)

    return () => clearInterval(interval)
  }, [id, task?.status])

  const fetchTaskData = async (runId: string) => {
    setLoading(true)
    try {
      const taskData = await taskApi.get(runId) as any
      setTask(taskData)

      // Fetch progress
      await fetchProgress(runId)

      // Fetch stats if completed
      if (taskData.status === 'completed') {
        try {
          const statsData = await taskApi.getStats(runId) as any
          setStats(statsData)
        } catch (e) {
          // Ignore
        }
      }
    } catch (error: any) {
      message.error(error.message || '获取任务信息失败')
    } finally {
      setLoading(false)
    }
  }

  const fetchProgress = async (runId: string) => {
    try {
      const progressData = await taskApi.getProgress(runId) as any
      setProgress(progressData)
    } catch (e) {
      // Ignore
    }
  }

  const handleCancel = async () => {
    if (!id) return
    try {
      await taskApi.cancel(id)
      message.success('任务已取消')
      fetchTaskData(id)
    } catch (error: any) {
      message.error(error.message || '取消失败')
    }
  }

  const handleRetry = async () => {
    if (!id) return
    try {
      await taskApi.retry(id)
      message.success('任务已重新启动')
      navigate('/tasks')
    } catch (error: any) {
      message.error(error.message || '重试失败')
    }
  }

  const handleExport = async (format: string) => {
    if (!id) return
    try {
      const result = await taskApi.export(id, format) as any
      if (result.output_path) {
        message.success(`已导出到: ${result.output_path}`)
      }
    } catch (error: any) {
      message.error(error.message || '导出失败')
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!task) {
    return <Empty description="任务不存在" />
  }

  const executorColumns = [
    { title: 'Executor ID', dataIndex: 'executor_id', key: 'executor_id' },
    { title: '请求数', dataIndex: 'total_requests', key: 'total_requests' },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (v: number) => `${v.toFixed(1)}%`,
    },
    {
      title: '平均延迟',
      dataIndex: 'avg_first_resp_time',
      key: 'avg_first_resp_time',
      render: (v: number) => `${v.toFixed(0)}ms`,
    },
    {
      title: 'P95延迟',
      dataIndex: 'p95_first_resp_time',
      key: 'p95_first_resp_time',
      render: (v: number) => `${v.toFixed(0)}ms`,
    },
    {
      title: '费用',
      dataIndex: 'total_cost',
      key: 'total_cost',
      render: (v: number) => `${v.toFixed(4)}`,
    },
  ]

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
      </div>

      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
          <div>
            <Title level={4} style={{ margin: 0, marginBottom: 8 }}>
              {task.task_name || '未命名任务'}
            </Title>
            <Space>
              <Text type="secondary">Run ID: {task.run_id}</Text>
              <StatusTag status={task.status} />
            </Space>
          </div>
          <Space>
            {task.status === 'running' && (
              <Popconfirm title="确定要取消此任务吗？" onConfirm={handleCancel}>
                <Button danger icon={<StopOutlined />}>取消任务</Button>
              </Popconfirm>
            )}
            {task.status === 'failed' && (
              <Button icon={<RedoOutlined />} onClick={handleRetry}>重试</Button>
            )}
            {task.status === 'completed' && (
              <>
                <Button icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
                  导出 CSV
                </Button>
                <Button icon={<FileTextOutlined />} onClick={() => handleExport('html')}>
                  生成报告
                </Button>
                <Button
                  type="primary"
                  icon={<LineChartOutlined />}
                  onClick={() => navigate(`/analysis/${task.run_id}`)}
                >
                  查看分析
                </Button>
              </>
            )}
          </Space>
        </div>

        {/* Progress Section */}
        {task.status === 'running' && progress && (
          <>
            <Card style={{ marginBottom: 24, background: '#fafafa' }}>
              <Row gutter={24}>
                <Col span={16}>
                  <TaskProgressBar
                    percent={progress.progress_percent}
                    successCount={progress.success_count}
                    errorCount={progress.error_count}
                    total={progress.total}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="当前费用"
                    value={progress.current_cost}
                    precision={4}
                    suffix={progress.currency}
                  />
                </Col>
              </Row>
              <Row gutter={24} style={{ marginTop: 16 }}>
                <Col span={8}>
                  <Statistic title="已运行" value={Math.floor(progress.elapsed_seconds)} suffix="秒" />
                </Col>
                <Col span={8}>
                  <Statistic title="预计剩余" value={progress.eta_seconds ? Math.floor(progress.eta_seconds) : '-'} suffix="秒" />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="成功率"
                    value={progress.total > 0 ? (progress.success_count / progress.total * 100) : 0}
                    precision={1}
                    suffix="%"
                  />
                </Col>
              </Row>
            </Card>
          </>
        )}

        {/* Stats Section */}
        {task.status === 'completed' && stats && (
          <>
            <Divider>执行统计</Divider>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic title="总请求数" value={stats.total_requests} />
              </Col>
              <Col span={6}>
                <Statistic
                  title="成功率"
                  value={stats.success_rate}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: stats.success_rate > 95 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="平均延迟"
                  value={stats.avg_first_resp_time}
                  precision={0}
                  suffix="ms"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="总费用"
                  value={stats.total_cost}
                  precision={4}
                  prefix="¥"
                />
              </Col>
            </Row>
          </>
        )}

        <Divider>任务信息</Divider>

        <Descriptions bordered column={2}>
          <Descriptions.Item label="状态">
            <StatusTag status={task.status} />
          </Descriptions.Item>
          <Descriptions.Item label="配置文件">
            {task.config_path || '-'}
          </Descriptions.Item>
          <Descriptions.Item label="创建时间">
            {task.created_at ? dayjs(task.created_at).format('YYYY-MM-DD HH:mm:ss') : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="开始时间">
            {task.started_at ? dayjs(task.started_at).format('YYYY-MM-DD HH:mm:ss') : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="完成时间">
            {task.completed_at ? dayjs(task.completed_at).format('YYYY-MM-DD HH:mm:ss') : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="错误信息">
            {task.error_message ? (
              <Text type="danger">{task.error_message}</Text>
            ) : '-'}
          </Descriptions.Item>
        </Descriptions>
      </Card>
    </div>
  )
}
