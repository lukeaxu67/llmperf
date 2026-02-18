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
  List,
} from 'antd'
import {
  ArrowLeftOutlined,
  StopOutlined,
  RedoOutlined,
  DownloadOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  DollarOutlined,
  SafetyOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, Task, TaskProgress, TaskStats } from '@/services/api'

const { Title, Text } = Typography

interface QuickReport {
  run_id: string
  task_name: string
  completed_at: string | null
  duration_seconds: number
  score: number
  grade: string
  dimension_scores: {
    latency: number
    throughput: number
    success_rate: number
    cost: number
  }
  metrics: {
    total_requests: number
    success_rate: number
    avg_ttft: number
    p95_ttft: number
    avg_tps: number
    total_cost: number
    currency: string
    total_input_tokens: number
    total_output_tokens: number
  }
  executor_summary: Array<{
    id: string
    requests: number
    success_rate: number
    avg_ttft: number
    avg_tps: number
    cost: number
  }>
  alerts: Array<{
    type: string
    severity: 'info' | 'warning' | 'error'
    message: string
    suggestion?: string
  }>
  recommendations: Array<{
    category: 'performance' | 'cost' | 'reliability'
    title: string
    description: string
    impact: 'high' | 'medium' | 'low'
  }>
}

const gradeColors: Record<string, string> = {
  S: '#52c41a',
  A: '#1890ff',
  B: '#722ed1',
  C: '#fa8c16',
  D: '#fa541c',
  F: '#f5222d',
}

const gradeBgColors: Record<string, string> = {
  S: '#f6ffed',
  A: '#e6f7ff',
  B: '#f9f0ff',
  C: '#fff7e6',
  D: '#fff2e8',
  F: '#fff1f0',
}

export default function TaskDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [task, setTask] = useState<Task | null>(null)
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const [stats, setStats] = useState<TaskStats | null>(null)
  const [report, setReport] = useState<QuickReport | null>(null)

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

      // Fetch stats and report if completed
      if (taskData.status === 'completed') {
        try {
          const statsData = await taskApi.getStats(runId) as any
          setStats(statsData)
        } catch (e) {
          // Ignore
        }

        try {
          const reportData = await taskApi.getReport(runId) as any
          setReport(reportData)
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

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <StopOutlined style={{ color: '#f5222d' }} />
      case 'warning':
        return <WarningOutlined style={{ color: '#fa8c16' }} />
      default:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'red'
      case 'medium':
        return 'orange'
      default:
        return 'blue'
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'performance':
        return <ThunderboltOutlined />
      case 'cost':
        return <DollarOutlined />
      case 'reliability':
        return <SafetyOutlined />
      default:
        return <InfoCircleOutlined />
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

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
      </div>

      {/* Quick Report Section - Show when completed */}
      {task.status === 'completed' && report && (
        <Card style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
            <div>
              <Title level={3} style={{ margin: 0, marginBottom: 8 }}>
                测试报告
              </Title>
              <Space>
                <Text type="secondary">{report.task_name}</Text>
                <Text type="secondary">|</Text>
                <Text type="secondary">Run ID: {report.run_id}</Text>
              </Space>
            </div>
            <Space>
              <Button icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
                导出 CSV
              </Button>
              <Button icon={<FileTextOutlined />} onClick={() => handleExport('html')}>
                生成报告
              </Button>
            </Space>
          </div>

          {/* Score Card */}
          <Row gutter={24} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card
                style={{
                  background: gradeBgColors[report.grade] || '#f5f5f5',
                  textAlign: 'center',
                  height: '100%',
                }}
                bodyStyle={{ padding: '24px 16px' }}
              >
                <div
                  style={{
                    fontSize: 64,
                    fontWeight: 'bold',
                    color: gradeColors[report.grade] || '#666',
                    lineHeight: 1,
                    marginBottom: 8,
                  }}
                >
                  {report.grade}
                </div>
                <div style={{ fontSize: 24, color: '#666' }}>
                  {report.score}分
                </div>
                <div style={{ fontSize: 12, color: '#999', marginTop: 8 }}>
                  耗时 {Math.floor(report.duration_seconds)}秒
                </div>
              </Card>
            </Col>
            <Col span={18}>
              <Row gutter={16}>
                <Col span={6}>
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} bodyStyle={{ padding: 16 }}>
                    <DashboardOutlined style={{ fontSize: 24, color: '#1890ff', marginBottom: 8 }} />
                    <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                      {report.dimension_scores.latency}
                    </div>
                    <div style={{ fontSize: 12, color: '#666' }}>延迟评分</div>
                    <Progress
                      percent={report.dimension_scores.latency}
                      showInfo={false}
                      strokeColor="#1890ff"
                      size="small"
                      style={{ marginTop: 8 }}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} bodyStyle={{ padding: 16 }}>
                    <ThunderboltOutlined style={{ fontSize: 24, color: '#52c41a', marginBottom: 8 }} />
                    <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                      {report.dimension_scores.throughput}
                    </div>
                    <div style={{ fontSize: 12, color: '#666' }}>吞吐评分</div>
                    <Progress
                      percent={report.dimension_scores.throughput}
                      showInfo={false}
                      strokeColor="#52c41a"
                      size="small"
                      style={{ marginTop: 8 }}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} bodyStyle={{ padding: 16 }}>
                    <SafetyOutlined style={{ fontSize: 24, color: '#722ed1', marginBottom: 8 }} />
                    <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                      {report.dimension_scores.success_rate}
                    </div>
                    <div style={{ fontSize: 12, color: '#666' }}>成功率评分</div>
                    <Progress
                      percent={report.dimension_scores.success_rate}
                      showInfo={false}
                      strokeColor="#722ed1"
                      size="small"
                      style={{ marginTop: 8 }}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} bodyStyle={{ padding: 16 }}>
                    <DollarOutlined style={{ fontSize: 24, color: '#fa8c16', marginBottom: 8 }} />
                    <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                      {report.dimension_scores.cost}
                    </div>
                    <div style={{ fontSize: 12, color: '#666' }}>成本评分</div>
                    <Progress
                      percent={report.dimension_scores.cost}
                      showInfo={false}
                      strokeColor="#fa8c16"
                      size="small"
                      style={{ marginTop: 8 }}
                    />
                  </Card>
                </Col>
              </Row>
            </Col>
          </Row>

          {/* Core Metrics */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={4}>
              <Statistic
                title="总请求数"
                value={report.metrics.total_requests}
                suffix="次"
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="成功率"
                value={report.metrics.success_rate}
                precision={1}
                suffix="%"
                valueStyle={{ color: report.metrics.success_rate > 95 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="平均TTFT"
                value={report.metrics.avg_ttft}
                precision={0}
                suffix="ms"
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="P95 TTFT"
                value={report.metrics.p95_ttft}
                precision={0}
                suffix="ms"
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="平均TPS"
                value={report.metrics.avg_tps}
                precision={1}
                suffix="tok/s"
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="总费用"
                value={report.metrics.total_cost}
                precision={4}
                prefix={report.metrics.currency === 'CNY' ? '¥' : '$'}
              />
            </Col>
          </Row>

          {/* Alerts and Recommendations */}
          {report.alerts.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <Title level={5}>
                <WarningOutlined style={{ marginRight: 8, color: '#fa8c16' }} />
                发现问题
              </Title>
              <List
                size="small"
                dataSource={report.alerts}
                renderItem={(alert) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={getAlertIcon(alert.severity)}
                      title={alert.message}
                      description={alert.suggestion}
                    />
                  </List.Item>
                )}
              />
            </div>
          )}

          {report.recommendations.length > 0 && (
            <div>
              <Title level={5}>
                <CheckCircleOutlined style={{ marginRight: 8, color: '#52c41a' }} />
                优化建议
              </Title>
              <List
                size="small"
                dataSource={report.recommendations}
                renderItem={(rec) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={getCategoryIcon(rec.category)}
                      title={
                        <Space>
                          <span>{rec.title}</span>
                          <Tag color={getImpactColor(rec.impact)}>
                            {rec.impact === 'high' ? '高影响' : rec.impact === 'medium' ? '中影响' : '低影响'}
                          </Tag>
                        </Space>
                      }
                      description={rec.description}
                    />
                  </List.Item>
                )}
              />
            </div>
          )}
        </Card>
      )}

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
            {task.status === 'completed' && !report && (
              <>
                <Button icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
                  导出 CSV
                </Button>
                <Button icon={<FileTextOutlined />} onClick={() => handleExport('html')}>
                  生成报告
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

        {/* Stats Section - Only show if no report */}
        {task.status === 'completed' && stats && !report && (
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
