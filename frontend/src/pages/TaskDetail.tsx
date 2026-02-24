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
  Tabs,
  Alert,
} from 'antd'
import {
  ArrowLeftOutlined,
  StopOutlined,
  RedoOutlined,
  DownloadOutlined,
  FileTextOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  DollarOutlined,
  SafetyOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, Task, TaskProgress, DetailedReport } from '@/services/api'
import {
  LatencyDistributionChart,
  TimeSeriesChart,
  HeatmapChart,
  RadarChart,
  TokenDistributionChart,
  ComparisonTable,
} from '@/components/charts'

const { Title, Text } = Typography

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
  const [report, setReport] = useState<DetailedReport | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  // 确保每次进入页面都重新获取数据
  useEffect(() => {
    // 重置状态，避免使用旧数据
    setReport(null)
    setProgress(null)
    if (id) {
      fetchTaskData(id)
    }
  }, [id])

  useEffect(() => {
    if (!id || (task?.status !== 'running' && task?.status !== 'paused')) return

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

      await fetchProgress(runId)

      // 每次都重新获取报告，不使用缓存
      if (taskData.status === 'completed') {
        try {
          const reportData = await taskApi.getReport(runId) as any
          setReport(reportData)
        } catch (e) {
          console.warn('获取报告失败:', e)
          setReport(null)
        }
      } else {
        // 非完成状态清除报告数据
        setReport(null)
      }
    } catch (error: any) {
      message.error(error.message || '获取任务数据失败')
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

  const handleRetry = async () => {
    if (!id) return
    try {
      await taskApi.retry(id)
      message.success('任务已重启')
      navigate('/tasks')
    } catch (error: any) {
      message.error(error.message || '重试失败')
    }
  }

  const handleExport = async (format: string) => {
    if (!id) return
    try {
      const result = await taskApi.export(id, format as any) as any
      if (result.output_path) {
        message.success(`已导出到: ${result.output_path}`)
      }
    } catch (error: any) {
      message.error(error.message || '导出失败')
    }
  }

  const handleExportResults = async (format: 'jsonl' | 'csv') => {
    if (!id) return
    try {
      const blob = await taskApi.exportResults(id, format) as any
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${id}_results.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      message.success(`已导出${format.toUpperCase()}文件`)
    } catch (error: any) {
      message.error(error.message || '导出失败')
    }
  }

  const handlePause = async () => {
    if (!id) return
    try {
      await taskApi.pause(id)
      message.success('任务已暂停')
      fetchTaskData(id)
    } catch (error: any) {
      message.error(error.message || '暂停失败')
    }
  }

  const handleResume = async () => {
    if (!id) return
    try {
      await taskApi.resume(id)
      message.success('任务已恢复')
      fetchTaskData(id)
    } catch (error: any) {
      message.error(error.message || '恢复失败')
    }
  }

  const handleStop = async () => {
    if (!id) return
    try {
      await taskApi.stop(id)
      message.success('任务已停止')
      fetchTaskData(id)
    } catch (error: any) {
      message.error(error.message || '停止失败')
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

  const isMonitoringTask = task.task_type === 'monitoring'
  // 修复：确保executor_summary存在且有数据，且有多个执行器才显示对比标签
  const hasMultipleExecutors = report?.executor_summary && Array.isArray(report.executor_summary) && report.executor_summary.length > 1

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
      </div>

      {/* Quick Report Header - Show when completed */}
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
                <Text type="secondary">运行ID: {report.run_id}</Text>
                <Tag color={isMonitoringTask ? 'purple' : 'blue'}>
                  {isMonitoringTask ? '24小时监控' : '基准测试'}
                </Tag>
              </Space>
            </div>
            <Space>
              <Button icon={<DownloadOutlined />} onClick={() => handleExportResults('csv')}>
                导出CSV
              </Button>
              <Button icon={<DownloadOutlined />} onClick={() => handleExportResults('jsonl')}>
                导出JSONL
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
                styles={{ body: { padding: '24px 16px' } }}
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
                  {report.score} 分
                </div>
                <div style={{ fontSize: 12, color: '#999', marginTop: 8 }}>
                  耗时: {Math.floor(report.duration_seconds)}秒
                </div>
              </Card>
            </Col>
            <Col span={18}>
              <Row gutter={16}>
                <Col span={6}>
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} styles={{ body: { padding: 16 } }}>
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
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} styles={{ body: { padding: 16 } }}>
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
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} styles={{ body: { padding: 16 } }}>
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
                  <Card size="small" style={{ textAlign: 'center', height: '100%' }} styles={{ body: { padding: 16 } }}>
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
                suffix="毫秒"
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="P95 TTFT"
                value={report.metrics.p95_ttft}
                precision={0}
                suffix="毫秒"
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
                title="总成本"
                value={report.metrics.total_cost}
                precision={4}
                prefix={report.metrics.currency === 'CNY' ? '¥' : '$'}
              />
            </Col>
          </Row>

          {/* Alerts */}
          {report.alerts && report.alerts.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              {report.alerts.map((alert, idx) => (
                <Alert
                  key={idx}
                  type={alert.severity === 'error' ? 'error' : alert.severity === 'warning' ? 'warning' : 'info'}
                  message={alert.message}
                  description={alert.suggestion}
                  icon={getAlertIcon(alert.severity)}
                  closable
                  style={{ marginBottom: 8 }}
                />
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Progress Section for running tasks */}
      {task.status === 'running' && progress && (
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
                title="当前成本"
                value={progress.current_cost}
                precision={4}
                suffix={progress.currency === 'CNY' ? '元' : '美元'}
              />
            </Col>
          </Row>
          <Row gutter={24} style={{ marginTop: 16 }}>
            <Col span={8}>
              <Statistic title="已用时间" value={Math.floor(progress.elapsed_seconds)} suffix="秒" />
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
      )}

      {/* Detail Tabs */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab} items={[
          {
            key: 'overview',
            label: '概览',
            children: (
              <>
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="状态">
                    <StatusTag status={task.status} />
                  </Descriptions.Item>
                  <Descriptions.Item label="任务类型">
                    <Tag color={isMonitoringTask ? 'purple' : 'blue'}>
                      {isMonitoringTask ? '24小时监控' : '基准测试'}
                    </Tag>
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
                  <Descriptions.Item label="错误信息" span={2}>
                    {task.error_message ? (
                      <Text type="danger">{task.error_message}</Text>
                    ) : '-'}
                  </Descriptions.Item>
                </Descriptions>

                {/* Recommendations */}
                {report && report.recommendations && report.recommendations.length > 0 && (
                  <>
                    <Divider>优化建议</Divider>
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
                  </>
                )}
              </>
            ),
          },
          ...(task.status === 'completed' && report ? [
            {
              key: 'latency',
              label: '延迟分析',
              children: (
                <>
                  {/* 多执行器时显示各执行器的延迟汇总 */}
                  {report.executor_summary && report.executor_summary.length > 1 && (
                    <Row gutter={16} style={{ marginBottom: 16 }}>
                      {report.executor_summary.map(exec => (
                        <Col span={Math.floor(24 / report.executor_summary.length)} key={exec.id}>
                          <Card size="small" title={exec.id}>
                            <Statistic
                              title="平均TTFT"
                              value={exec.avg_ttft || 0}
                              precision={2}
                              suffix="毫秒"
                            />
                            <Statistic
                              title="TPS"
                              value={exec.avg_tps || 0}
                              precision={1}
                              suffix="tok/s"
                              style={{ marginTop: 8 }}
                            />
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  )}

                  {/* 总体延迟分位数统计 */}
                  <Row gutter={16} style={{ marginBottom: 16 }}>
                    <Col span={6}>
                      <Card>
                        <Statistic title="P50" value={report.metrics.p50_ttft} precision={2} suffix="毫秒" />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card>
                        <Statistic title="P90" value={report.metrics.p90_ttft} precision={2} suffix="毫秒" />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card>
                        <Statistic title="P95" value={report.metrics.p95_ttft} precision={2} suffix="毫秒" />
                      </Card>
                    </Col>
                    <Col span={6}>
                      <Card>
                        <Statistic title="P99" value={report.metrics.p99_ttft} precision={2} suffix="毫秒" />
                      </Card>
                    </Col>
                  </Row>

                  {/* 延迟分布图表 - 支持分执行器显示 */}
                  {report.latency_analysis ? (
                    <>
                      {/* 如果有分执行器数据，显示各执行器延迟分布 */}
                      {report.latency_analysis.by_executor && Object.keys(report.latency_analysis.by_executor).length > 0 ? (
                        <>
                          {Object.entries(report.latency_analysis.by_executor).map(([executorId, data]) => (
                            <Card
                              key={executorId}
                              size="small"
                              title={`${executorId} 延迟分布`}
                              style={{ marginBottom: 16 }}
                            >
                              <LatencyDistributionChart data={data} />
                            </Card>
                          ))}
                        </>
                      ) : (
                        <LatencyDistributionChart data={report.latency_analysis} />
                      )}
                    </>
                  ) : (
                    <Empty description="延迟分析数据不可用" />
                  )}
                </>
              ),
            },
            {
              key: 'throughput',
              label: '吞吐分析',
              children: (
                <>
                  {/* 多执行器时显示各执行器的吞吐汇总 */}
                  {report.executor_summary && report.executor_summary.length > 1 && (
                    <Row gutter={16} style={{ marginBottom: 16 }}>
                      {report.executor_summary.map(exec => (
                        <Col span={Math.floor(24 / report.executor_summary.length)} key={exec.id}>
                          <Card size="small" title={exec.id}>
                            <Statistic
                              title="平均TPS"
                              value={exec.avg_tps || 0}
                              precision={1}
                              suffix="tok/s"
                            />
                            <Statistic
                              title="平均输出Token"
                              value={exec.avg_output_tokens || 0}
                              precision={0}
                              suffix="tokens"
                              style={{ marginTop: 8 }}
                            />
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  )}

                  {/* Token分布图表 - 支持分执行器显示 */}
                  {report.token_analysis ? (
                    <>
                      {report.token_analysis.by_executor && Object.keys(report.token_analysis.by_executor).length > 0 ? (
                        <>
                          {Object.entries(report.token_analysis.by_executor).map(([executorId, data]) => (
                            <Card
                              key={executorId}
                              size="small"
                              title={`${executorId} Token分布`}
                              style={{ marginBottom: 16 }}
                            >
                              <TokenDistributionChart data={[{
                                values: data.values
                              }]} />
                              <Row gutter={16} style={{ marginTop: 12 }}>
                                <Col span={12}>
                                  <Statistic
                                    title="平均Token数"
                                    value={data.avg}
                                    precision={0}
                                  />
                                </Col>
                                <Col span={12}>
                                  <Statistic
                                    title="P90"
                                    value={data.p90}
                                    precision={0}
                                  />
                                </Col>
                              </Row>
                            </Card>
                          ))}
                        </>
                      ) : (
                        <TokenDistributionChart data={[{
                          values: report.token_analysis.output_tokens
                        }]} />
                      )}
                    </>
                  ) : (
                    <Empty description="Token分布数据不可用" />
                  )}

                  {report.time_series && (
                    <div style={{ marginTop: 16 }}>
                      <TimeSeriesChart
                        title="Token速率变化"
                        data={[
                          {
                            name: 'TPS (每秒Token数)',
                            data: report.time_series.timeline.map((t, i) => ({
                              timestamp: t,
                              value: report.time_series!.tps[i]
                            })),
                            unit: ' tok/s',
                            color: '#52c41a'
                          }
                        ]}
                      />
                    </div>
                  )}
                </>
              ),
            },
            ...(isMonitoringTask && report.time_frame_analysis ? [{
              key: 'timeframe',
              label: '时段分析',
              children: (
                <>
                  <Alert
                    message="24小时时段分析"
                    description="对比不同时间段的表现，识别性能模式和高峰时段"
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />

                  <HeatmapChart
                    data={report.time_frame_analysis.time_frames.flatMap(tf =>
                      ['TTFT', 'Total Time', 'TPS', 'Success Rate'].map(metric => ({
                        period: tf.name,
                        metric,
                        value: tf.metrics[
                          metric === 'TTFT' ? 'avg_ttft' :
                          metric === 'Total Time' ? 'avg_total_time' :
                          metric === 'TPS' ? 'avg_tps' : 'success_rate'
                        ]
                      }))
                    )}
                    periods={report.time_frame_analysis.time_frames.map(tf => tf.name)}
                  />

                  <Card title="分析结论" style={{ marginTop: 16 }}>
                    <List
                      dataSource={report.time_frame_analysis.insights}
                      renderItem={(insight, idx) => (
                        <List.Item key={idx}>
                          <Text>{insight}</Text>
                        </List.Item>
                      )}
                    />
                  </Card>

                  {report.time_series && (
                    <div style={{ marginTop: 16 }}>
                      <TimeSeriesChart
                        title="指标随时间变化"
                        data={[
                          {
                            name: 'TTFT',
                            data: report.time_series.timeline.map((t, i) => ({
                              timestamp: t,
                              value: report.time_series!.ttft[i]
                            })),
                            unit: '毫秒'
                          },
                          {
                            name: 'TPS',
                            data: report.time_series.timeline.map((t, i) => ({
                              timestamp: t,
                              value: report.time_series!.tps[i]
                            })),
                            unit: 'tok/s'
                          }
                        ]}
                        height={500}
                      />
                    </div>
                  )}
                </>
              ),
            }] : []),
            ...(hasMultipleExecutors ? [{
              key: 'comparison',
              label: '模型对比',
              children: (
                <>
                  {(() => {
                    // 安全计算雷达图数据，处理除零情况
                    const execSummary = report.executor_summary || []
                    const maxTtft = Math.max(...execSummary.map(e => e.avg_ttft || 0), 1)
                    const maxTps = Math.max(...execSummary.map(e => e.avg_tps || 0), 1)
                    const maxCost = Math.max(...execSummary.map(e => e.cost || 0), 1)

                    const radarData = execSummary.map(exec => ({
                      name: exec.id,
                      value: [
                        maxTtft > 0 ? 100 - ((exec.avg_ttft || 0) / maxTtft * 100) : 50,
                        maxTps > 0 ? ((exec.avg_tps || 0) / maxTps * 100) : 50,
                        exec.success_rate || 0,
                        maxCost > 0 ? 100 - ((exec.cost || 0) / maxCost * 100) : 50,
                        exec.success_rate || 0
                      ]
                    }))

                    return (
                      <RadarChart
                        data={radarData}
                        indicators={['延迟(反向)', '吞吐量', '成功率', '成本效率', '稳定性']}
                      />
                    )
                  })()}

                  <ComparisonTable
                    data={report.executor_summary.map(exec => ({
                      model: exec.id,
                      requests: exec.requests || 0,
                      successRate: exec.success_rate || 0,
                      avgTtft: exec.avg_ttft || 0,
                      p95Ttft: exec.p95_ttft || 0,
                      avgTps: exec.avg_tps || 0,
                      totalCost: exec.cost || 0,
                      currency: report.metrics.currency,
                      avgTokens: exec.avg_output_tokens || 0
                    }))}
                    baselineModel={report.executor_summary[0]?.id}
                  />
                </>
              ),
            }] : []),
            {
              key: 'cost',
              label: '成本分析',
              children: (
                <>
                  <Row gutter={16} style={{ marginBottom: 16 }}>
                    <Col span={12}>
                      <Card>
                        <Statistic
                          title="总成本"
                          value={report.metrics.total_cost}
                          precision={4}
                          prefix={report.metrics.currency === 'CNY' ? '¥' : '$'}
                        />
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card>
                        <Statistic
                          title="每千次请求成本"
                          value={report.metrics.total_requests > 0
                            ? (report.metrics.total_cost / report.metrics.total_requests) * 1000
                            : 0}
                          precision={4}
                          prefix={report.metrics.currency === 'CNY' ? '¥' : '$'}
                        />
                      </Card>
                    </Col>
                  </Row>

                  {/* 按执行器统计成本 - 使用executor_summary作为数据源 */}
                  {report.executor_summary && report.executor_summary.length > 0 ? (
                    <Card title="按模型统计成本" style={{ marginBottom: 16 }}>
                      <List
                        dataSource={report.executor_summary.map(exec => ({
                          executor: exec.id,
                          cost: exec.cost || 0,
                          request_count: exec.requests || 0,
                          avg_cost_per_request: exec.requests > 0 ? (exec.cost || 0) / exec.requests : 0
                        }))}
                        renderItem={(item) => (
                          <List.Item>
                            <List.Item.Meta
                              title={<Text strong>{item.executor}</Text>}
                              description={
                                <Space direction="vertical" size="small">
                                  <Text>总计: {report.metrics.currency === 'CNY' ? '¥' : '$'}{item.cost.toFixed(4)}</Text>
                                  <Text type="secondary">
                                    平均每次请求: {report.metrics.currency === 'CNY' ? '¥' : '$'}{item.avg_cost_per_request.toFixed(6)}
                                  </Text>
                                </Space>
                              }
                            />
                            <Statistic
                              value={item.request_count}
                              suffix="次"
                              style={{ textAlign: 'right' }}
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  ) : (
                    <Empty description="成本分析数据不可用" style={{ marginBottom: 16 }} />
                  )}

                  {/* 成本趋势图 - 如果有cost_analysis数据则显示 */}
                  {report.cost_analysis?.cost_trend ? (
                    <div style={{ marginTop: 16 }}>
                      <TimeSeriesChart
                        title="成本随时间累积"
                        data={[
                          {
                            name: '累积成本',
                            data: report.cost_analysis.cost_trend.map(d => ({
                              timestamp: d.time,
                              value: d.accumulated_cost
                            })),
                            unit: ` ${report.metrics.currency}`
                          }
                        ]}
                      />
                    </div>
                  ) : null}
                </>
              ),
            },
            ...(report.error_analysis ? [{
              key: 'errors',
              label: '错误分析',
              children: (
                <>
                  <Row gutter={16} style={{ marginBottom: 16 }}>
                    <Col span={12}>
                      <Card>
                        <Statistic
                          title="总错误数"
                          value={report.error_analysis.total_errors}
                          valueStyle={{ color: '#cf1322' }}
                        />
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card>
                        <Statistic
                          title="错误率"
                          value={report.error_analysis.error_rate}
                          precision={2}
                          suffix="%"
                          valueStyle={{ color: report.error_analysis.error_rate > 5 ? '#cf1322' : '#3f8600' }}
                        />
                      </Card>
                    </Col>
                  </Row>

                  <Card title="按类型统计错误">
                    <List
                      dataSource={Object.entries(report.error_analysis.by_type)}
                      renderItem={([type, data]) => (
                        <List.Item>
                          <List.Item.Meta
                            title={<Text strong>{type}</Text>}
                            description={
                              <Space direction="vertical" size="small">
                                <Text>数量: {data.count}</Text>
                                <Text type="secondary">比例: {data.rate.toFixed(2)}%</Text>
                                {data.examples && data.examples.length > 0 && (
                                  <Text type="secondary" ellipsis>
                                    示例: {data.examples[0]}
                                  </Text>
                                )}
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  </Card>

                  {report.error_analysis.by_time && report.error_analysis.by_time.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <TimeSeriesChart
                        title="错误率随时间变化"
                        data={[
                          {
                            name: '错误率',
                            data: report.error_analysis.by_time.map(d => ({
                              timestamp: d.time,
                              value: d.error_rate
                            })),
                            unit: '%',
                            color: '#f5222d'
                          }
                        ]}
                      />
                    </div>
                  )}
                </>
              ),
            }] : []),
          ] : []),
        ]} />

        {/* Action buttons */}
        <Divider />
        <Space>
          {task.status === 'running' && (
            <>
              <Button icon={<PauseCircleOutlined />} onClick={handlePause}>
                暂停
              </Button>
              <Popconfirm title="确定要停止此任务吗?" onConfirm={handleStop}>
                <Button danger icon={<StopOutlined />}>停止</Button>
              </Popconfirm>
            </>
          )}
          {task.status === 'paused' && (
            <>
              <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleResume}>
                恢复
              </Button>
              <Popconfirm title="确定要停止此任务吗?" onConfirm={handleStop}>
                <Button danger icon={<StopOutlined />}>停止</Button>
              </Popconfirm>
            </>
          )}
          {task.status === 'failed' && (
            <Button icon={<RedoOutlined />} onClick={handleRetry}>重试</Button>
          )}
          {task.status === 'completed' && !report && (
            <>
              <Button icon={<DownloadOutlined />} onClick={() => handleExportResults('csv')}>
                导出CSV
              </Button>
              <Button icon={<FileTextOutlined />} onClick={() => handleExport('html')}>
                生成报告
              </Button>
            </>
          )}
        </Space>
      </Card>
    </div>
  )
}
