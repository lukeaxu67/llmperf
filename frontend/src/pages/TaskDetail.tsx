import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Progress,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  FileTextOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  StopOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import ExecutorTopologyGraph from '@/components/ExecutorTopologyGraph'
import { DetailedReport, ExecutorProgress, taskApi, Task, TaskProgress } from '@/services/api'
import { mergeTopologyProgress } from '@/utils/executorTopology'

const { Title, Text } = Typography

function formatDelta(current: number, baseline?: number, reverse = false): string {
  if (baseline === undefined || baseline === null || baseline === 0) {
    return '-'
  }
  const delta = ((current - baseline) / baseline) * 100
  const normalized = reverse ? -delta : delta
  const sign = normalized > 0 ? '+' : ''
  return `${sign}${normalized.toFixed(1)}%`
}

function metricColor(value: number, baseline?: number, reverse = false): string | undefined {
  if (baseline === undefined || baseline === null) {
    return undefined
  }
  if (reverse) {
    return value <= baseline ? '#52c41a' : '#ff4d4f'
  }
  return value >= baseline ? '#52c41a' : '#ff4d4f'
}

export default function TaskDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [task, setTask] = useState<Task | null>(null)
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const [report, setReport] = useState<DetailedReport | null>(null)
  const [baselineId, setBaselineId] = useState<string>()

  const loadTaskBundle = async (runId: string, silent = false) => {
    if (!silent) {
      setLoading(true)
    }
    try {
      const [taskResult, progressResult, reportResult] = await Promise.allSettled([
        taskApi.get(runId),
        taskApi.getProgress(runId),
        taskApi.getReport(runId),
      ])

      if (taskResult.status === 'fulfilled') {
        setTask(taskResult.value as any as Task)
      }
      if (progressResult.status === 'fulfilled') {
        setProgress(progressResult.value as any as TaskProgress)
      } else {
        setProgress(null)
      }
      if (reportResult.status === 'fulfilled') {
        setReport(reportResult.value as any as DetailedReport)
      } else {
        setReport(null)
      }
    } catch (error: any) {
      message.error(error.message || '获取任务详情失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (id) {
      setReport(null)
      setProgress(null)
      loadTaskBundle(id)
    }
  }, [id])

  useEffect(() => {
    if (!id) {
      return
    }
    if (
      task?.status !== 'scheduled'
      && task?.status !== 'pending'
      && task?.status !== 'running'
      && task?.status !== 'paused'
    ) {
      return
    }
    const timer = window.setInterval(() => {
      loadTaskBundle(id, true)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [id, task?.status])

  useEffect(() => {
    if (!baselineId && report?.executor_summary?.length) {
      setBaselineId(report.executor_summary[0].id)
    }
  }, [baselineId, report])

  const handleExport = async (format: 'jsonl' | 'csv' | 'html') => {
    if (!id) return
    try {
      if (format === 'html') {
        const result = await taskApi.export(id, 'html') as any
        if (result.output_path) {
          message.success(`报告已生成到 ${result.output_path}`)
        }
        return
      }
      const blob = await taskApi.exportResults(id, format) as any
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${id}_results.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      message.success(`已导出 ${format.toUpperCase()} 结果`)
    } catch (error: any) {
      message.error(error.message || '导出失败')
    }
  }

  const handlePause = async () => {
    if (!id) return
    try {
      await taskApi.pause(id)
      message.success('任务已暂停')
      loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '暂停失败')
    }
  }

  const handleResume = async () => {
    if (!id) return
    try {
      await taskApi.resume(id)
      message.success('任务已恢复')
      loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '恢复失败')
    }
  }

  const handleStop = async () => {
    if (!id) return
    try {
      await taskApi.stop(id)
      message.success('任务已停止')
      loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '停止失败')
    }
  }

  const handleStartNow = async () => {
    if (!id) return
    try {
      await taskApi.start(id)
      message.success('任务已开始执行')
      loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '立即启动失败')
    }
  }

  const handleRecover = async () => {
    if (!id) return
    try {
      await taskApi.recover(id)
      message.success('任务已恢复执行，将继续补齐未完成部分')
      loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '恢复执行失败')
    }
  }

  const executorItems = useMemo<ExecutorProgress[]>(
    () => report?.executor_summary || progress?.executors || [],
    [progress, report],
  )

  const baseline = executorItems.find((item) => item.id === baselineId) || executorItems[0]
  const topology = useMemo(
    () => mergeTopologyProgress(report?.topology || progress?.topology, executorItems),
    [executorItems, progress?.topology, report?.topology],
  )

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

  const tableData = executorItems.map((item) => ({
    key: item.id,
    ...item,
  }))

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
        <Button icon={<DownloadOutlined />} onClick={() => handleExport('csv')}>
          导出 CSV
        </Button>
        <Button icon={<DownloadOutlined />} onClick={() => handleExport('jsonl')}>
          导出 JSONL
        </Button>
        <Button icon={<FileTextOutlined />} onClick={() => handleExport('html')}>
          生成 HTML 报告
        </Button>
        {(task.status === 'scheduled' || task.status === 'pending') && (
          <>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleStartNow}>立即启动</Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>取消任务</Button>
          </>
        )}
        {task.status === 'running' && (
          <>
            <Button icon={<PauseCircleOutlined />} onClick={handlePause}>暂停</Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>停止</Button>
          </>
        )}
        {task.status === 'paused' && (
          <>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleResume}>恢复</Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>停止</Button>
          </>
        )}
        {(task.status === 'failed' || task.status === 'cancelled') && (
          <Button type="primary" icon={<SyncOutlined />} onClick={handleRecover}>恢复执行</Button>
        )}
      </Space>

      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[24, 16]} align="middle">
          <Col span={16}>
            <Title level={4} style={{ margin: 0 }}>{task.task_name || '未命名任务'}</Title>
            <Space size="middle" style={{ marginTop: 8 }} wrap>
              <StatusTag status={task.status as any} />
              <Text type="secondary">Run ID: {task.run_id}</Text>
              {report?.generated_at && (
                <Text type="secondary">报告生成时间: {dayjs(report.generated_at).format('YYYY-MM-DD HH:mm:ss')}</Text>
              )}
            </Space>
          </Col>
          <Col span={8}>
            {progress && (
              <TaskProgressBar
                percent={progress.progress_percent}
                successCount={progress.success_count}
                errorCount={progress.error_count}
                total={progress.total}
              />
            )}
          </Col>
        </Row>
      </Card>

      {(task.status === 'running' || task.status === 'paused' || report?.is_partial) && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="报告基于当前已落库数据实时重算"
          description="任务未跑完时也会展示阶段性指标和执行器结论，页面每次请求都会重新计算，不使用缓存。"
        />
      )}

      <Card title="任务概览" style={{ marginBottom: 16 }}>
        <Descriptions column={4} size="small">
          <Descriptions.Item label="创建时间">{dayjs(task.created_at).format('YYYY-MM-DD HH:mm:ss')}</Descriptions.Item>
          <Descriptions.Item label="开始时间">{task.started_at ? dayjs(task.started_at).format('YYYY-MM-DD HH:mm:ss') : '-'}</Descriptions.Item>
          <Descriptions.Item label="完成时间">{task.completed_at ? dayjs(task.completed_at).format('YYYY-MM-DD HH:mm:ss') : '-'}</Descriptions.Item>
          <Descriptions.Item label="状态">{task.status}</Descriptions.Item>
        </Descriptions>

        <Row gutter={[16, 16]} style={{ marginTop: 8 }}>
          <Col span={4}>
            <Statistic title="总请求" value={report?.metrics.total_requests || progress?.completed || 0} />
          </Col>
          <Col span={4}>
            <Statistic title="成功率" value={report?.metrics.success_rate || 0} precision={1} suffix="%" />
          </Col>
          <Col span={4}>
            <Statistic title="总成本" value={report?.metrics.total_cost || progress?.current_cost || 0} precision={4} suffix={report?.metrics.currency || progress?.currency || 'CNY'} />
          </Col>
          <Col span={4}>
            <Statistic title="平均输入 tokens" value={report?.metrics.avg_input_tokens || 0} precision={1} />
          </Col>
          <Col span={4}>
            <Statistic title="平均输出 tokens" value={report?.metrics.avg_output_tokens || 0} precision={1} />
          </Col>
          <Col span={4}>
            <Statistic title="平均首响" value={report?.metrics.avg_ttft || 0} precision={0} suffix="ms" />
          </Col>
        </Row>
      </Card>

      <Card title="执行拓扑与进度" style={{ marginBottom: 16 }}>
        {topology.nodes.length > 0 ? (
          <>
            <ExecutorTopologyGraph topology={topology} height={360} />
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              {executorItems.map((executor) => (
                <Col span={8} key={executor.id}>
                  <Card size="small">
                    <Space direction="vertical" style={{ width: '100%' }} size={8}>
                      <Space wrap>
                        <Text strong>{executor.name}</Text>
                        <Tag color={executor.status === 'completed' ? 'success' : executor.status === 'running' ? 'processing' : 'default'}>
                          {executor.status}
                        </Tag>
                      </Space>
                      <Progress percent={Math.round(executor.progress_percent)} size="small" />
                      <Text type="secondary">
                        {executor.completed}/{executor.total}，成功 {executor.success_count}，失败 {executor.error_count}
                      </Text>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </>
        ) : (
          <Empty description="暂无拓扑信息" />
        )}
      </Card>

      <Card
        title="执行器结论与对比"
        extra={(
          <Space>
            <Text type="secondary">对比基准</Text>
            <Select
              style={{ width: 240 }}
              value={baseline?.id}
              onChange={setBaselineId}
              options={executorItems.map((item) => ({ label: item.name, value: item.id }))}
            />
          </Space>
        )}
        style={{ marginBottom: 16 }}
      >
        {executorItems.length === 0 ? (
          <Empty description="暂无执行器结果" />
        ) : (
          <Table
            size="small"
            dataSource={tableData}
            pagination={false}
            scroll={{ x: 1500 }}
            columns={[
              {
                title: '执行器',
                dataIndex: 'name',
                fixed: 'left',
                width: 220,
                render: (_value, record: ExecutorProgress) => (
                  <Space direction="vertical" size={4}>
                    <Space wrap>
                      <Text strong>{record.name}</Text>
                      <Tag color={record.id === baseline?.id ? 'blue' : 'default'}>
                        {record.id === baseline?.id ? '基准' : record.status}
                      </Tag>
                    </Space>
                    <Text type="secondary">{record.model || record.provider}</Text>
                    <Text>{record.conclusion}</Text>
                  </Space>
                ),
              },
              {
                title: '完成度',
                dataIndex: 'progress_percent',
                width: 140,
                render: (value: number, record: ExecutorProgress) => (
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Progress percent={Math.round(value)} size="small" />
                    <Text type="secondary">{record.completed}/{record.total}</Text>
                  </Space>
                ),
              },
              { title: '成功率', dataIndex: 'success_rate', width: 100, render: (value: number) => `${value.toFixed(1)}%` },
              { title: '输入 tokens', dataIndex: 'avg_input_tokens', width: 120, render: (value: number) => value.toFixed(1) },
              { title: '输出 tokens', dataIndex: 'avg_output_tokens', width: 120, render: (value: number) => value.toFixed(1) },
              {
                title: '首响(ms)',
                dataIndex: 'avg_ttft',
                width: 120,
                render: (value: number) => (
                  <Text style={{ color: metricColor(value, baseline?.avg_ttft, true) }}>
                    {value.toFixed(0)}
                  </Text>
                ),
              },
              {
                title: '尾响(ms)',
                dataIndex: 'avg_total_time',
                width: 120,
                render: (value: number) => value.toFixed(0),
              },
              {
                title: 'token速率(不带首响)',
                dataIndex: 'avg_token_per_second',
                width: 160,
                render: (value: number) => value.toFixed(2),
              },
              {
                title: 'token速率(带首响)',
                dataIndex: 'avg_token_per_second_with_calltime',
                width: 160,
                render: (value: number) => (
                  <Text style={{ color: metricColor(value, baseline?.avg_token_per_second_with_calltime) }}>
                    {value.toFixed(2)}
                  </Text>
                ),
              },
              { title: '成本', dataIndex: 'cost', width: 110, render: (value: number) => value.toFixed(4) },
              { title: '综合分', dataIndex: 'score', width: 90 },
              {
                title: '相对基准 TTFT',
                key: 'delta_ttft',
                width: 120,
                render: (_value, record: ExecutorProgress) => formatDelta(record.avg_ttft, baseline?.avg_ttft, true),
              },
              {
                title: '相对基准 TPS',
                key: 'delta_tps',
                width: 120,
                render: (_value, record: ExecutorProgress) =>
                  formatDelta(record.avg_token_per_second_with_calltime, baseline?.avg_token_per_second_with_calltime),
              },
            ]}
          />
        )}
      </Card>

      {report?.alerts?.length ? (
        <Card title="告警与建议">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            {report.alerts.map((alert, index) => (
              <Alert
                key={`${alert.type}-${index}`}
                type={alert.severity === 'error' ? 'error' : alert.severity === 'warning' ? 'warning' : 'info'}
                showIcon
                message={alert.message}
                description={alert.suggestion}
              />
            ))}
          </Space>
        </Card>
      ) : null}
    </div>
  )
}
