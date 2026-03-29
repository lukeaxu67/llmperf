import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Input,
  Modal,
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
  EditOutlined,
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

function executorStatusColor(status: string): string {
  switch (status) {
    case 'completed':
      return 'success'
    case 'running':
      return 'processing'
    case 'paused':
      return 'warning'
    case 'failed':
      return 'error'
    case 'cancelled':
      return 'default'
    case 'blocked':
      return 'orange'
    default:
      return 'default'
  }
}

export default function TaskDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [task, setTask] = useState<Task | null>(null)
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const [report, setReport] = useState<DetailedReport | null>(null)
  const [baselineId, setBaselineId] = useState<string>()
  const [renameOpen, setRenameOpen] = useState(false)
  const [renameValue, setRenameValue] = useState('')
  const [renaming, setRenaming] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [errorsLoading, setErrorsLoading] = useState(false)
  const [taskErrors, setTaskErrors] = useState<any[]>([])

  const loadTaskAndProgress = async (runId: string, silent = false) => {
    if (!silent) {
      setLoading(true)
    }
    try {
      const [taskResult, progressResult] = await Promise.allSettled([
        taskApi.get(runId),
        taskApi.getProgress(runId),
      ])

      if (taskResult.status === 'fulfilled') {
        setTask(taskResult.value as any as Task)
      }
      if (progressResult.status === 'fulfilled') {
        setProgress(progressResult.value as any as TaskProgress)
      } else {
        setProgress(null)
      }
    } catch (error: any) {
      message.error(error.message || '获取任务详情失败')
    } finally {
      setLoading(false)
    }
  }

  const loadReport = async (runId: string, silent = false) => {
    if (!silent) {
      setReportLoading(true)
    }
    try {
      const nextReport = await taskApi.getReport(runId) as any
      setReport(nextReport as DetailedReport)
    } catch {
      setReport(null)
    } finally {
      setReportLoading(false)
    }
  }

  const loadTaskErrors = async (runId: string, silent = false) => {
    if (!silent) {
      setErrorsLoading(true)
    }
    try {
      const response = await taskApi.getErrors(runId, { limit: 20 }) as any
      setTaskErrors(response.errors || [])
    } catch {
      setTaskErrors([])
    } finally {
      setErrorsLoading(false)
    }
  }

  const loadTaskBundle = async (runId: string, silent = false) => {
    await Promise.all([
      loadTaskAndProgress(runId, silent),
      loadReport(runId, true),
      loadTaskErrors(runId, true),
    ])
  }

  useEffect(() => {
    if (id) {
      setReport(null)
      setProgress(null)
      setTaskErrors([])
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
      if (document.visibilityState !== 'visible') {
        return
      }
      loadTaskAndProgress(id, true)
    }, 5000)
    return () => window.clearInterval(timer)
  }, [id, task?.status])

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
      if (document.visibilityState !== 'visible') {
        return
      }
      loadReport(id, true)
      loadTaskErrors(id, true)
    }, 20000)
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

  const handleRename = async () => {
    if (!id) return
    const nextName = renameValue.trim()
    if (!nextName) {
      message.error('任务名称不能为空')
      return
    }
    try {
      setRenaming(true)
      await taskApi.rename(id, nextName)
      setTask((prev) => (prev ? { ...prev, task_name: nextName } : prev))
      setReport((prev) => (prev ? { ...prev, task_name: nextName } : prev))
      setRenameOpen(false)
      message.success('任务名称已更新')
      loadTaskAndProgress(id, true)
    } catch (error: any) {
      message.error(error.message || '更新任务名称失败')
    } finally {
      setRenaming(false)
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
  const canRenameTask = task.status !== 'running' && task.status !== 'paused'

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
        <Button icon={<SyncOutlined />} loading={reportLoading} onClick={() => id && loadReport(id)}>
          刷新分析
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
            <Space align="center" size="small" wrap>
              <Title level={4} style={{ margin: 0 }}>{task.task_name || '未命名任务'}</Title>
              {canRenameTask && (
                <Button
                  type="text"
                  size="small"
                  icon={<EditOutlined />}
                  onClick={() => {
                    setRenameValue(task.task_name || '')
                    setRenameOpen(true)
                  }}
                >
                  修改名称
                </Button>
              )}
            </Space>
            <Space size="middle" style={{ marginTop: 8 }} wrap>
              <StatusTag status={task.status as any} />
              <Text type="secondary">Run ID: {task.run_id}</Text>
              {task.scheduled_at && (
                <Text type="secondary">计划执行时间: {dayjs(task.scheduled_at).format('YYYY-MM-DD HH:mm:ss')}</Text>
              )}
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

      {task.error_message && (
        <Alert
          type={task.status === 'cancelled' ? 'warning' : 'error'}
          showIcon
          style={{ marginBottom: 16 }}
          message={task.status === 'cancelled' ? '任务已取消' : '任务失败原因'}
          description={task.error_message}
        />
      )}

      <Modal
        title="修改任务名称"
        open={renameOpen}
        onOk={handleRename}
        okText="保存"
        cancelText="取消"
        confirmLoading={renaming}
        onCancel={() => {
          if (!renaming) {
            setRenameOpen(false)
          }
        }}
      >
        <Input
          maxLength={120}
          value={renameValue}
          onChange={(event) => setRenameValue(event.target.value)}
          placeholder="输入任务名称"
          onPressEnter={() => {
            void handleRename()
          }}
        />
      </Modal>

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

      {(taskErrors.length > 0 || errorsLoading) && (
        <Card
          title="错误明细"
          extra={(
            <Button size="small" loading={errorsLoading} onClick={() => id && loadTaskErrors(id)}>
              刷新错误
            </Button>
          )}
          style={{ marginBottom: 16 }}
        >
          <Table
            size="small"
            rowKey="id"
            loading={errorsLoading}
            pagination={false}
            scroll={{ x: 900 }}
            dataSource={taskErrors}
            columns={[
              { title: '执行器', dataIndex: 'executor_id', key: 'executor_id', width: 140 },
              { title: 'Provider', dataIndex: 'provider', key: 'provider', width: 120 },
              { title: '模型', dataIndex: 'model', key: 'model', width: 180 },
              { title: '状态码', dataIndex: ['error', 'status_code'], key: 'status_code', width: 90 },
              { title: '错误类型', dataIndex: ['error', 'error_type'], key: 'error_type', width: 150 },
              {
                title: '错误信息',
                key: 'error_message',
                render: (_value: unknown, record: any) => (
                  <Typography.Paragraph style={{ marginBottom: 0 }} ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}>
                    {record.error?.error_message || '-'}
                  </Typography.Paragraph>
                ),
              },
            ]}
          />
        </Card>
      )}

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
                        <Tag color={executorStatusColor(executor.status)}>
                          {executor.status}
                        </Tag>
                      </Space>
                      <Progress
                        percent={Math.round(executor.progress_percent)}
                        size="small"
                        status={executor.status === 'failed' ? 'exception' : 'normal'}
                      />
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
