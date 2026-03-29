import axios from 'axios'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  InputNumber,
  message,
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
import ExecutorTopologyGraph from '@/components/ExecutorTopologyGraph'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import {
  DetailedReport,
  ExecutorProgress,
  Task,
  TaskProgress,
  TaskRuntimeConfig,
  taskApi,
} from '@/services/api'
import { mergeTopologyProgress } from '@/utils/executorTopology'

const { Paragraph, Text, Title } = Typography

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
  const [taskErrors, setTaskErrors] = useState<any[]>([])
  const [reportLoading, setReportLoading] = useState(false)
  const [errorsLoading, setErrorsLoading] = useState(false)
  const [baselineId, setBaselineId] = useState<string>()

  const [renameOpen, setRenameOpen] = useState(false)
  const [renameValue, setRenameValue] = useState('')
  const [renaming, setRenaming] = useState(false)

  const [runtimeConfigOpen, setRuntimeConfigOpen] = useState(false)
  const [runtimeConfigLoading, setRuntimeConfigLoading] = useState(false)
  const [runtimeConfigSaving, setRuntimeConfigSaving] = useState(false)
  const [runtimeConfig, setRuntimeConfig] = useState<TaskRuntimeConfig | null>(null)
  const [runtimeForm] = Form.useForm()
  const requestControllersRef = useRef<{
    bundle?: AbortController
    report?: AbortController
    errors?: AbortController
  }>({})
  const requestVersionRef = useRef({
    bundle: 0,
    report: 0,
    errors: 0,
  })

  const isCanceledRequest = (error: unknown): boolean =>
    axios.isCancel(error) || (typeof error === 'object' && error !== null && (error as any).code === 'ERR_CANCELED')

  const startRequest = (key: 'bundle' | 'report' | 'errors') => {
    requestControllersRef.current[key]?.abort()
    const controller = new AbortController()
    requestControllersRef.current[key] = controller
    requestVersionRef.current[key] += 1
    return {
      controller,
      version: requestVersionRef.current[key],
    }
  }

  const loadTaskAndProgress = async (runId: string, silent = false) => {
    const { controller, version } = startRequest('bundle')
    if (!silent) {
      setLoading(true)
    }
    try {
      const [taskResult, progressResult] = await Promise.allSettled([
        taskApi.get(runId, { signal: controller.signal }),
        taskApi.getProgress(runId, { signal: controller.signal }),
      ])

      if (version !== requestVersionRef.current.bundle) {
        return
      }

      if (taskResult.status === 'fulfilled') {
        setTask(taskResult.value as any as Task)
      }
      if (progressResult.status === 'fulfilled') {
        setProgress(progressResult.value as any as TaskProgress)
      } else {
        setProgress(null)
      }
    } catch (error: any) {
      if (isCanceledRequest(error)) {
        return
      }
      message.error(error.message || '获取任务详情失败')
    } finally {
      if (version === requestVersionRef.current.bundle) {
        setLoading(false)
      }
    }
  }

  const loadReport = async (runId: string, silent = false) => {
    const { controller, version } = startRequest('report')
    if (!silent) {
      setReportLoading(true)
    }
    try {
      const nextReport = await taskApi.getReport(runId, { signal: controller.signal }) as any
      if (version === requestVersionRef.current.report) {
        setReport(nextReport as DetailedReport)
      }
    } catch (error) {
      if (isCanceledRequest(error)) {
        return
      }
      if (version === requestVersionRef.current.report) {
        setReport(null)
      }
    } finally {
      if (version === requestVersionRef.current.report) {
        setReportLoading(false)
      }
    }
  }

  const loadTaskErrors = async (runId: string, silent = false) => {
    const { controller, version } = startRequest('errors')
    if (!silent) {
      setErrorsLoading(true)
    }
    try {
      const response = await taskApi.getErrors(runId, { limit: 20 }, { signal: controller.signal }) as any
      if (version === requestVersionRef.current.errors) {
        setTaskErrors(response.errors || [])
      }
    } catch (error) {
      if (isCanceledRequest(error)) {
        return
      }
      if (version === requestVersionRef.current.errors) {
        setTaskErrors([])
      }
    } finally {
      if (version === requestVersionRef.current.errors) {
        setErrorsLoading(false)
      }
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
    if (!id) {
      return
    }
    setProgress(null)
    setReport(null)
    setTaskErrors([])
    void loadTaskBundle(id)
    return () => {
      requestControllersRef.current.bundle?.abort()
      requestControllersRef.current.report?.abort()
      requestControllersRef.current.errors?.abort()
    }
  }, [id])

  useEffect(() => {
    if (!id) {
      return
    }
    if (!task || !['scheduled', 'pending', 'running', 'paused'].includes(task.status)) {
      return
    }

    const fastTimer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void loadTaskAndProgress(id, true)
      }
    }, 5000)

    const slowTimer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void loadReport(id, true)
        void loadTaskErrors(id, true)
      }
    }, 20000)

    return () => {
      window.clearInterval(fastTimer)
      window.clearInterval(slowTimer)
    }
  }, [id, task])

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
          message.success(`HTML 报告已生成到 ${result.output_path}`)
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
      await loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '暂停失败')
    }
  }

  const handleResume = async () => {
    if (!id) return
    try {
      await taskApi.resume(id)
      message.success('任务已恢复')
      await loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '恢复失败')
    }
  }

  const handleStop = async () => {
    if (!id) return
    try {
      await taskApi.stop(id)
      message.success('任务已取消')
      await loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '取消失败')
    }
  }

  const handleStartNow = async () => {
    if (!id) return
    try {
      await taskApi.start(id)
      message.success('任务已开始执行')
      await loadTaskBundle(id, true)
    } catch (error: any) {
      message.error(error.message || '立即启动失败')
    }
  }

  const handleRecover = async () => {
    if (!id) return
    try {
      await taskApi.recover(id)
      message.success('任务已恢复执行，将继续补齐未完成部分')
      await loadTaskBundle(id, true)
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
      await loadTaskAndProgress(id, true)
    } catch (error: any) {
      message.error(error.message || '更新任务名称失败')
    } finally {
      setRenaming(false)
    }
  }

  const openRuntimeConfigEditor = async () => {
    if (!id) return
    try {
      setRuntimeConfigLoading(true)
      const response = await taskApi.getRuntimeConfig(id) as any
      const nextConfig = response as TaskRuntimeConfig
      setRuntimeConfig(nextConfig)
      runtimeForm.setFieldsValue({
        max_workers: nextConfig.multiprocess?.max_workers ?? undefined,
        executors: nextConfig.executors.map((executor) => ({
          id: executor.id,
          name: executor.name,
          provider: executor.provider,
          model: executor.model,
          concurrency: executor.concurrency,
          api_key: executor.api_key || '',
          after: executor.after || [],
        })),
      })
      setRuntimeConfigOpen(true)
    } catch (error: any) {
      message.error(error.message || '获取可编辑运行配置失败')
    } finally {
      setRuntimeConfigLoading(false)
    }
  }

  const handleSaveRuntimeConfig = async () => {
    if (!id) return
    try {
      const values = await runtimeForm.validateFields()
      setRuntimeConfigSaving(true)
      const payload = {
        max_workers: values.max_workers,
        executors: (values.executors || []).map((executor: any) => ({
          id: executor.id,
          concurrency: executor.concurrency,
          ...(String(executor.api_key || '').trim() ? { api_key: String(executor.api_key).trim() } : {}),
          after: Array.isArray(executor.after) ? executor.after : [],
        })),
      }
      const updated = await taskApi.updateRuntimeConfig(id, payload) as any
      setRuntimeConfig(updated as TaskRuntimeConfig)
      setRuntimeConfigOpen(false)
      message.success('运行时配置已更新')
      await loadTaskBundle(id, true)
    } catch (error: any) {
      if (error?.errorFields) {
        return
      }
      message.error(error.message || '更新运行时配置失败')
    } finally {
      setRuntimeConfigSaving(false)
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

  const tableData = executorItems.map((item) => ({ key: item.id, ...item }))
  const canRenameTask = !['running', 'paused'].includes(task.status)
  const canEditRuntimeConfig = !['running', 'paused'].includes(task.status)

  return (
    <div>
      <Space style={{ marginBottom: 16 }} wrap>
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
          刷新报告
        </Button>
        {canEditRuntimeConfig && (
          <Button icon={<EditOutlined />} loading={runtimeConfigLoading} onClick={openRuntimeConfigEditor}>
            编辑运行配置
          </Button>
        )}
        {(task.status === 'scheduled' || task.status === 'pending') && (
          <>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleStartNow}>
              立即启动
            </Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>
              取消任务
            </Button>
          </>
        )}
        {task.status === 'running' && (
          <>
            <Button icon={<PauseCircleOutlined />} onClick={handlePause}>
              暂停
            </Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>
              停止
            </Button>
          </>
        )}
        {task.status === 'paused' && (
          <>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleResume}>
              恢复
            </Button>
            <Button danger icon={<StopOutlined />} onClick={handleStop}>
              停止
            </Button>
          </>
        )}
        {(task.status === 'failed' || task.status === 'cancelled') && (
          <Button type="primary" icon={<SyncOutlined />} onClick={handleRecover}>
            恢复续跑
          </Button>
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
              {task.scheduled_at && (
                <Text type="secondary">
                  计划执行时间: {dayjs(task.scheduled_at).format('YYYY-MM-DD HH:mm:ss')}
                </Text>
              )}
              {report?.generated_at && (
                <Text type="secondary">
                  报告生成时间: {dayjs(report.generated_at).format('YYYY-MM-DD HH:mm:ss')}
                </Text>
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
        onCancel={() => !renaming && setRenameOpen(false)}
      >
        <Input
          maxLength={120}
          value={renameValue}
          onChange={(event) => setRenameValue(event.target.value)}
          placeholder="输入任务名称"
          onPressEnter={() => void handleRename()}
        />
      </Modal>

      <Modal
        title="编辑运行时配置"
        open={runtimeConfigOpen}
        width={920}
        okText="保存配置"
        cancelText="取消"
        confirmLoading={runtimeConfigSaving}
        onOk={handleSaveRuntimeConfig}
        onCancel={() => !runtimeConfigSaving && setRuntimeConfigOpen(false)}
      >
        {runtimeConfig ? (
          <Form form={runtimeForm} layout="vertical">
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message="这里只开放安全修改项"
              description="可修改项包括执行器并发、API Key、前置依赖 after，以及全局 max_workers。数据集、执行器 ID、Provider、模型等核心配置保持不变，用于保证续跑时复用已有结果。"
            />
            <Form.Item label="全局 max_workers" name="max_workers">
              <InputNumber min={1} style={{ width: '100%' }} placeholder="留空表示沿用原配置" />
            </Form.Item>
            <Form.List name="executors">
              {(fields) => (
                <Space direction="vertical" style={{ width: '100%' }} size={12}>
                  {fields.map((field) => (
                    <Card
                      key={field.key}
                      size="small"
                      title={runtimeForm.getFieldValue(['executors', field.name, 'name']) || `Executor ${field.name + 1}`}
                    >
                      <Form.Item name={[field.name, 'id']} hidden>
                        <Input />
                      </Form.Item>
                      <Form.Item name={[field.name, 'name']} hidden>
                        <Input />
                      </Form.Item>
                      <Form.Item name={[field.name, 'provider']} hidden>
                        <Input />
                      </Form.Item>
                      <Form.Item name={[field.name, 'model']} hidden>
                        <Input />
                      </Form.Item>

                      <Row gutter={16}>
                        <Col span={8}>
                          <Form.Item label="Provider / 模型">
                            <Text type="secondary">
                              {runtimeForm.getFieldValue(['executors', field.name, 'provider']) || '-'}
                              {' / '}
                              {runtimeForm.getFieldValue(['executors', field.name, 'model']) || '-'}
                            </Text>
                          </Form.Item>
                        </Col>
                        <Col span={4}>
                          <Form.Item
                            label="并发"
                            name={[field.name, 'concurrency']}
                            rules={[{ required: true, message: '请输入并发' }]}
                          >
                            <InputNumber min={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item label="API Key" name={[field.name, 'api_key']}>
                            <Input.Password placeholder="留空表示沿用原配置" />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Form.Item label="前置执行器 after" name={[field.name, 'after']}>
                        <Select
                          mode="multiple"
                          allowClear
                          placeholder="选择前置执行器"
                          options={(runtimeConfig?.executors || [])
                            .filter((executor) => executor.id !== runtimeForm.getFieldValue(['executors', field.name, 'id']))
                            .map((executor) => ({
                              label: `${executor.name} (${executor.id})`,
                              value: executor.id,
                            }))}
                        />
                      </Form.Item>
                    </Card>
                  ))}
                </Space>
              )}
            </Form.List>
          </Form>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
            <Spin />
          </div>
        )}
      </Modal>

      {(task.status === 'running' || task.status === 'paused' || report?.is_partial) && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="当前报告基于已落库数据实时计算"
          description="即使任务还没有完全跑完，页面也会展示阶段性指标、执行器结论和错误分布。每次请求都会重新计算，不使用缓存。"
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
            <Statistic title="总请求数" value={report?.metrics.total_requests || progress?.completed || 0} />
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
            scroll={{ x: 960 }}
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
                  <Paragraph style={{ marginBottom: 0 }} ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}>
                    {record.error?.error_message || '-'}
                  </Paragraph>
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
                        <Tag color={executorStatusColor(executor.status)}>{executor.status}</Tag>
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
            scroll={{ x: 1520 }}
            columns={[
              {
                title: '执行器',
                dataIndex: 'name',
                fixed: 'left',
                width: 240,
                render: (_value, record: ExecutorProgress) => (
                  <Space direction="vertical" size={4}>
                    <Space wrap>
                      <Text strong>{record.name}</Text>
                      <Tag color={record.id === baseline?.id ? 'blue' : executorStatusColor(record.status)}>
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
              { title: '尾响(ms)', dataIndex: 'avg_total_time', width: 120, render: (value: number) => value.toFixed(0) },
              { title: 'token速率(不带首响)', dataIndex: 'avg_token_per_second', width: 160, render: (value: number) => value.toFixed(2) },
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
