import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Card,
  Button,
  Space,
  Tag,
  Select,
  Typography,
  Tooltip,
  Popconfirm,
  message,
  Empty,
  Statistic,
  Row,
  Col,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  StopOutlined,
  RedoOutlined,
  PlayCircleOutlined,
  SyncOutlined,
  FileTextOutlined,
  DollarOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, pricingApi, Task, TaskProgress } from '@/services/api'

const { Title } = Typography

interface TaskWithCost extends Task {
  total_cost?: number
  currency?: string
}

export default function Tasks() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [tasks, setTasks] = useState<TaskWithCost[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [statusFilter, setStatusFilter] = useState<string | undefined>()
  const [progressMap, setProgressMap] = useState<Record<string, TaskProgress>>({})
  const [totalCost, setTotalCost] = useState(0)

  useEffect(() => {
    fetchTasks()
  }, [page, pageSize, statusFilter])

  useEffect(() => {
    const activeTasks = tasks.filter(
      (t) =>
        t.status === 'scheduled'
        || t.status === 'pending'
        || t.status === 'running'
        || t.status === 'paused',
    )
    if (activeTasks.length === 0) {
      return
    }

    const interval = setInterval(async () => {
      for (const task of activeTasks) {
        try {
          const progress = await taskApi.getProgress(task.run_id) as any
          setProgressMap((prev) => ({ ...prev, [task.run_id]: progress }))
        } catch {
          // ignore polling errors
        }
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [tasks])

  const fetchTasks = async () => {
    setLoading(true)
    try {
      const [response, costRes] = await Promise.all([
        taskApi.list({
          limit: pageSize,
          offset: (page - 1) * pageSize,
          status: statusFilter,
        }) as any,
        pricingApi.getTotalCost() as any,
      ])
      setTasks(response.tasks || [])
      setTotal(response.total || 0)
      setTotalCost(costRes?.total_cost || 0)
    } catch (error: any) {
      message.error(error.message || '获取任务列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCancel = async (runId: string) => {
    try {
      await taskApi.cancel(runId)
      message.success('任务已取消')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '取消失败')
    }
  }

  const handleRetry = async (runId: string) => {
    try {
      await taskApi.retry(runId)
      message.success('失败任务已重新提交')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '重试失败')
    }
  }

  const handleRerun = async (runId: string) => {
    try {
      await taskApi.rerun(runId, { auto_start: true })
      message.success('已按原配置重新执行')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '重新执行失败')
    }
  }

  const handleStart = async (runId: string) => {
    try {
      await taskApi.start(runId)
      message.success('任务已开始执行')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '立即启动失败')
    }
  }

  const handleRecover = async (runId: string) => {
    try {
      await taskApi.recover(runId)
      message.success('任务已恢复执行，将继续补齐未完成部分')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '恢复执行失败')
    }
  }

  const handleReuseConfig = async (runId: string) => {
    try {
      const response = await taskApi.getConfig(runId) as any
      navigate('/tasks/create', { state: { configContent: response.config_content } })
    } catch (error: any) {
      message.error(error.message || '载入历史配置失败')
    }
  }

  const handleDelete = async (runId: string) => {
    try {
      await taskApi.delete(runId)
      message.success('任务已删除')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '删除失败')
    }
  }

  const columns = [
    {
      title: '任务名称',
      dataIndex: 'task_name',
      key: 'task_name',
      ellipsis: true,
      render: (name: string, record: TaskWithCost) => {
        const displayName = name || '未命名任务'
        if (record.status === 'completed' || record.status === 'failed' || record.status === 'cancelled') {
          return <a onClick={() => navigate(`/tasks/${record.run_id}`)}>{displayName}</a>
        }
        return displayName
      },
    },
    {
      title: 'Run ID',
      dataIndex: 'run_id',
      key: 'run_id',
      width: 150,
      render: (id: string) => (
        <Typography.Text code copyable={{ text: id }} style={{ fontSize: 12 }}>
          {id.slice(0, 8)}...
        </Typography.Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 180,
      render: (status: string, record: TaskWithCost) => {
        if (status === 'completed') {
          return (
            <Space>
              <StatusTag status={status as any} />
              <Tag color="green" style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${record.run_id}`)}>
                <FileTextOutlined /> 报告
              </Tag>
            </Space>
          )
        }
        return <StatusTag status={status as any} />
      },
    },
    {
      title: '费用',
      dataIndex: 'total_cost',
      key: 'total_cost',
      width: 120,
      render: (cost: number, record: TaskWithCost) => {
        if (typeof cost !== 'number' || cost <= 0) {
          return '-'
        }
        return <Tag color="red">¥{cost.toFixed(4)} {record.currency || 'CNY'}</Tag>
      },
    },
    {
      title: '进度',
      key: 'progress',
      width: 220,
      render: (_: unknown, record: TaskWithCost) => {
        if (record.status !== 'running' && record.status !== 'paused') {
          return '-'
        }
        const progress = progressMap[record.run_id]
        if (!progress) {
          return <Tag>加载中...</Tag>
        }
        return (
          <TaskProgressBar
            percent={progress.progress_percent}
            successCount={progress.success_count}
            errorCount={progress.error_count}
            total={progress.total}
            size="small"
          />
        )
      },
    },
    {
      title: '计划时间',
      dataIndex: 'scheduled_at',
      key: 'scheduled_at',
      width: 180,
      render: (time?: string) => (time ? dayjs(time).format('MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) => (time ? dayjs(time).format('MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 220,
      render: (_: unknown, record: TaskWithCost) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/tasks/${record.run_id}`)}
            />
          </Tooltip>
          <Tooltip title="复用配置">
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => handleReuseConfig(record.run_id)}
            />
          </Tooltip>
          {(record.status === 'scheduled' || record.status === 'pending') && (
            <Tooltip title="立即启动">
              <Button
                type="text"
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleStart(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'failed' || record.status === 'cancelled') && (
            <Tooltip title="恢复执行（续跑未完成部分）">
              <Button
                type="text"
                size="small"
                icon={<SyncOutlined />}
                onClick={() => handleRecover(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'completed' || record.status === 'failed' || record.status === 'cancelled') && (
            <Tooltip title="立即重跑（创建新任务）">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined />}
                onClick={() => handleRerun(record.run_id)}
              />
            </Tooltip>
          )}
          {record.status === 'failed' && (
            <Tooltip title="按失败任务重试接口重提">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined />}
                onClick={() => handleRetry(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'running' || record.status === 'paused' || record.status === 'scheduled' || record.status === 'pending') && (
            <Popconfirm title="确定要取消此任务吗？" onConfirm={() => handleCancel(record.run_id)}>
              <Tooltip title="取消任务">
                <Button type="text" size="small" danger icon={<StopOutlined />} />
              </Tooltip>
            </Popconfirm>
          )}
          {(record.status === 'completed' || record.status === 'failed' || record.status === 'cancelled') && (
            <Popconfirm title="确定要删除此任务吗？" onConfirm={() => handleDelete(record.run_id)}>
              <Tooltip title="删除任务">
                <Button type="text" size="small" danger icon={<DeleteOutlined />} />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>任务管理</Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/tasks/create')}>
          创建任务
        </Button>
      </div>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="累计总费用"
              value={totalCost}
              precision={4}
              prefix={<DollarOutlined style={{ color: '#cf1322' }} />}
              suffix="元"
              valueStyle={{ color: '#cf1322', fontSize: 20 }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic title="任务总数" value={total} suffix="个" />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="平均任务费用"
              value={total > 0 ? totalCost / total : 0}
              precision={4}
              prefix="¥"
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <div style={{ marginBottom: 16, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Select
            placeholder="状态筛选"
            allowClear
            style={{ width: 160 }}
            value={statusFilter}
            onChange={setStatusFilter}
            options={[
              { value: 'scheduled', label: '已定时' },
              { value: 'pending', label: '等待中' },
              { value: 'running', label: '运行中' },
              { value: 'paused', label: '已暂停' },
              { value: 'completed', label: '已完成' },
              { value: 'failed', label: '失败' },
              { value: 'cancelled', label: '已取消' },
            ]}
          />
          <Button icon={<ReloadOutlined />} onClick={fetchTasks}>
            刷新
          </Button>
        </div>

        <Table
          columns={columns}
          dataSource={tasks}
          rowKey="run_id"
          loading={loading}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            showTotal: (count) => `共 ${count} 条`,
            onChange: (nextPage, nextPageSize) => {
              setPage(nextPage)
              setPageSize(nextPageSize)
            },
          }}
          locale={{
            emptyText: <Empty description="暂无任务" />,
          }}
        />
      </Card>
    </div>
  )
}
