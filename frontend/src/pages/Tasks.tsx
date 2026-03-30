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
  Input,
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
  CopyOutlined,
  SearchOutlined,
  FileTextOutlined,
  DollarOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, pricingApi, Task, TaskProgress } from '@/services/api'

const { Title, Text } = Typography

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
  const [searchText, setSearchText] = useState('')
  const [progressMap, setProgressMap] = useState<Record<string, TaskProgress>>({})
  const [totalCost, setTotalCost] = useState(0)

  useEffect(() => {
    fetchTasks()
  }, [page, pageSize, statusFilter])

  useEffect(() => {
    const activeTasks = tasks.filter(
      (t) =>
        t.status === 'scheduled' ||
        t.status === 'pending' ||
        t.status === 'running' ||
        t.status === 'paused'
    )
    if (activeTasks.length === 0) return

    const interval = setInterval(async () => {
      if (document.visibilityState !== 'visible') return
      try {
        const response = (await taskApi.getProgressBatch(
          activeTasks.map((task) => task.run_id)
        )) as any
        setProgressMap((prev) => ({ ...prev, ...(response.items || {}) }))
      } catch {
        // ignore polling errors
      }
    }, 5000)

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
      message.success('任务已恢复执行')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '恢复执行失败')
    }
  }

  const handleReuseConfig = async (runId: string) => {
    try {
      const response = (await taskApi.getConfig(runId)) as any
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
      fixed: 'left' as const,
      width: 200,
      render: (name: string, record: TaskWithCost) => {
        const displayName = name || '未命名任务'
        if (record.status === 'completed' || record.status === 'failed') {
          return (
            <a
              onClick={() => navigate(`/tasks/${record.run_id}`)}
              style={{ fontWeight: 500 }}
            >
              {displayName}
            </a>
          )
        }
        return <Text>{displayName}</Text>
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 140,
      render: (status: string, record: TaskWithCost) => {
        if (status === 'completed') {
          return (
            <Space size="small">
              <StatusTag status={status as any} />
              <Tag
                color="green"
                style={{
                  cursor: 'pointer',
                  borderRadius: 4,
                  fontSize: 12,
                }}
                onClick={() => navigate(`/tasks/${record.run_id}`)}
              >
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
      render: (cost: number, _record: TaskWithCost) => {
        if (typeof cost !== 'number' || cost <= 0) return '-'
        return (
          <Tag color="red" style={{ borderRadius: 4 }}>
            ¥{cost.toFixed(4)}
          </Tag>
        )
      },
    },
    {
      title: '进度',
      key: 'progress',
      width: 200,
      render: (_: unknown, record: TaskWithCost) => {
        if (record.status !== 'running' && record.status !== 'paused') return '-'
        const progress = progressMap[record.run_id]
        if (!progress) return <Tag style={{ borderRadius: 4 }}>加载中...</Tag>
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
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) =>
        time ? dayjs(time).format('MM-DD HH:mm:ss') : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 180,
      fixed: 'right' as const,
      render: (_: unknown, record: TaskWithCost) => (
        <Space size={2}>
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined style={{ color: 'var(--color-text)' }} />}
              onClick={() => navigate(`/tasks/${record.run_id}`)}
            />
          </Tooltip>
          <Tooltip title="复用配置">
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined style={{ color: 'var(--color-text)' }} />}
              onClick={() => handleReuseConfig(record.run_id)}
            />
          </Tooltip>
          {(record.status === 'scheduled' || record.status === 'pending') && (
            <Tooltip title="立即启动">
              <Button
                type="text"
                size="small"
                icon={<PlayCircleOutlined style={{ color: '#52c41a' }} />}
                onClick={() => handleStart(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'failed' || record.status === 'cancelled') && (
            <Tooltip title="恢复执行">
              <Button
                type="text"
                size="small"
                icon={<SyncOutlined style={{ color: '#1677ff' }} />}
                onClick={() => handleRecover(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'completed' ||
            record.status === 'failed' ||
            record.status === 'cancelled') && (
            <Tooltip title="重新执行">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined style={{ color: 'var(--color-text)' }} />}
                onClick={() => handleRerun(record.run_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'running' ||
            record.status === 'paused' ||
            record.status === 'scheduled' ||
            record.status === 'pending') && (
            <Popconfirm
              title="确定要取消此任务吗？"
              onConfirm={() => handleCancel(record.run_id)}
            >
              <Tooltip title="取消任务">
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<StopOutlined />}
                />
              </Tooltip>
            </Popconfirm>
          )}
          {(record.status === 'completed' ||
            record.status === 'failed' ||
            record.status === 'cancelled') && (
            <Popconfirm
              title="确定要删除此任务吗？"
              onConfirm={() => handleDelete(record.run_id)}
            >
              <Tooltip title="删除任务">
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <Title level={3} style={{ margin: 0, marginBottom: 4 }}>
            任务管理
          </Title>
          <Text type="secondary">管理和监控所有任务执行</Text>
        </div>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => navigate('/tasks/create')}
          style={{ borderRadius: 8 }}
        >
          创建任务
        </Button>
      </div>

      {/* Stats Row */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
            }}
          >
            <Space>
              <DollarOutlined style={{ color: '#cf1322', fontSize: 20 }} />
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  累计总费用
                </Text>
                <div style={{ fontWeight: 600, color: '#cf1322', fontSize: 18 }}>
                  ¥{totalCost.toFixed(4)}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
            }}
          >
            <Space>
              <DollarOutlined style={{ color: '#1677ff', fontSize: 20 }} />
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  任务总数
                </Text>
                <div style={{ fontWeight: 600, fontSize: 18 }}>{total}</div>
              </div>
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card
            size="small"
            style={{
              borderRadius: 10,
              border: '1px solid var(--color-border-secondary)',
            }}
          >
            <Space>
              <DollarOutlined style={{ color: '#722ed1', fontSize: 20 }} />
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  平均任务费用
                </Text>
                <div style={{ fontWeight: 600, color: '#722ed1', fontSize: 18 }}>
                  ¥{total > 0 ? (totalCost / total).toFixed(4) : '0.0000'}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Main Table Card */}
      <Card
        style={{
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        {/* Filters */}
        <div
          style={{
            marginBottom: 16,
            display: 'flex',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <Input
            placeholder="搜索任务名称"
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 200, borderRadius: 8 }}
            allowClear
          />
          <Select
            placeholder="状态筛选"
            allowClear
            style={{ width: 140 }}
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
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchTasks}
            style={{ borderRadius: 8 }}
          >
            刷新
          </Button>
        </div>

        {/* Table */}
        <Table
          columns={columns}
          dataSource={tasks.filter((t) =>
            searchText
              ? t.task_name?.toLowerCase().includes(searchText.toLowerCase())
              : true
          )}
          rowKey="run_id"
          loading={loading}
          scroll={{ x: 1000 }}
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
