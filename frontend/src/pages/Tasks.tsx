import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Card,
  Button,
  Space,
  Tag,
  Input,
  Select,
  Typography,
  Tooltip,
  Popconfirm,
  message,
  Empty,
  Badge,
} from 'antd'
import {
  PlusOutlined,
  SearchOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  StopOutlined,
  RedoOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import StatusTag from '@/components/StatusTag'
import TaskProgressBar from '@/components/TaskProgressBar'
import { taskApi, Task, TaskProgress } from '@/services/api'

const { Title } = Typography
const { Search } = Input

export default function Tasks() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [tasks, setTasks] = useState<Task[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [statusFilter, setStatusFilter] = useState<string | undefined>()
  const [searchText, setSearchText] = useState('')
  const [progressMap, setProgressMap] = useState<Record<string, TaskProgress>>({})

  useEffect(() => {
    fetchTasks()
  }, [page, pageSize, statusFilter])

  // Poll for running tasks progress
  useEffect(() => {
    const runningTasks = tasks.filter(t => t.status === 'running')
    if (runningTasks.length === 0) return

    const interval = setInterval(async () => {
      for (const task of runningTasks) {
        try {
          const progress = await taskApi.getProgress(task.run_id) as any
          setProgressMap(prev => ({ ...prev, [task.run_id]: progress }))
        } catch (e) {
          // Ignore
        }
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [tasks])

  const fetchTasks = async () => {
    setLoading(true)
    try {
      const response = await taskApi.list({
        limit: pageSize,
        offset: (page - 1) * pageSize,
        status: statusFilter,
      }) as any
      setTasks(response.tasks || [])
      setTotal(response.total || 0)
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
      message.success('任务已重新启动')
      fetchTasks()
    } catch (error: any) {
      message.error(error.message || '重试失败')
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
      render: (name: string) => name || '未命名任务',
    },
    {
      title: 'Run ID',
      dataIndex: 'run_id',
      key: 'run_id',
      width: 180,
      render: (id: string) => (
        <Typography.Text code copyable={{ text: id }} style={{ fontSize: 12 }}>
          {id.slice(0, 12)}...
        </Typography.Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => <StatusTag status={status as any} />,
    },
    {
      title: '进度',
      key: 'progress',
      width: 250,
      render: (_: any, record: Task) => {
        if (record.status !== 'running') return '-'
        const progress = progressMap[record.run_id]
        if (!progress) return <Tag>加载中...</Tag>
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
      render: (time: string) => time ? dayjs(time).format('YYYY-MM-DD HH:mm:ss') : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 180,
      render: (_: any, record: Task) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/tasks/${record.run_id}`)}
            />
          </Tooltip>
          {record.status === 'running' && (
            <Popconfirm
              title="确定要取消此任务吗？"
              onConfirm={() => handleCancel(record.run_id)}
            >
              <Tooltip title="取消任务">
                <Button type="text" size="small" danger icon={<StopOutlined />} />
              </Tooltip>
            </Popconfirm>
          )}
          {record.status === 'failed' && (
            <Tooltip title="重试任务">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined />}
                onClick={() => handleRetry(record.run_id)}
              />
            </Tooltip>
          )}
          {record.status !== 'running' && (
            <Popconfirm
              title="确定要删除此任务吗？"
              onConfirm={() => handleDelete(record.run_id)}
            >
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

      <Card>
        <div style={{ marginBottom: 16, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Select
            placeholder="状态筛选"
            allowClear
            style={{ width: 150 }}
            value={statusFilter}
            onChange={setStatusFilter}
            options={[
              { value: 'pending', label: '等待中' },
              { value: 'running', label: '运行中' },
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
            showTotal: (total) => `共 ${total} 条`,
            onChange: (p, ps) => {
              setPage(p)
              setPageSize(ps)
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
