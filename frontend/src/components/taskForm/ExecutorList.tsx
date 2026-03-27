import { useEffect, useState } from 'react'
import {
  Card,
  List,
  Button,
  Space,
  Tag,
  Typography,
  Empty,
  Popconfirm,
  Tooltip,
  Badge,
  message,
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CopyOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DollarOutlined,
} from '@ant-design/icons'
import { pricingApi } from '@/services/api'
import useTaskFormStore from '@/stores/taskFormStore'
import { ExecutorConfig } from '@/types/taskConfig'
import ExecutorTopologyGraph from '@/components/ExecutorTopologyGraph'
import { buildTopologyFromExecutors } from '@/utils/executorTopology'
import ExecutorForm from './ExecutorForm'

const { Text } = Typography

const TYPE_COLORS: Record<string, string> = {
  openai: 'green',
  qianwen: 'orange',
  zhipu: 'blue',
  deepseek: 'purple',
  spark: 'cyan',
  hunyuan: 'geekblue',
  huoshan: 'volcano',
  moonshot: 'magenta',
  mock: 'default',
}

export default function ExecutorList() {
  const {
    executors,
    addExecutor,
    updateExecutor,
    removeExecutor,
    duplicateExecutor,
    reorderExecutors,
    setExecutorPriceStatus,
  } = useTaskFormStore()

  const [formOpen, setFormOpen] = useState(false)
  const [editingExecutor, setEditingExecutor] = useState<ExecutorConfig | null>(null)

  useEffect(() => {
    executors.forEach(async (executor) => {
      if (executor.type !== 'mock' && executor.model) {
        const hasPrice = await checkPrice(executor.type, executor.model)
        if (executor.hasPrice !== hasPrice) {
          setExecutorPriceStatus(executor.id, hasPrice)
        }
      }
    })
  }, [executors, setExecutorPriceStatus])

  const checkPrice = async (provider: string, model: string): Promise<boolean> => {
    try {
      const response = await pricingApi.list({ provider, model }) as any
      return Boolean(response.items?.length)
    } catch {
      return false
    }
  }

  const handleAdd = () => {
    setEditingExecutor(null)
    setFormOpen(true)
  }

  const handleEdit = (executor: ExecutorConfig) => {
    setEditingExecutor(executor)
    setFormOpen(true)
  }

  const handleSave = (executor: ExecutorConfig) => {
    if (editingExecutor) {
      updateExecutor(editingExecutor.id, executor)
      return
    }
    const newId = addExecutor(executor.type)
    updateExecutor(newId, executor)
  }

  const handleDelete = (id: string) => {
    removeExecutor(id)
    message.success('执行器已删除')
  }

  const handleDuplicate = (id: string) => {
    duplicateExecutor(id)
    message.success('执行器已复制')
  }

  const topology = buildTopologyFromExecutors(executors)

  return (
    <div>
      <Card
        title={(
          <Space>
            <ApiOutlined />
            <span>执行器配置</span>
            <Badge count={executors.length} style={{ backgroundColor: '#1890ff' }} />
          </Space>
        )}
        extra={(
          <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>
            添加执行器
          </Button>
        )}
      >
        {executors.length === 0 ? (
          <Empty
            description="暂无执行器，点击上方按钮添加"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <>
            <Card size="small" style={{ marginBottom: 16, background: '#fafafa' }}>
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                <Text strong>执行拓扑预览</Text>
                <Text type="secondary">
                  可通过列表顺序和前置依赖共同编排执行路径。同一厂商要串行压满并发时，建议为后续模型配置前置执行器。
                </Text>
                <ExecutorTopologyGraph topology={topology} height={320} />
              </Space>
            </Card>

            <List
              dataSource={executors}
              renderItem={(executor, index) => (
                <List.Item
                  actions={[
                    (
                      <Tooltip title="上移" key="up">
                        <Button
                          type="text"
                          icon={<ArrowUpOutlined />}
                          disabled={index === 0}
                          onClick={() => reorderExecutors(index, index - 1)}
                        />
                      </Tooltip>
                    ),
                    (
                      <Tooltip title="下移" key="down">
                        <Button
                          type="text"
                          icon={<ArrowDownOutlined />}
                          disabled={index === executors.length - 1}
                          onClick={() => reorderExecutors(index, index + 1)}
                        />
                      </Tooltip>
                    ),
                    (
                      <Tooltip title="编辑" key="edit">
                        <Button
                          type="text"
                          icon={<EditOutlined />}
                          onClick={() => handleEdit(executor)}
                        />
                      </Tooltip>
                    ),
                    (
                      <Tooltip title="复制" key="copy">
                        <Button
                          type="text"
                          icon={<CopyOutlined />}
                          onClick={() => handleDuplicate(executor.id)}
                        />
                      </Tooltip>
                    ),
                    (
                      <Popconfirm
                        key="delete"
                        title="确定要删除这个执行器吗？"
                        onConfirm={() => handleDelete(executor.id)}
                        okText="删除"
                        cancelText="取消"
                      >
                        <Tooltip title="删除">
                          <Button type="text" danger icon={<DeleteOutlined />} />
                        </Tooltip>
                      </Popconfirm>
                    ),
                  ]}
                >
                  <List.Item.Meta
                    title={(
                      <Space wrap>
                        <Tag color={TYPE_COLORS[executor.type] || 'default'}>
                          {executor.type.toUpperCase()}
                        </Tag>
                        <Text strong>{executor.name}</Text>
                        <Tag color="blue">顺序 {index + 1}</Tag>
                        {executor.type !== 'mock' && (
                          <PriceStatusTag
                            hasPrice={executor.hasPrice}
                            type={executor.type}
                            model={executor.model}
                          />
                        )}
                      </Space>
                    )}
                    description={(
                      <Space size="large" wrap>
                        <Text type="secondary">ID: {executor.id}</Text>
                        <Text type="secondary">并发: {executor.concurrency}</Text>
                        {executor.model && (
                          <Text type="secondary">模型: {executor.model}</Text>
                        )}
                        {executor.rate?.qps && (
                          <Text type="secondary">QPS: {executor.rate.qps}</Text>
                        )}
                        {executor.after && executor.after.length > 0 && (
                          <Text type="secondary">前置: {executor.after.join(', ')}</Text>
                        )}
                      </Space>
                    )}
                  />
                </List.Item>
              )}
            />
          </>
        )}
      </Card>

      <ExecutorForm
        open={formOpen}
        executor={editingExecutor}
        executors={executors}
        onClose={() => setFormOpen(false)}
        onSave={handleSave}
        onCheckPrice={checkPrice}
      />
    </div>
  )
}

function PriceStatusTag({
  hasPrice,
  type,
  model,
}: {
  hasPrice?: boolean
  type: string
  model?: string | null
}) {
  if (hasPrice === undefined || hasPrice === null) {
    return null
  }

  if (hasPrice) {
    return (
      <Tooltip title="已配置价格">
        <Tag icon={<CheckCircleOutlined />} color="success">
          <DollarOutlined />
        </Tag>
      </Tooltip>
    )
  }

  return (
    <Tooltip title={`未找到 ${type}/${model} 的价格配置`}>
      <Tag icon={<ExclamationCircleOutlined />} color="warning">
        <DollarOutlined />
      </Tag>
    </Tooltip>
  )
}
