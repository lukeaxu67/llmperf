/**
 * Executor list component
 * Display and manage list of executors
 */

import { useState, useEffect } from 'react'
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
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DollarOutlined,
} from '@ant-design/icons'
import { pricingApi } from '@/services/api'
import useTaskFormStore from '@/stores/taskFormStore'
import { ExecutorConfig } from '@/types/taskConfig'
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
    setExecutorPriceStatus,
  } = useTaskFormStore()

  const [formOpen, setFormOpen] = useState(false)
  const [editingExecutor, setEditingExecutor] = useState<ExecutorConfig | null>(null)

  // Check prices for all executors on mount
  useEffect(() => {
    executors.forEach(async (exec) => {
      if (exec.type !== 'mock' && exec.model) {
        const hasPrice = await checkPrice(exec.type, exec.model)
        if (exec.hasPrice !== hasPrice) {
          setExecutorPriceStatus(exec.id, hasPrice)
        }
      }
    })
  }, [executors])

  const checkPrice = async (provider: string, model: string): Promise<boolean> => {
    try {
      const response = await pricingApi.list({ provider, model }) as any
      return response.items && response.items.length > 0
    } catch (error) {
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
    } else {
      addExecutor(executor.type)
      const store = useTaskFormStore.getState()
      const newId = store.executors[store.executors.length - 1]?.id
      if (newId) {
        updateExecutor(newId, executor)
      }
    }
  }

  const handleDelete = (id: string) => {
    removeExecutor(id)
    message.success('执行器已删除')
  }

  const handleDuplicate = (id: string) => {
    duplicateExecutor(id)
    message.success('执行器已复制')
  }

  const handleCheckPrice = async (provider: string, model: string): Promise<boolean> => {
    return await checkPrice(provider, model)
  }

  return (
    <div>
      <Card
        title={
          <Space>
            <ApiOutlined />
            <span>执行器配置</span>
            <Badge count={executors.length} style={{ backgroundColor: '#1890ff' }} />
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>
            添加执行器
          </Button>
        }
      >
        {executors.length === 0 ? (
          <Empty
            description="暂无执行器，点击上方按钮添加"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <List
            dataSource={executors}
            renderItem={(executor) => (
              <List.Item
                actions={[
                  <Tooltip title="编辑" key="edit">
                    <Button
                      type="text"
                      icon={<EditOutlined />}
                      onClick={() => handleEdit(executor)}
                    />
                  </Tooltip>,
                  <Tooltip title="复制" key="copy">
                    <Button
                      type="text"
                      icon={<CopyOutlined />}
                      onClick={() => handleDuplicate(executor.id)}
                    />
                  </Tooltip>,
                  <Popconfirm
                    key="delete"
                    title="确定要删除这个执行器吗？"
                    onConfirm={() => handleDelete(executor.id)}
                    okText="删除"
                    cancelText="取消"
                  >
                    <Tooltip title="删除">
                      <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                      />
                    </Tooltip>
                  </Popconfirm>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <Tag color={TYPE_COLORS[executor.type] || 'default'}>
                        {executor.type.toUpperCase()}
                      </Tag>
                      <Text strong>{executor.name}</Text>
                      {executor.type !== 'mock' && (
                        <PriceStatusTag
                          hasPrice={executor.hasPrice}
                          type={executor.type}
                          model={executor.model}
                        />
                      )}
                    </Space>
                  }
                  description={
                    <Space size="large" wrap>
                      <Text type="secondary">
                        ID: {executor.id}
                      </Text>
                      <Text type="secondary">
                        并发: {executor.concurrency}
                      </Text>
                      {executor.model && (
                        <Text type="secondary">
                          模型: {executor.model}
                        </Text>
                      )}
                      {executor.rate?.qps && (
                        <Text type="secondary">
                          QPS: {executor.rate.qps}
                        </Text>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>

      <ExecutorForm
        open={formOpen}
        executor={editingExecutor}
        onClose={() => setFormOpen(false)}
        onSave={handleSave}
        onCheckPrice={handleCheckPrice}
      />
    </div>
  )
}

// Price status tag component
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
