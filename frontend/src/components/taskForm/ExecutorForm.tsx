/**
 * Executor form modal component
 * Modal for creating/editing executor configurations
 */

import { useEffect, useState } from 'react'
import { Modal, Form, Input, InputNumber, Select, Space, Alert, Divider } from 'antd'
import { ApiOutlined, SettingOutlined } from '@ant-design/icons'
import {
  ExecutorConfig,
  EXECUTOR_TYPES,
  EXECUTOR_IMPLS,
  RateConfig,
} from '@/types/taskConfig'

interface ExecutorFormProps {
  open: boolean
  executor: ExecutorConfig | null
  onClose: () => void
  onSave: (executor: ExecutorConfig) => void
  onCheckPrice?: (provider: string, model: string) => Promise<boolean>
}

function stringifyJsonObject(value: unknown): string | undefined {
  if (
    !value
    || typeof value !== 'object'
    || Array.isArray(value)
    || Object.keys(value as Record<string, unknown>).length === 0
  ) {
    return undefined
  }

  return JSON.stringify(value, null, 2)
}

function parseOptionalJsonObject(value: string | undefined, label: string): Record<string, any> | null {
  const raw = value?.trim()
  if (!raw) {
    return null
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(`${label} must be valid JSON`)
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`)
  }

  return parsed as Record<string, any>
}

function validateJsonObject(label: string) {
  return async (_: unknown, value?: string) => {
    parseOptionalJsonObject(value, label)
  }
}

export default function ExecutorForm({
  open,
  executor,
  onClose,
  onSave,
  onCheckPrice,
}: ExecutorFormProps) {
  const [form] = Form.useForm()
  const [, setLoading] = useState(false)
  const [priceStatus, setPriceStatus] = useState<boolean | null>(null)

  // Reset form when executor changes
  useEffect(() => {
    if (open) {
      if (executor) {
        const param = executor.param || {}
        const {
          extra_body: extraBody,
          extra_headers: extraHeaders,
          extra_header: extraHeader,
          ...otherParam
        } = param
        form.setFieldsValue({
          ...executor,
          rate_qps: executor.rate?.qps,
          rate_interval: executor.rate?.interval_seconds,
          extra_body_json: stringifyJsonObject(extraBody),
          extra_headers_json: stringifyJsonObject(extraHeaders || extraHeader),
          other_param_json: stringifyJsonObject(otherParam),
        })
        setPriceStatus(executor.hasPrice ?? null)
      } else {
        form.resetFields()
        form.setFieldsValue({
          type: 'mock',
          impl: 'chat',
          concurrency: 1,
        })
        setPriceStatus(null)
      }
    }
  }, [open, executor, form])

  const handleOk = async () => {
    try {
      const values = await form.validateFields()
      const rate: RateConfig | null = values.rate_qps || values.rate_interval
        ? {
            qps: values.rate_qps || null,
            interval_seconds: values.rate_interval || null,
          }
        : null
      const extraBody = parseOptionalJsonObject(values.extra_body_json, 'Extra Body')
      const extraHeaders = parseOptionalJsonObject(values.extra_headers_json, 'Extra Headers')
      const otherParam = parseOptionalJsonObject(values.other_param_json, 'Custom Params') || {}
      const param = { ...otherParam }

      if (extraBody) {
        param.extra_body = extraBody
      }
      if (extraHeaders) {
        param.extra_headers = extraHeaders
      }

      const config: ExecutorConfig = {
        id: executor?.id || `executor-${Date.now().toString(36)}`,
        name: values.name,
        type: values.type,
        impl: values.impl,
        concurrency: values.concurrency,
        after: values.after || [],
        model: values.model || null,
        api_url: values.api_url || null,
        api_key: values.api_key || null,
        param,
        rate,
        hasPrice: priceStatus ?? undefined,
      }

      onSave(config)
      onClose()
    } catch (error) {
      console.error('Validation failed:', error)
    }
  }

  const handleCheckPrice = async () => {
    const type = form.getFieldValue('type')
    const model = form.getFieldValue('model')

    if (!type || !model) return

    setLoading(true)
    try {
      if (onCheckPrice) {
        const hasPrice = await onCheckPrice(type, model)
        setPriceStatus(hasPrice)
      }
    } finally {
      setLoading(false)
    }
  }

  const selectedType = Form.useWatch('type', form)

  return (
    <Modal
      title={executor ? '编辑执行器' : '添加执行器'}
      open={open}
      onOk={handleOk}
      onCancel={onClose}
      width={600}
      okText="保存"
      cancelText="取消"
      destroyOnClose
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          type: 'mock',
          impl: 'chat',
          concurrency: 1,
        }}
      >
        <Form.Item
          name="name"
          label="执行器名称"
          rules={[{ required: true, message: '请输入执行器名称' }]}
        >
          <Input placeholder="例如: GPT-4 测试" />
        </Form.Item>

        <Form.Item
          name="type"
          label="执行器类型"
          rules={[{ required: true, message: '请选择执行器类型' }]}
        >
          <Select options={EXECUTOR_TYPES} />
        </Form.Item>

        <Form.Item
          name="impl"
          label="API 类型"
          rules={[{ required: true, message: '请选择 API 类型' }]}
        >
          <Select options={EXECUTOR_IMPLS} />
        </Form.Item>

        <Form.Item
          name="concurrency"
          label="并发数"
          rules={[{ required: true, message: '请输入并发数' }]}
        >
          <InputNumber min={1} max={100} style={{ width: '100%' }} />
        </Form.Item>

        {selectedType !== 'mock' && (
          <>
            <Divider>
              <Space>
                <ApiOutlined />
                API 配置
              </Space>
            </Divider>

            <Form.Item
              name="model"
              label="模型名称"
              rules={[{ required: selectedType !== 'mock', message: '请输入模型名称' }]}
            >
              <Input
                placeholder="例如: gpt-4, qwen-max, deepseek-chat"
                onBlur={handleCheckPrice}
              />
            </Form.Item>

            <Form.Item
              name="api_url"
              label="API URL"
              rules={[{ required: selectedType !== 'mock', message: '请输入 API URL' }]}
            >
              <Input placeholder="例如: https://api.openai.com/v1" />
            </Form.Item>

            <Form.Item
              name="api_key"
              label="API Key"
            >
              <Input.Password placeholder="输入 API 密钥" />
            </Form.Item>

            {priceStatus !== null && (
              <Alert
                message={priceStatus
                  ? '已找到价格配置'
                  : '未找到价格配置，建议在价格管理中添加'}
                type={priceStatus ? 'success' : 'warning'}
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}

            <Form.Item
              name="extra_headers_json"
              label="Extra Headers (JSON)"
              tooltip="OpenAI-compatible extra_headers request option"
              rules={[{ validator: validateJsonObject('Extra Headers') }]}
            >
              <Input.TextArea
                rows={4}
                placeholder={'{\n  "x-trace-id": "demo-id"\n}'}
              />
            </Form.Item>

            <Form.Item
              name="extra_body_json"
              label="Extra Body (JSON)"
              tooltip="OpenAI-compatible extra_body request option"
              rules={[{ validator: validateJsonObject('Extra Body') }]}
            >
              <Input.TextArea
                rows={5}
                placeholder={'{\n  "thinking": {\n    "type": "enabled"\n  }\n}'}
              />
            </Form.Item>

            <Form.Item
              name="other_param_json"
              label="Custom Params (JSON)"
              tooltip="Other request params merged into executor.param"
              rules={[{ validator: validateJsonObject('Custom Params') }]}
            >
              <Input.TextArea
                rows={4}
                placeholder={'{\n  "temperature": 0.7,\n  "timeout": 120\n}'}
              />
            </Form.Item>
          </>
        )}

        <Divider>
          <Space>
            <SettingOutlined />
            高级设置
          </Space>
        </Divider>

        <Form.Item
          name="rate_qps"
          label="QPS 限制（每秒请求数）"
          help="可选：限制每秒发送的请求数"
        >
          <InputNumber
            min={0.1}
            step={0.1}
            precision={1}
            style={{ width: '100%' }}
            placeholder="不限制"
          />
        </Form.Item>

        <Form.Item
          name="rate_interval"
          label="请求间隔（秒）"
          help="可选：每次请求之间的间隔时间"
        >
          <InputNumber
            min={0}
            step={0.1}
            precision={2}
            style={{ width: '100%' }}
            placeholder="不限制"
          />
        </Form.Item>
      </Form>
    </Modal>
  )
}
