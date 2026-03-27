/**
 * Execution settings component
 * Configure iteration rounds, max duration, and mutation methods
 */

import { Card, Form, InputNumber, Checkbox, Space, Typography, Divider } from 'antd'
import { ClockCircleOutlined, SyncOutlined, ThunderboltOutlined } from '@ant-design/icons'
import useTaskFormStore from '@/stores/taskFormStore'
import { MUTATION_METHODS } from '@/types/taskConfig'

const { Text, Paragraph } = Typography

export default function ExecutionSettings() {
  const {
    iteratorConfig,
    setMaxRounds,
    setMaxTotalSeconds,
    setMutationChain,
  } = useTaskFormStore()

  return (
    <div>
      <Card title="执行方式设置" style={{ marginBottom: 16 }}>
        <Form layout="vertical">
          <Form.Item
            label={
              <Space>
                <SyncOutlined />
                <span>执行轮数</span>
              </Space>
            }
            help="每条数据将被执行的次数（经过变异后）"
          >
            <InputNumber
              min={1}
              max={100}
              value={iteratorConfig.max_rounds}
              onChange={(value) => setMaxRounds(value)}
              style={{ width: '100%' }}
              placeholder="默认 1 轮"
            />
          </Form.Item>

          <Form.Item
            label={
              <Space>
                <ClockCircleOutlined />
                <span>最大执行时长（秒）</span>
              </Space>
            }
            help="任务执行的最大时长，超过后停止（可选）"
          >
            <InputNumber
              min={1}
              value={iteratorConfig.max_total_seconds}
              onChange={(value) => setMaxTotalSeconds(value)}
              style={{ width: '100%' }}
              placeholder="不限制"
            />
          </Form.Item>
        </Form>
      </Card>

      <Card
        title={
          <Space>
            <ThunderboltOutlined />
            <span>变异方法</span>
          </Space>
        }
      >
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          选择要对数据应用的变异方法。变异方法会按顺序依次应用。
        </Paragraph>

        <Checkbox.Group
          value={iteratorConfig.mutation_chain}
          onChange={(values) => setMutationChain(values as string[])}
          style={{ width: '100%' }}
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {MUTATION_METHODS.map((method) => (
              <Card
                key={method.value}
                size="small"
                hoverable
                style={{
                  cursor: 'pointer',
                  borderColor: iteratorConfig.mutation_chain.includes(method.value)
                    ? '#1890ff'
                    : undefined,
                }}
                onClick={() => {
                  const current = iteratorConfig.mutation_chain
                  if (current.includes(method.value)) {
                    setMutationChain(current.filter((v) => v !== method.value))
                  } else {
                    setMutationChain([...current, method.value])
                  }
                }}
              >
                <Space>
                  <Checkbox value={method.value} />
                  <div>
                    <Text strong>{method.label}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {method.description}
                    </Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {method.example}
                    </Text>
                  </div>
                </Space>
              </Card>
            ))}
          </Space>
        </Checkbox.Group>

        <Divider />

        <Paragraph type="secondary" style={{ fontSize: 12 }}>
          <strong>当前配置：</strong>
          <br />
          执行轮数: {iteratorConfig.max_rounds || '不限制'} 轮
          <br />
          最大时长: {iteratorConfig.max_total_seconds ? `${iteratorConfig.max_total_seconds} 秒` : '不限制'}
          <br />
          变异方法: {iteratorConfig.mutation_chain.length > 0
            ? iteratorConfig.mutation_chain.join(' → ')
            : '无'}
        </Paragraph>
      </Card>
    </div>
  )
}
