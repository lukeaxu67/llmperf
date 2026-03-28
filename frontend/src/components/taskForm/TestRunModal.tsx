import { useState } from 'react'
import {
  Modal,
  Button,
  Space,
  Typography,
  Descriptions,
  Collapse,
  Spin,
  Alert,
  Card,
  Tag,
  Progress,
  Divider,
} from 'antd'
import {
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
} from '@ant-design/icons'
import { testRunApi, TestRunResponse } from '@/services/api'

const { Text, Paragraph } = Typography

interface TestRunModalProps {
  open: boolean
  yamlContent: string
  onClose: () => void
}

export default function TestRunModal({
  open,
  yamlContent,
  onClose,
}: TestRunModalProps) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<TestRunResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await testRunApi.run({ config_content: yamlContent })
      setResult(response as unknown as TestRunResponse)
    } catch (err: any) {
      setError(err.message || 'Test run failed')
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    setResult(null)
    setError(null)
    onClose()
  }

  const getLatencyColor = (ms: number) => {
    if (ms < 200) return 'success'
    if (ms < 500) return 'processing'
    if (ms < 1000) return 'warning'
    return 'error'
  }

  const getTpsColor = (tps: number) => {
    if (tps > 50) return 'success'
    if (tps > 20) return 'processing'
    if (tps > 10) return 'warning'
    return 'error'
  }

  return (
    <Modal
      title={(
        <Space>
          <PlayCircleOutlined />
          <span>测试运行</span>
        </Space>
      )}
      open={open}
      onCancel={handleClose}
      width={760}
      footer={[
        <Button key="close" onClick={handleClose}>
          关闭
        </Button>,
        <Button
          key="run"
          type="primary"
          icon={<PlayCircleOutlined />}
          loading={loading}
          onClick={handleRun}
        >
          {loading ? '运行中...' : '开始测试'}
        </Button>,
      ]}
    >
      <Alert
        message="测试运行说明"
        description="将使用数据集第一条记录，对当前配置中的每个执行器分别执行一次测试。结果只用于校验配置，不会写入数据库。"
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      {loading && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <Paragraph style={{ marginTop: 16 }} type="secondary">
            正在执行测试运行...
          </Paragraph>
        </div>
      )}

      {error && (
        <Alert
          message="测试运行失败"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {result && !loading && (
        <div>
          <Card size="small" style={{ marginBottom: 16 }}>
            <Space size="large" wrap>
              {result.success ? (
                <Tag icon={<CheckCircleOutlined />} color="success" style={{ fontSize: 14, padding: '4px 8px' }}>
                  全部执行器测试成功
                </Tag>
              ) : (
                <Tag icon={<CloseCircleOutlined />} color="error" style={{ fontSize: 14, padding: '4px 8px' }}>
                  存在执行器测试失败
                </Tag>
              )}
              <Text type="secondary">
                已测试执行器数: <Text strong>{result.results?.length || 0}</Text>
              </Text>
            </Space>
          </Card>

          <Collapse
            defaultActiveKey={(result.results || []).map((item) => item.executor_id)}
            items={(result.results || []).map((item) => ({
              key: item.executor_id,
              label: (
                <Space wrap>
                  <Text strong>{item.executor_name}</Text>
                  <Tag color={item.success ? 'success' : 'error'}>
                    {item.success ? '成功' : '失败'}
                  </Tag>
                  <Text type="secondary">{item.provider}/{item.model}</Text>
                </Space>
              ),
              children: (
                <Space direction="vertical" size={16} style={{ width: '100%' }}>
                  <Descriptions bordered size="small" column={2}>
                    <Descriptions.Item label="状态码">
                      <Tag color={item.success ? 'success' : 'error'}>
                        {typeof item.status_code === 'number' ? item.status_code : (item.success ? 200 : '-1')}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="错误类型">
                      {item.success ? (
                        <Text type="secondary">-</Text>
                      ) : (
                        <Text code>{item.error_type || 'UnknownError'}</Text>
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={(
                        <Space>
                          <ClockCircleOutlined />
                          首响
                        </Space>
                      )}
                    >
                      <Tag color={getLatencyColor(item.first_token_ms)}>
                        {item.first_token_ms.toFixed(0)} ms
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={(
                        <Space>
                          <ThunderboltOutlined />
                          token速率
                        </Space>
                      )}
                    >
                      <Tag color={getTpsColor(item.tokens_per_second)}>
                        {item.tokens_per_second.toFixed(1)} TPS
                      </Tag>
                    </Descriptions.Item>
                  </Descriptions>

                  <Card size="small" title="首响分析">
                    <Progress
                      percent={Math.min((item.first_token_ms / 2000) * 100, 100)}
                      status={item.first_token_ms < 500 ? 'success' : item.first_token_ms < 1000 ? 'normal' : 'exception'}
                      format={() => `${item.first_token_ms.toFixed(0)} ms`}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {item.first_token_ms < 200
                        ? '优秀，响应非常快'
                        : item.first_token_ms < 500
                        ? '良好，响应速度正常'
                        : item.first_token_ms < 1000
                        ? '一般，响应略慢'
                        : '偏慢，建议进一步排查'}
                    </Text>
                  </Card>

                  <Card
                    size="small"
                    title={(
                      <Space>
                        <FileTextOutlined />
                        <span>响应内容</span>
                      </Space>
                    )}
                  >
                    {item.success ? (
                      <Paragraph
                        ellipsis={{ rows: 6, expandable: true, symbol: '展开' }}
                        style={{
                          marginBottom: 0,
                          padding: 12,
                          background: '#f5f5f5',
                          borderRadius: 4,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {item.response || '(无内容)'}
                      </Paragraph>
                    ) : (
                      <div>
                        <Alert
                          message="执行失败"
                          description="已返回详细错误信息，可直接据此排查鉴权、地址、模型名或请求参数问题。"
                          type="error"
                          showIcon
                        />
                        <Divider style={{ margin: '12px 0' }} />
                        <Paragraph
                          style={{
                            marginBottom: 0,
                            padding: 12,
                            background: '#fff2f0',
                            border: '1px solid #ffccc7',
                            borderRadius: 4,
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            fontFamily: 'Consolas, Monaco, monospace',
                          }}
                        >
                          {item.error || 'Unknown error'}
                        </Paragraph>
                      </div>
                    )}
                  </Card>
                </Space>
              ),
            }))}
          />
        </div>
      )}

      {!result && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
          <PlayCircleOutlined style={{ fontSize: 48, marginBottom: 16 }} />
          <Paragraph type="secondary">
            点击“开始测试”执行一轮首样本校验
          </Paragraph>
        </div>
      )}
    </Modal>
  )
}
