/**
 * Test run modal component
 * Run a single test request and display results
 */

import { useState } from 'react'
import {
  Modal,
  Button,
  Space,
  Typography,
  Descriptions,
  Spin,
  Alert,
  Card,
  Tag,
  Progress,
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
      title={
        <Space>
          <PlayCircleOutlined />
          <span>测试运行</span>
        </Space>
      }
      open={open}
      onCancel={handleClose}
      width={700}
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
        description="将使用配置中的第一个执行器，对数据集的第一条记录进行测试运行。测试结果不会保存到数据库。"
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
          {/* Status Card */}
          <Card size="small" style={{ marginBottom: 16 }}>
            <Space size="large">
              {result.success ? (
                <Tag icon={<CheckCircleOutlined />} color="success" style={{ fontSize: 14, padding: '4px 8px' }}>
                  测试成功
                </Tag>
              ) : (
                <Tag icon={<CloseCircleOutlined />} color="error" style={{ fontSize: 14, padding: '4px 8px' }}>
                  测试失败
                </Tag>
              )}
              <Text type="secondary">
                总耗时: <Text strong>{result.duration_ms.toFixed(0)} ms</Text>
              </Text>
            </Space>
          </Card>

          {result.success ? (
            <>
              {/* Metrics */}
              <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
                <Descriptions.Item
                  label={
                    <Space>
                      <ClockCircleOutlined />
                      首字延迟
                    </Space>
                  }
                >
                  <Tag color={getLatencyColor(result.first_token_ms)}>
                    {result.first_token_ms.toFixed(0)} ms
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <Space>
                      <ThunderboltOutlined />
                      吞吐量
                    </Space>
                  }
                >
                  <Tag color={getTpsColor(result.tokens_per_second)}>
                    {result.tokens_per_second.toFixed(1)} TPS
                  </Tag>
                </Descriptions.Item>
              </Descriptions>

              {/* Latency Progress */}
              <Card size="small" title="首字延迟分析" style={{ marginBottom: 16 }}>
                <Progress
                  percent={Math.min((result.first_token_ms / 2000) * 100, 100)}
                  status={result.first_token_ms < 500 ? 'success' : result.first_token_ms < 1000 ? 'normal' : 'exception'}
                  format={() => `${result.first_token_ms.toFixed(0)} ms`}
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {result.first_token_ms < 200
                    ? '优秀 - 响应非常快'
                    : result.first_token_ms < 500
                    ? '良好 - 响应速度正常'
                    : result.first_token_ms < 1000
                    ? '一般 - 响应稍慢'
                    : '较慢 - 建议优化'}
                </Text>
              </Card>

              {/* Response Content */}
              <Card
                size="small"
                title={
                  <Space>
                    <FileTextOutlined />
                    <span>响应内容</span>
                  </Space>
                }
              >
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
                  {result.response || '(无内容)'}
                </Paragraph>
              </Card>
            </>
          ) : (
            <Alert
              message="错误信息"
              description={result.error || 'Unknown error'}
              type="error"
              showIcon
            />
          )}
        </div>
      )}

      {!result && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
          <PlayCircleOutlined style={{ fontSize: 48, marginBottom: 16 }} />
          <Paragraph type="secondary">
            点击"开始测试"运行一次测试请求
          </Paragraph>
        </div>
      )}
    </Modal>
  )
}
