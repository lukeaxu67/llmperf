import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Card,
  Button,
  Space,
  Typography,
  message,
  Alert,
  List,
  Divider,
} from 'antd'
import {
  ArrowLeftOutlined,
  PlayCircleOutlined,
  FileTextOutlined,
} from '@ant-design/icons'
import YamlEditor from '@/components/YamlEditor'
import { configApi, taskApi } from '@/services/api'

const { Title, Paragraph } = Typography

const defaultConfig = `info: "示例任务"

dataset:
  source:
    type: "jsonl"
    name: "demo"
    config:
      path: "resource/demo.jsonl"
  iterator:
    mutation_chain: ["identity"]
    max_rounds: 1

executors:
  - id: "executor-001"
    name: "测试执行器"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-model"
`

export default function CreateTask() {
  const navigate = useNavigate()
  const [config, setConfig] = useState(defaultConfig)
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[]; warnings: string[] } | null>(null)
  const [templates, setTemplates] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetchTemplates()
  }, [])

  const fetchTemplates = async () => {
    setLoading(true)
    try {
      const response = await configApi.listTemplates() as any
      setTemplates(response || [])
    } catch (error) {
      console.error('Failed to fetch templates:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleValidate = async () => {
    try {
      const result = await configApi.validate(config) as any
      setValidation({
        valid: result.valid,
        errors: result.errors || [],
        warnings: result.warnings || [],
      })
      if (result.valid) {
        message.success('配置验证通过')
      } else {
        message.error('配置验证失败')
      }
    } catch (error: any) {
      message.error(error.message || '验证失败')
    }
  }

  const handleSubmit = async () => {
    // Validate first
    try {
      const result = await configApi.validate(config) as any
      if (!result.valid) {
        message.error('请先修复配置错误')
        setValidation({
          valid: result.valid,
          errors: result.errors || [],
          warnings: result.warnings || [],
        })
        return
      }
    } catch (error: any) {
      message.error('验证失败: ' + error.message)
      return
    }

    setSubmitting(true)
    try {
      const response = await taskApi.create({ config_content: config }) as any
      message.success('任务创建成功')
      navigate(`/tasks/${response.run_id}`)
    } catch (error: any) {
      message.error(error.message || '创建失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleLoadTemplate = async (name: string) => {
    try {
      const response = await configApi.getTemplate(name) as any
      if (response.content) {
        setConfig(response.content)
        setValidation(null)
        message.success(`已加载模板: ${name}`)
      }
    } catch (error: any) {
      message.error(error.message || '加载模板失败')
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
      </div>

      <Title level={4}>创建任务</Title>
      <Paragraph type="secondary">
        通过 YAML 配置文件创建新的基准测试任务
      </Paragraph>

      <div style={{ display: 'flex', gap: 24 }}>
        {/* Main Editor */}
        <div style={{ flex: 1 }}>
          <Card
            title="配置编辑器"
            extra={
              <Space>
                <Button onClick={handleValidate}>
                  验证配置
                </Button>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  loading={submitting}
                  onClick={handleSubmit}
                >
                  创建并运行
                </Button>
              </Space>
            }
          >
            {validation && !validation.valid && (
              <Alert
                message="配置验证错误"
                description={
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {validation.errors.map((err, i) => (
                      <li key={i} style={{ color: '#ff4d4f' }}>{err}</li>
                    ))}
                  </ul>
                }
                type="error"
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}

            {validation && validation.valid && (
              <Alert
                message="配置验证通过"
                type="success"
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}

            <YamlEditor
              value={config}
              onChange={(value) => {
                setConfig(value)
                setValidation(null)
              }}
              height={500}
            />
          </Card>
        </div>

        {/* Sidebar */}
        <div style={{ width: 300 }}>
          <Card title="配置模板" size="small" loading={loading}>
            <List
              size="small"
              dataSource={templates}
              renderItem={(item) => (
                <List.Item
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleLoadTemplate(item.name)}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined />}
                    title={item.name}
                    description={item.description || '无描述'}
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无模板' }}
            />
          </Card>

          <Divider />

          <Card title="配置说明" size="small">
            <div style={{ fontSize: 12, color: '#666' }}>
              <p><strong>info:</strong> 任务描述信息</p>
              <p><strong>dataset:</strong> 数据集配置</p>
              <p><strong>executors:</strong> 执行器列表</p>
              <p><strong>pricing:</strong> 价格配置（可选）</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
