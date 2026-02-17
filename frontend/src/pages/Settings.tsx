import { useEffect, useState } from 'react'
import {
  Card,
  Tabs,
  Typography,
  List,
  Descriptions,
  Tag,
  Button,
  Form,
  Input,
  Switch,
  Select,
  message,
  Space,
  Divider,
  Alert,
} from 'antd'
import {
  SettingOutlined,
  BellOutlined,
  ApiOutlined,
  DatabaseOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import { configApi } from '@/services/api'

const { Title, Text, Paragraph } = Typography

export default function Settings() {
  const [runtimeConfig, setRuntimeConfig] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchRuntimeConfig()
  }, [])

  const fetchRuntimeConfig = async () => {
    setLoading(true)
    try {
      const response = await configApi.getRuntime() as any
      setRuntimeConfig(response)
    } catch (error) {
      console.error('Failed to fetch runtime config:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveNotification = async (values: any) => {
    message.success('通知设置已保存（功能开发中）')
  }

  return (
    <div>
      <Title level={4}>系统设置</Title>
      <Paragraph type="secondary">
        配置系统运行参数和通知渠道
      </Paragraph>

      <Tabs
        defaultActiveKey="general"
        items={[
          {
            key: 'general',
            label: (
              <span>
                <SettingOutlined />
                常规设置
              </span>
            ),
            children: (
              <Card loading={loading}>
                {runtimeConfig && (
                  <Descriptions bordered column={1}>
                    <Descriptions.Item label="数据库路径">
                      <Text code>{runtimeConfig.db_path}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="日志目录">
                      <Text code>{runtimeConfig.log_dir}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="日志级别">
                      <Tag color={runtimeConfig.log_level === 'DEBUG' ? 'blue' : 'green'}>
                        {runtimeConfig.log_level}
                      </Tag>
                    </Descriptions.Item>
                  </Descriptions>
                )}
              </Card>
            ),
          },
          {
            key: 'notifications',
            label: (
              <span>
                <BellOutlined />
                通知设置
              </span>
            ),
            children: (
              <Card>
                <Alert
                  message="通知功能"
                  description="配置任务状态变更的通知渠道，支持邮件、Webhook、钉钉等方式"
                  type="info"
                  showIcon
                  style={{ marginBottom: 24 }}
                />

                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleSaveNotification}
                  initialValues={{
                    enableEmail: false,
                    enableWebhook: false,
                    enableDingtalk: false,
                  }}
                >
                  <Form.Item label="任务完成通知" name="onTaskComplete" valuePropName="checked">
                    <Switch />
                  </Form.Item>

                  <Form.Item label="任务失败通知" name="onTaskFail" valuePropName="checked">
                    <Switch />
                  </Form.Item>

                  <Form.Item label="错误率告警" name="onErrorThreshold" valuePropName="checked">
                    <Switch />
                  </Form.Item>

                  <Divider />

                  <Title level={5}>邮件通知</Title>
                  <Form.Item label="SMTP 服务器" name="smtpHost">
                    <Input placeholder="smtp.example.com" />
                  </Form.Item>
                  <Form.Item label="SMTP 端口" name="smtpPort">
                    <Input type="number" placeholder="587" />
                  </Form.Item>
                  <Form.Item label="发件人邮箱" name="smtpUser">
                    <Input placeholder="your-email@example.com" />
                  </Form.Item>
                  <Form.Item label="SMTP 密码" name="smtpPass">
                    <Input.Password placeholder="••••••••" />
                  </Form.Item>
                  <Form.Item label="收件人列表" name="emailRecipients">
                    <Input.TextArea
                      rows={2}
                      placeholder="多个邮箱用逗号分隔"
                    />
                  </Form.Item>

                  <Divider />

                  <Title level={5}>Webhook 通知</Title>
                  <Form.Item label="Webhook URL" name="webhookUrl">
                    <Input placeholder="https://hooks.example.com/webhook" />
                  </Form.Item>

                  <Divider />

                  <Title level={5}>钉钉通知</Title>
                  <Form.Item label="钉钉 Webhook" name="dingtalkWebhook">
                    <Input placeholder="https://oapi.dingtalk.com/robot/send?access_token=xxx" />
                  </Form.Item>
                  <Form.Item label="加签密钥" name="dingtalkSecret">
                    <Input.Password placeholder="SECxxx" />
                  </Form.Item>

                  <Form.Item>
                    <Button type="primary" htmlType="submit" icon={<SaveOutlined />}>
                      保存设置
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            ),
          },
          {
            key: 'api',
            label: (
              <span>
                <ApiOutlined />
                API 密钥
              </span>
            ),
            children: (
              <Card>
                <Alert
                  message="API 密钥管理"
                  description="通过环境变量配置各 LLM 提供商的 API 密钥，不支持在界面中直接配置"
                  type="warning"
                  showIcon
                  style={{ marginBottom: 24 }}
                />

                <List
                  itemLayout="horizontal"
                  dataSource={[
                    { name: 'OpenAI', env: 'OPENAI_API_KEY', configured: true },
                    { name: '智谱 AI', env: 'ZHIPU_API_KEY', configured: false },
                    { name: 'DeepSeek', env: 'DEEPSEEK_API_KEY', configured: false },
                    { name: '讯飞星火', env: 'IFLYTEK_API_KEY', configured: false },
                    { name: '火山引擎', env: 'HUOSHAN_API_KEY', configured: false },
                  ]}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={item.name}
                        description={<Text code>{item.env}</Text>}
                      />
                      <Tag color={item.configured ? 'green' : 'default'}>
                        {item.configured ? '已配置' : '未配置'}
                      </Tag>
                    </List.Item>
                  )}
                />
              </Card>
            ),
          },
          {
            key: 'about',
            label: (
              <span>
                <DatabaseOutlined />
                关于
              </span>
            ),
            children: (
              <Card>
                <Descriptions column={1}>
                  <Descriptions.Item label="应用名称">LLMPerf</Descriptions.Item>
                  <Descriptions.Item label="版本">0.1.0</Descriptions.Item>
                  <Descriptions.Item label="描述">
                    Unified benchmarking toolkit for large language model providers
                  </Descriptions.Item>
                  <Descriptions.Item label="技术栈">
                    <Space>
                      <Tag>Python 3.12</Tag>
                      <Tag>FastAPI</Tag>
                      <Tag>React</Tag>
                      <Tag>Ant Design</Tag>
                      <Tag>SQLite</Tag>
                    </Space>
                  </Descriptions.Item>
                </Descriptions>

                <Divider />

                <Paragraph type="secondary">
                  LLMPerf 是一个统一的 LLM 性能基准测试工具包，支持多种 LLM 提供商，
                  提供全面的性能分析、成本计算和可视化报告功能。
                </Paragraph>
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}
