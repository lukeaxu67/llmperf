import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import dayjs, { Dayjs } from 'dayjs'
import {
  Card,
  Button,
  Steps,
  Space,
  Typography,
  message,
  Result,
  Alert,
  Radio,
  Row,
  Col,
  DatePicker,
} from 'antd'
import {
  ArrowLeftOutlined,
  DatabaseOutlined,
  SettingOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  PlayCircleOutlined,
  BugOutlined,
  ExperimentOutlined,
  MonitorOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import useTaskFormStore from '@/stores/taskFormStore'
import { taskApi } from '@/services/api'
import {
  DatasetSelector,
  ExecutionSettings,
  ExecutorList,
  YamlPreview,
  TestRunModal,
} from '@/components/taskForm'

const { Title, Paragraph, Text } = Typography

const steps = [
  {
    title: '数据集',
    description: '选择数据集和描述',
    icon: <DatabaseOutlined />,
  },
  {
    title: '执行设置',
    description: '配置轮数和变异方式',
    icon: <SettingOutlined />,
  },
  {
    title: '执行器',
    description: '配置模型和接口参数',
    icon: <ApiOutlined />,
  },
  {
    title: '确认提交',
    description: '预览 YAML 并执行',
    icon: <CheckCircleOutlined />,
  },
]

type StartMode = 'now' | 'scheduled'

interface LocationState {
  configContent?: string
}

export default function CreateTask() {
  const navigate = useNavigate()
  const location = useLocation()
  const locationState = (location.state || {}) as LocationState
  const {
    currentStep,
    setStep,
    nextStep,
    prevStep,
    generateYamlContent,
    loadFromYaml,
    reset,
    taskType,
    setTaskType,
  } = useTaskFormStore()

  const [submitting, setSubmitting] = useState(false)
  const [yamlValid, setYamlValid] = useState(false)
  const [testRunOpen, setTestRunOpen] = useState(false)
  const [createdTaskId, setCreatedTaskId] = useState<string | null>(null)
  const [createdMessage, setCreatedMessage] = useState('任务创建成功')
  const [startMode, setStartMode] = useState<StartMode>('now')
  const [scheduledAt, setScheduledAt] = useState<Dayjs | null>(null)

  useEffect(() => {
    if (!locationState.configContent) {
      return
    }
    try {
      loadFromYaml(locationState.configContent)
      setStep(0)
      message.success('已载入历史任务配置')
      navigate(location.pathname, { replace: true, state: null })
    } catch (error: any) {
      message.error(error?.message || '载入历史配置失败')
    }
  }, [loadFromYaml, location.pathname, locationState.configContent, navigate, setStep])

  const canProceed = () => {
    const { selectedDataset, executors } = useTaskFormStore.getState()

    switch (currentStep) {
      case 0:
        if (!selectedDataset) {
          message.warning('请选择一个数据集')
          return false
        }
        return true
      case 1:
        return true
      case 2:
        if (executors.length === 0) {
          message.warning('请至少添加一个执行器')
          return false
        }
        return true
      case 3:
        if (!yamlValid) {
          message.warning('请先验证 YAML 配置')
          return false
        }
        if (startMode === 'scheduled' && !scheduledAt) {
          message.warning('请选择定时执行时间')
          return false
        }
        if (startMode === 'scheduled' && scheduledAt && scheduledAt.isBefore(dayjs())) {
          message.warning('定时执行时间必须晚于当前时间')
          return false
        }
        return true
      default:
        return true
    }
  }

  const handleNext = () => {
    if (canProceed()) {
      nextStep()
    }
  }

  const handleSubmit = async () => {
    if (!canProceed()) {
      return
    }

    const yamlContent = generateYamlContent()
    setSubmitting(true)

    try {
      const isScheduled = startMode === 'scheduled' && scheduledAt
      const response = (await taskApi.create({
        config_content: yamlContent,
        auto_start: !isScheduled,
        task_type: taskType,
        scheduled_at: isScheduled ? scheduledAt.toISOString() : undefined,
      })) as any

      setCreatedTaskId(response.run_id)
      setCreatedMessage(isScheduled ? '任务已创建并加入定时执行' : '任务已创建并开始执行')
      message.success(isScheduled ? '定时任务创建成功' : '任务创建成功')
    } catch (error: any) {
      message.error(error.message || '创建任务失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleViewTask = () => {
    if (createdTaskId) {
      navigate(`/tasks/${createdTaskId}`)
    }
  }

  const handleCreateAnother = () => {
    reset()
    setCreatedTaskId(null)
    setCreatedMessage('任务创建成功')
    setStartMode('now')
    setScheduledAt(null)
  }

  if (createdTaskId) {
    return (
      <div className="fade-in">
        <div style={{ marginBottom: 16 }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/tasks')}
            style={{ borderRadius: 8 }}
          >
            返回列表
          </Button>
        </div>

        <Card
          style={{
            borderRadius: 16,
            textAlign: 'center',
            padding: 40,
          }}
        >
          <Result
            status="success"
            title={<Title level={3} style={{ marginBottom: 8 }}>{createdMessage}</Title>}
            subTitle={<Text type="secondary">任务 ID: {createdTaskId}</Text>}
            extra={[
              <Button
                type="primary"
                key="view"
                onClick={handleViewTask}
                style={{ borderRadius: 8, height: 40 }}
              >
                查看任务
              </Button>,
              <Button
                key="another"
                onClick={handleCreateAnother}
                style={{ borderRadius: 8, height: 40 }}
              >
                再建一个
              </Button>,
              <Button
                key="list"
                onClick={() => navigate('/tasks')}
                style={{ borderRadius: 8, height: 40 }}
              >
                返回列表
              </Button>,
            ]}
          />
        </Card>
      </div>
    )
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return <DatasetSelector />
      case 1:
        return <ExecutionSettings />
      case 2:
        return <ExecutorList />
      case 3:
        return (
          <div>
            <YamlPreview onValidChange={setYamlValid} />
            <Card
              style={{
                marginTop: 16,
                borderRadius: 12,
                border: '1px solid var(--color-border-secondary)',
              }}
            >
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Space>
                  <Button
                    icon={<BugOutlined />}
                    onClick={() => setTestRunOpen(true)}
                    style={{ borderRadius: 8 }}
                  >
                    测试运行
                  </Button>
                  <Alert
                    message="测试运行会使用第 1 条数据，对所有执行器各执行 1 次，结果不会落库。"
                    type="info"
                    showIcon
                    style={{ display: 'inline-flex', borderRadius: 8 }}
                  />
                </Space>

                <div
                  style={{
                    padding: 16,
                    background: 'var(--color-bg-spotlight)',
                    borderRadius: 8,
                  }}
                >
                  <Text strong style={{ display: 'block', marginBottom: 12 }}>
                    启动方式
                  </Text>
                  <Radio.Group
                    value={startMode}
                    onChange={(e) => setStartMode(e.target.value)}
                    optionType="button"
                    buttonStyle="solid"
                    style={{ marginBottom: startMode === 'scheduled' ? 12 : 0 }}
                  >
                    <Radio.Button value="now" style={{ borderRadius: '8px 0 0 8px' }}>
                      <Space size={4}>
                        <PlayCircleOutlined />
                        立即执行
                      </Space>
                    </Radio.Button>
                    <Radio.Button value="scheduled" style={{ borderRadius: '0 8px 8px 0' }}>
                      <Space size={4}>
                        <ClockCircleOutlined />
                        定时执行
                      </Space>
                    </Radio.Button>
                  </Radio.Group>
                  {startMode === 'scheduled' && (
                    <div style={{ marginTop: 12 }}>
                      <DatePicker
                        showTime
                        style={{ width: 320, borderRadius: 8 }}
                        placeholder="选择执行时间"
                        value={scheduledAt}
                        onChange={setScheduledAt}
                        disabledDate={(current) =>
                          !!current && current.endOf('day').isBefore(dayjs().startOf('day'))
                        }
                      />
                    </div>
                  )}
                </div>
              </Space>
            </Card>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div style={{ marginBottom: 16 }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/tasks')}
          style={{ borderRadius: 8 }}
        >
          返回列表
        </Button>
      </div>

      <Title level={3} style={{ marginBottom: 4 }}>
        创建任务
      </Title>
      <Paragraph type="secondary" style={{ marginBottom: 24 }}>
        按步骤配置数据集、执行参数和执行器，最后选择立即执行或定时执行。
      </Paragraph>

      {/* Task Type Card */}
      <Card
        style={{
          marginBottom: 24,
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Space direction="vertical" size={0}>
              <Text strong>任务类型</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                选择任务类型以决定任务用途
              </Text>
            </Space>
          </Col>
          <Col>
            <Radio.Group
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              optionType="button"
              buttonStyle="solid"
            >
              <Radio.Button value="benchmark" style={{ borderRadius: '8px 0 0 8px' }}>
                <Space>
                  <ExperimentOutlined />
                  基准测试
                </Space>
              </Radio.Button>
              <Radio.Button value="monitoring" style={{ borderRadius: '0 8px 8px 0' }}>
                <Space>
                  <MonitorOutlined />
                  持续监控
                </Space>
              </Radio.Button>
            </Radio.Group>
          </Col>
          <Col flex="auto">
            <Alert
              message={
                taskType === 'benchmark'
                  ? '基准测试会立即或按计划执行一轮完整评测'
                  : '持续监控用于周期性地观察模型表现变化'
              }
              type="info"
              showIcon
              style={{ borderRadius: 8 }}
            />
          </Col>
        </Row>
      </Card>

      {/* Steps Card */}
      <Card
        style={{
          marginBottom: 24,
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        <Steps
          current={currentStep}
          items={steps}
          onChange={setStep}
          style={{ padding: '0 24px' }}
        />
      </Card>

      {/* Step Content */}
      <div style={{ marginBottom: 24 }}>{renderStepContent()}</div>

      {/* Navigation Card */}
      <Card
        style={{
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
        styles={{ body: { padding: '16px 24px' } }}
      >
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Button
            disabled={currentStep === 0}
            onClick={prevStep}
            style={{ borderRadius: 8, minWidth: 80 }}
          >
            上一步
          </Button>

          <Space>
            {currentStep < steps.length - 1 ? (
              <Button
                type="primary"
                onClick={handleNext}
                style={{ borderRadius: 8, minWidth: 80 }}
              >
                下一步
              </Button>
            ) : (
              <Button
                type="primary"
                icon={startMode === 'scheduled' ? <ClockCircleOutlined /> : <PlayCircleOutlined />}
                loading={submitting}
                onClick={handleSubmit}
                style={{ borderRadius: 8, minWidth: 120 }}
              >
                {startMode === 'scheduled' ? '创建并定时' : '创建并执行'}
              </Button>
            )}
          </Space>
        </Space>
      </Card>

      <TestRunModal
        open={testRunOpen}
        yamlContent={generateYamlContent()}
        onClose={() => setTestRunOpen(false)}
      />
    </div>
  )
}
