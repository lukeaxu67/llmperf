/**
 * Create Task Page
 * Step-by-step task creation wizard
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
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
    description: '配置执行轮数和变异',
    icon: <SettingOutlined />,
  },
  {
    title: '执行器',
    description: '配置执行器参数',
    icon: <ApiOutlined />,
  },
  {
    title: '确认',
    description: '预览并提交',
    icon: <CheckCircleOutlined />,
  },
]

export default function CreateTask() {
  const navigate = useNavigate()
  const {
    currentStep,
    setStep,
    nextStep,
    prevStep,
    generateYamlContent,
    reset,
    taskType,
    setTaskType,
  } = useTaskFormStore()

  const [submitting, setSubmitting] = useState(false)
  const [yamlValid, setYamlValid] = useState(false)
  const [testRunOpen, setTestRunOpen] = useState(false)
  const [createdTaskId, setCreatedTaskId] = useState<string | null>(null)

  // Reset form on mount
  useEffect(() => {
    reset()
  }, [])

  // Validate current step before proceeding
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
      const response = await taskApi.create({
        config_content: yamlContent,
        auto_start: true,
        task_type: taskType,
      }) as any

      setCreatedTaskId(response.run_id)
      message.success('任务创建成功')
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
  }

  // Show success result after task creation
  if (createdTaskId) {
    return (
      <div>
        <div style={{ marginBottom: 16 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
            返回列表
          </Button>
        </div>

        <Result
          status="success"
          title="任务创建成功"
          subTitle={`任务 ID: ${createdTaskId}`}
          extra={[
            <Button type="primary" key="view" onClick={handleViewTask}>
              查看任务
            </Button>,
            <Button key="another" onClick={handleCreateAnother}>
              创建新任务
            </Button>,
            <Button key="list" onClick={() => navigate('/tasks')}>
              返回列表
            </Button>,
          ]}
        />
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
            <Card style={{ marginTop: 16 }}>
              <Space>
                <Button
                  icon={<BugOutlined />}
                  onClick={() => setTestRunOpen(true)}
                >
                  测试运行
                </Button>
                <Alert
                  message="测试运行会使用第一条数据，对所有执行器各执行一次，结果不会保存"
                  type="info"
                  showIcon
                  style={{ display: 'inline-flex' }}
                />
              </Space>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
          返回列表
        </Button>
      </div>

      <Title level={4}>创建任务</Title>
      <Paragraph type="secondary">
        通过分步向导创建基准测试任务
      </Paragraph>

      {/* Task Type Selection */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Space direction="vertical" size={0}>
              <Text strong>任务类型</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                选择任务类型以确定测试策略
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
              <Radio.Button value="benchmark">
                <Space>
                  <ExperimentOutlined />
                  基准测试
                </Space>
              </Radio.Button>
              <Radio.Button value="monitoring">
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
                  ? '基准测试：对选定模型进行一次性性能测试'
                  : '持续监控：定期监控模型性能，适用于长期跟踪'
              }
              type="info"
              showIcon
            />
          </Col>
        </Row>
      </Card>

      {/* Steps */}
      <Card style={{ marginBottom: 24 }}>
        <Steps current={currentStep} items={steps} onChange={setStep} />
      </Card>

      {/* Step Content */}
      <div style={{ marginBottom: 24 }}>
        {renderStepContent()}
      </div>

      {/* Navigation */}
      <Card>
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Button
            disabled={currentStep === 0}
            onClick={prevStep}
          >
            上一步
          </Button>

          <Space>
            {currentStep < steps.length - 1 ? (
              <Button type="primary" onClick={handleNext}>
                下一步
              </Button>
            ) : (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                loading={submitting}
                onClick={handleSubmit}
              >
                创建并运行
              </Button>
            )}
          </Space>
        </Space>
      </Card>

      {/* Test Run Modal */}
      <TestRunModal
        open={testRunOpen}
        yamlContent={generateYamlContent()}
        onClose={() => setTestRunOpen(false)}
      />
    </div>
  )
}
