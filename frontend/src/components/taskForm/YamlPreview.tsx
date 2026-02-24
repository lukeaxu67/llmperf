/**
 * YAML Preview component
 * Display generated YAML, allow editing, import/export
 */

import { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Space,
  Switch,
  Upload,
  message,
  Tooltip,
  Alert,
} from 'antd'
import {
  EditOutlined,
  EyeOutlined,
  DownloadOutlined,
  UploadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import YamlEditor from '@/components/YamlEditor'
import useTaskFormStore from '@/stores/taskFormStore'
import { validateYaml, exportYaml, readYamlFile } from '@/utils/yamlGenerator'
import { configApi } from '@/services/api'

interface YamlPreviewProps {
  onValidChange?: (valid: boolean) => void
}

export default function YamlPreview({ onValidChange }: YamlPreviewProps) {
  const { generateYamlContent, loadFromYaml } = useTaskFormStore()
  const [editMode, setEditMode] = useState(false)
  const [yamlContent, setYamlContent] = useState('')
  const [validation, setValidation] = useState<{ valid: boolean; error?: string } | null>(null)
  const [isValidating, setIsValidating] = useState(false)

  // Initialize YAML content
  useEffect(() => {
    setYamlContent(generateYamlContent())
  }, [generateYamlContent])

  const handleToggleEdit = () => {
    if (!editMode) {
      // Entering edit mode, keep current content
      if (!yamlContent) {
        setYamlContent(generateYamlContent())
      }
    }
    setEditMode(!editMode)
  }

  const handleValidate = async (content?: string) => {
    const yaml = content || yamlContent
    setIsValidating(true)

    try {
      // First do local validation
      const localResult = validateYaml(yaml)
      if (!localResult.valid) {
        setValidation(localResult)
        onValidChange?.(false)
        return
      }

      // Then do server validation
      const result = await configApi.validate(yaml) as any
      const serverValid = result.valid

      setValidation({
        valid: serverValid,
        error: serverValid ? undefined : (result.errors?.join('; ') || 'Validation failed'),
      })
      onValidChange?.(serverValid)

      if (serverValid) {
        message.success('配置验证通过')
      }
    } catch (error: any) {
      setValidation({
        valid: false,
        error: error.message || 'Validation failed',
      })
      onValidChange?.(false)
    } finally {
      setIsValidating(false)
    }
  }

  const handleExport = () => {
    if (!yamlContent) {
      message.warning('没有可导出的内容')
      return
    }
    exportYaml(yamlContent, 'task-config.yaml')
    message.success('配置已导出')
  }

  const handleImport = async (file: File) => {
    try {
      const content = await readYamlFile(file)
      setYamlContent(content)

      // Try to parse and load into form
      try {
        loadFromYaml(content)
        message.success('配置已导入')
      } catch (parseError) {
        message.warning('配置已导入，但格式可能有问题')
      }

      // Validate imported content
      handleValidate(content)
    } catch (error) {
      message.error('读取文件失败')
    }

    return false // Prevent default upload behavior
  }

  const handleContentChange = (value: string) => {
    setYamlContent(value)
    setValidation(null) // Clear validation when content changes
  }

  return (
    <div>
      <Card
        title="YAML 预览"
        extra={
          <Space>
            <Switch
              checkedChildren={<EditOutlined />}
              unCheckedChildren={<EyeOutlined />}
              checked={editMode}
              onChange={handleToggleEdit}
            />
            <Tooltip title="导入配置">
              <Upload
                accept=".yaml,.yml"
                beforeUpload={handleImport}
                showUploadList={false}
              >
                <Button icon={<UploadOutlined />}>
                  导入
                </Button>
              </Upload>
            </Tooltip>
            <Tooltip title="导出配置">
              <Button icon={<DownloadOutlined />} onClick={handleExport}>
                导出
              </Button>
            </Tooltip>
            <Button
              type="primary"
              icon={validation?.valid ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
              loading={isValidating}
              onClick={() => handleValidate()}
            >
              验证配置
            </Button>
          </Space>
        }
      >
        {validation && (
          <Alert
            message={validation.valid ? '配置验证通过' : '配置验证失败'}
            description={validation.error}
            type={validation.valid ? 'success' : 'error'}
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <YamlEditor
          value={yamlContent}
          onChange={handleContentChange}
          height={500}
          readOnly={!editMode}
          error={validation?.valid === false ? validation.error : undefined}
        />

        {editMode && (
          <Alert
            message="编辑模式"
            description="编辑 YAML 后，配置将在提交时使用。注意：手动编辑的内容不会同步回表单。"
            type="info"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
    </div>
  )
}
