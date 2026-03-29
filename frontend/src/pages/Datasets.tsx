import { useEffect, useMemo, useState } from 'react'
import {
  Button,
  Card,
  Descriptions,
  Empty,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Progress,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  Upload,
} from 'antd'
import {
  DatabaseOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileTextOutlined,
  InboxOutlined,
} from '@ant-design/icons'
import type { UploadProps } from 'antd'
import { Dataset, DatasetValidationResult, datasetApi } from '@/services/api'
import DatasetPreviewList from '@/components/DatasetPreviewList'

const { Dragger } = Upload
const { Paragraph, Text, Title } = Typography

type UploadFormValues = {
  name?: string
  description?: string
  encoding?: string
}

export default function Datasets() {
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewData, setPreviewData] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [validation, setValidation] = useState<DatasetValidationResult | null>(null)
  const [form] = Form.useForm<UploadFormValues>()

  useEffect(() => {
    void fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const response = await datasetApi.list() as any
      setDatasets(response?.datasets || [])
    } catch (error: any) {
      message.error(error.message || '获取数据集列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handlePreview = async (datasetId: string) => {
    setPreviewVisible(true)
    setPreviewLoading(true)
    try {
      const response = await datasetApi.preview(datasetId, 20) as any
      setPreviewData(response)
    } catch (error: any) {
      message.error(error.message || '获取数据集预览失败')
      setPreviewData(null)
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleDelete = async (dataset: Dataset) => {
    try {
      await datasetApi.delete(dataset.id)
      message.success(`已删除数据集 ${dataset.name}`)
      await fetchDatasets()
    } catch (error: any) {
      message.error(error.message || '删除数据集失败')
    }
  }

  const uploadProps: UploadProps = {
    name: 'file',
    accept: '.jsonl,.csv',
    showUploadList: false,
    beforeUpload: async (file) => {
      const values = form.getFieldsValue()
      const encoding = values.encoding || 'utf-8'
      setUploading(true)
      setUploadProgress(0)
      try {
        const validateResult = await datasetApi.validate(file, { encoding }) as any
        setValidation(validateResult)
        await datasetApi.upload(
          file,
          {
            name: values.name,
            description: values.description,
            encoding,
          },
          (progress) => setUploadProgress(progress),
        )
        message.success('数据集上传成功')
        form.resetFields(['name', 'description'])
        setValidation(null)
        await fetchDatasets()
      } catch (error: any) {
        message.error(error.message || '数据集上传失败')
      } finally {
        setUploading(false)
        setUploadProgress(0)
      }
      return false
    },
  }

  const columns = useMemo(
    () => [
      {
        title: '名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record: Dataset) => (
          <Space>
            <FileTextOutlined />
            <Text strong>{name}</Text>
            <Tag color={record.source === 'builtin' ? 'default' : 'blue'}>
              {record.source === 'builtin' ? '预置' : '运行时'}
            </Tag>
            {record.read_only && <Tag>只读</Tag>}
          </Space>
        ),
      },
      {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
        ellipsis: true,
        render: (value: string) => value || '-',
      },
      {
        title: '文件路径',
        dataIndex: 'file_path',
        key: 'file_path',
        render: (value: string) => <Text code>{value || '-'}</Text>,
      },
      {
        title: '类型',
        dataIndex: 'file_type',
        key: 'file_type',
        width: 100,
        render: (value: string) => <Tag>{(value || 'jsonl').toUpperCase()}</Tag>,
      },
      {
        title: '记录数',
        dataIndex: 'row_count',
        key: 'row_count',
        width: 110,
        render: (value: number) => <Tag color="blue">{value || 0}</Tag>,
      },
      {
        title: '编码',
        dataIndex: 'encoding',
        key: 'encoding',
        width: 100,
        render: (value: string) => value || 'utf-8',
      },
      {
        title: '操作',
        key: 'actions',
        width: 150,
        render: (_: unknown, record: Dataset) => (
          <Space>
            <Button type="text" size="small" icon={<EyeOutlined />} onClick={() => handlePreview(record.id)}>
              预览
            </Button>
            {!record.read_only && (
              <Popconfirm
                title={`确定删除 ${record.name} 吗？`}
                onConfirm={() => handleDelete(record)}
              >
                <Button type="text" size="small" danger icon={<DeleteOutlined />}>
                  删除
                </Button>
              </Popconfirm>
            )}
          </Space>
        ),
      },
    ],
    [],
  )

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>数据管理</Title>

      <Card style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          initialValues={{ encoding: 'utf-8' }}
          style={{ marginBottom: 16 }}
        >
          <Form.Item label="显示名称" name="name">
            <Input placeholder="可选，默认使用文件名生成数据集名称" />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={2} placeholder="可选，补充数据集来源、用途、筛选条件等信息" />
          </Form.Item>
          <Form.Item label="编码" name="encoding">
            <Input placeholder="utf-8" />
          </Form.Item>
        </Form>

        {uploading && uploadProgress > 0 ? (
          <div style={{ padding: 24 }}>
            <Progress percent={uploadProgress} status="active" />
            <Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0 }}>
              正在上传并分析数据集...
            </Paragraph>
          </div>
        ) : (
          <Dragger {...uploadProps} multiple={false} style={{ padding: 24 }}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 48, color: '#1677ff' }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
            <p className="ant-upload-hint">
              支持 `.jsonl` 和 `.csv`。运行时上传的数据集会保存在运行目录，不再写入项目仓库目录。
            </p>
          </Dragger>
        )}

        {validation && (
          <Card size="small" title="最近一次校验结果" style={{ marginTop: 16 }}>
            <Descriptions size="small" column={4}>
              <Descriptions.Item label="格式">{validation.file_type.toUpperCase()}</Descriptions.Item>
              <Descriptions.Item label="记录数">{validation.row_count}</Descriptions.Item>
              <Descriptions.Item label="编码">{validation.encoding}</Descriptions.Item>
              <Descriptions.Item label="字段">{validation.columns.join(', ') || '-'}</Descriptions.Item>
            </Descriptions>
            <div style={{ marginTop: 12 }}>
              <Text strong>预览</Text>
              <div style={{ marginTop: 12 }}>
                <DatasetPreviewList records={validation.preview_records || []} />
              </div>
            </div>
          </Card>
        )}
      </Card>

      <Card>
        <Table
          rowKey="id"
          loading={loading}
          columns={columns}
          dataSource={datasets}
          pagination={{ pageSize: 10 }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={(
                  <span>
                    <DatabaseOutlined style={{ marginRight: 8 }} />
                    暂无数据集
                  </span>
                )}
              />
            ),
          }}
        />
      </Card>

      <Modal
        title="数据集预览"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={920}
      >
        {previewLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
            <Spin />
          </div>
        ) : previewData ? (
          <div>
            <Descriptions size="small" column={3} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="标识">{previewData.id}</Descriptions.Item>
              <Descriptions.Item label="名称">{previewData.name}</Descriptions.Item>
              <Descriptions.Item label="文件">{previewData.file_path}</Descriptions.Item>
              <Descriptions.Item label="总记录数">{previewData.total_rows}</Descriptions.Item>
              <Descriptions.Item label="预览行数">{previewData.preview_rows}</Descriptions.Item>
              <Descriptions.Item label="字段">{(previewData.columns || []).join(', ') || '-'}</Descriptions.Item>
            </Descriptions>
            <DatasetPreviewList records={previewData.records || []} />
          </div>
        ) : (
          <Empty description="无法加载预览数据" />
        )}
      </Modal>
    </div>
  )
}
