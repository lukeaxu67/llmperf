import { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Upload,
  Modal,
  message,
  Popconfirm,
  Tag,
  Descriptions,
  List,
  Empty,
  Spin,
  Progress,
} from 'antd'
import {
  DeleteOutlined,
  EyeOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  InboxOutlined,
} from '@ant-design/icons'
import type { UploadProps } from 'antd'
import { datasetApi, Dataset } from '@/services/api'

const { Dragger } = Upload
const { Title, Text, Paragraph } = Typography

export default function Datasets() {
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewData, setPreviewData] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const response = await datasetApi.list() as any
      // API returns { datasets: [...], total: number }
      setDatasets(response?.datasets || [])
    } catch (error: any) {
      message.error(error.message || '获取数据集列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handlePreview = async (name: string) => {
    setPreviewLoading(true)
    setPreviewVisible(true)
    try {
      const response = await datasetApi.preview(name, 20) as any
      setPreviewData(response)
    } catch (error: any) {
      message.error(error.message || '获取数据集预览失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleDelete = async (name: string) => {
    try {
      await datasetApi.delete(name)
      message.success('数据集已删除')
      fetchDatasets()
    } catch (error: any) {
      message.error(error.message || '删除失败')
    }
  }

  const uploadProps: UploadProps = {
    name: 'file',
    accept: '.jsonl',
    showUploadList: false,
    beforeUpload: async (file) => {
      setUploadProgress(0)
      setUploading(true)
      try {
        // Validate first
        const validateResult = await datasetApi.validate(file) as any
        if (!validateResult.valid) {
          message.error(`验证失败: ${validateResult.errors?.[0] || '未知错误'}`)
          return false
        }

        // Upload with progress
        await datasetApi.upload(file, (progress) => {
          setUploadProgress(progress)
        })
        message.success('上传成功')
        fetchDatasets()
        setUploadProgress(0)
      } catch (error: any) {
        message.error(error.message || '上传失败')
        setUploadProgress(0)
      } finally {
        setUploading(false)
      }
      return false
    },
  }

  const draggerProps: UploadProps = {
    ...uploadProps,
    multiple: false,
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <FileTextOutlined />
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (description: string) => description || '-',
    },
    {
      title: '记录数',
      dataIndex: 'record_count',
      key: 'record_count',
      render: (count: number, record: Dataset) => (
        <Tag color="blue">{count || record.row_count || 0} 条</Tag>
      ),
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number, record: Dataset) => formatBytes(size || record.file_size || 0),
    },
    {
      title: '类型',
      dataIndex: 'file_type',
      key: 'file_type',
      render: (fileType: string, record: Dataset) => (
        <Tag>{(fileType || record.format || 'jsonl').toUpperCase()}</Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (timestamp: number) => {
        if (!timestamp) return '-'
        return new Date(timestamp * 1000).toLocaleString('zh-CN')
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: Dataset) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handlePreview(record.name)}
          >
            预览
          </Button>
          <Popconfirm
            title="确定要删除此数据集吗？"
            onConfirm={() => handleDelete(record.name)}
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>数据集管理</Title>

      {/* Upload Area */}
      <Card style={{ marginBottom: 24 }}>
        {uploading && uploadProgress > 0 ? (
          <div style={{ padding: 24, textAlign: 'center' }}>
            <Progress percent={uploadProgress} status="active" />
            <Paragraph type="secondary" style={{ marginTop: 16 }}>
              正在上传数据集...
            </Paragraph>
          </div>
        ) : (
          <Dragger {...draggerProps} style={{ padding: 24 }}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 48, color: '#1677ff' }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
            <p className="ant-upload-hint">
              支持 JSONL 格式的测试数据集，每行一个测试用例
            </p>
          </Dragger>
        )}
      </Card>

      <Card>
        <Table
          columns={columns}
          dataSource={datasets}
          rowKey="name"
          loading={loading}
          pagination={{ pageSize: 10 }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  <span>
                    <DatabaseOutlined style={{ marginRight: 8 }} />
                    暂无数据集，请上传数据集
                  </span>
                }
              />
            ),
          }}
        />
      </Card>

      {/* Preview Modal */}
      <Modal
        title="数据集预览"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={800}
      >
        {previewLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
            <Spin />
          </div>
        ) : previewData ? (
          <div>
            <Descriptions size="small" column={3} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="名称">{previewData.name}</Descriptions.Item>
              <Descriptions.Item label="总记录数">{previewData.total_rows}</Descriptions.Item>
              <Descriptions.Item label="预览数量">{previewData.preview_rows}</Descriptions.Item>
            </Descriptions>

            <Title level={5}>数据示例</Title>
            <List
              size="small"
              bordered
              dataSource={previewData.records?.slice(0, 5) || []}
              renderItem={(record: any, index: number) => (
                <List.Item>
                  <div style={{ width: '100%' }}>
                    <Text strong>ID: {record.id || index}</Text>
                    <div style={{ marginTop: 8 }}>
                      {record.messages?.map((msg: any, i: number) => (
                        <Tag key={i} color={msg.role === 'user' ? 'blue' : msg.role === 'assistant' ? 'green' : 'default'}>
                          {msg.role}: {msg.content?.slice(0, 50)}...
                        </Tag>
                      ))}
                    </div>
                  </div>
                </List.Item>
              )}
            />
          </div>
        ) : (
          <Empty description="无法加载预览数据" />
        )}
      </Modal>
    </div>
  )
}
