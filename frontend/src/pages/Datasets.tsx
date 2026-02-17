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
} from 'antd'
import {
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileTextOutlined,
  DatabaseOutlined,
} from '@ant-design/icons'
import type { UploadProps } from 'antd'
import { datasetApi, Dataset } from '@/services/api'

const { Title, Text, Paragraph } = Typography

export default function Datasets() {
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewData, setPreviewData] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const response = await datasetApi.list() as any
      setDatasets(response || [])
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
      const response = await datasetApi.get(name, 20) as any
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
      setUploading(true)
      try {
        // Validate first
        const validateResult = await datasetApi.validate(file) as any
        if (!validateResult.valid) {
          message.error(`验证失败: ${validateResult.errors?.[0] || '未知错误'}`)
          return false
        }

        // Upload
        await datasetApi.upload(file)
        message.success('上传成功')
        fetchDatasets()
      } catch (error: any) {
        message.error(error.message || '上传失败')
      } finally {
        setUploading(false)
      }
      return false
    },
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
      title: '记录数',
      dataIndex: 'record_count',
      key: 'record_count',
      render: (count: number) => <Tag color="blue">{count} 条</Tag>,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatBytes(size),
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format: string) => <Tag>{format.toUpperCase()}</Tag>,
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>数据集管理</Title>
        <Upload {...uploadProps}>
          <Button type="primary" icon={<UploadOutlined />} loading={uploading}>
            上传数据集
          </Button>
        </Upload>
      </div>

      <Paragraph type="secondary">
        上传 JSONL 格式的测试数据集，每行一个测试用例
      </Paragraph>

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
                    暂无数据集，点击上方按钮上传
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
              <Descriptions.Item label="总记录数">{previewData.total_records}</Descriptions.Item>
              <Descriptions.Item label="预览数量">{previewData.preview_count}</Descriptions.Item>
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
