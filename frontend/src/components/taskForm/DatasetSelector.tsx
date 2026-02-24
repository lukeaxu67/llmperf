/**
 * Dataset selection component
 * Allows users to select from available datasets and enter task description
 */

import { useEffect, useState } from 'react'
import { Radio, Card, Input, List, Spin, Empty, Tag, Space, Typography } from 'antd'
import { DatabaseOutlined, FileTextOutlined } from '@ant-design/icons'
import { datasetApi, Dataset } from '@/services/api'
import useTaskFormStore from '@/stores/taskFormStore'

const { TextArea } = Input
const { Text, Paragraph } = Typography

export default function DatasetSelector() {
  const { selectedDataset, setSelectedDataset, taskDescription, setTaskDescription } = useTaskFormStore()

  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [previewData, setPreviewData] = useState<any[]>([])
  const [loadingPreview, setLoadingPreview] = useState(false)

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets()
  }, [])

  // Fetch preview when dataset is selected
  useEffect(() => {
    if (selectedDataset) {
      fetchPreview(selectedDataset)
    } else {
      setPreviewData([])
    }
  }, [selectedDataset])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const response = await datasetApi.list()
      setDatasets(response as unknown as Dataset[] || [])
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchPreview = async (name: string) => {
    setLoadingPreview(true)
    try {
      const response = await datasetApi.get(name, 5) as any
      setPreviewData(response?.records || response || [])
    } catch (error) {
      console.error('Failed to fetch preview:', error)
      setPreviewData([])
    } finally {
      setLoadingPreview(false)
    }
  }

  return (
    <div>
      <Card title="任务描述" style={{ marginBottom: 16 }}>
        <TextArea
          placeholder="输入任务描述（可选）"
          value={taskDescription}
          onChange={(e) => setTaskDescription(e.target.value)}
          rows={2}
          showCount
          maxLength={200}
        />
      </Card>

      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <span>选择数据集</span>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        {loading ? (
          <Spin tip="加载数据集..." />
        ) : datasets.length === 0 ? (
          <Empty description="暂无数据集，请先上传数据集" />
        ) : (
          <Radio.Group
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            style={{ width: '100%' }}
          >
            <List
              dataSource={datasets}
              renderItem={(dataset) => (
                <List.Item
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedDataset(dataset.name)}
                >
                  <List.Item.Meta
                    avatar={
                      <Radio value={dataset.name} />
                    }
                    title={
                      <Space>
                        <FileTextOutlined />
                        <span>{dataset.name}</span>
                      </Space>
                    }
                    description={
                      <Space size="large">
                        <Text type="secondary">
                          {dataset.record_count} 条记录
                        </Text>
                        <Text type="secondary">
                          {(dataset.size / 1024).toFixed(1)} KB
                        </Text>
                        <Tag>{dataset.format.toUpperCase()}</Tag>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Radio.Group>
        )}
      </Card>

      {selectedDataset && (
        <Card
          title="数据预览（前 5 条）"
          extra={
            <Tag color="blue">{previewData.length} 条记录</Tag>
          }
        >
          {loadingPreview ? (
            <Spin tip="加载预览..." />
          ) : previewData.length === 0 ? (
            <Empty description="无法加载预览数据" />
          ) : (
            <List
              size="small"
              dataSource={previewData}
              renderItem={(item, index) => (
                <List.Item>
                  <div style={{ width: '100%' }}>
                    <Tag>{index + 1}</Tag>
                    <Paragraph
                      ellipsis={{ rows: 2, expandable: true }}
                      style={{ marginBottom: 0, display: 'inline' }}
                    >
                      {typeof item === 'string' ? item : JSON.stringify(item)}
                    </Paragraph>
                  </div>
                </List.Item>
              )}
            />
          )}
        </Card>
      )}
    </div>
  )
}
