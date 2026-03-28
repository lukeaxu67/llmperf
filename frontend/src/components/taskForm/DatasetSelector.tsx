/**
 * Dataset selection component
 * Allows users to select from available datasets and enter task description
 */

import { useEffect, useState } from 'react'
import { Radio, Card, Input, List, Spin, Empty, Tag, Space, Typography } from 'antd'
import { DatabaseOutlined, FileTextOutlined } from '@ant-design/icons'
import { datasetApi, Dataset } from '@/services/api'
import useTaskFormStore from '@/stores/taskFormStore'
import DatasetPreviewList from '@/components/DatasetPreviewList'

const { TextArea } = Input
const { Text } = Typography

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
      const response = await datasetApi.list() as any
      setDatasets(response?.datasets || [])
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedDataset || datasets.length === 0) {
      return
    }

    const matched = datasets.find((item) => item.id === selectedDataset)
      || datasets.find((item) => item.name === selectedDataset)
    if (matched && matched.id !== selectedDataset) {
      setSelectedDataset(matched.id, matched.file_path || null, matched.file_type || null)
    }
  }, [datasets, selectedDataset, setSelectedDataset])

  const fetchPreview = async (datasetId: string) => {
    setLoadingPreview(true)
    try {
      const response = await datasetApi.preview(datasetId, 5) as any
      setPreviewData(response?.records || [])
    } catch (error) {
      console.error('Failed to fetch preview:', error)
      setPreviewData([])
    } finally {
      setLoadingPreview(false)
    }
  }

  const handleSelectDataset = (datasetId: string | null) => {
    if (!datasetId) {
      setSelectedDataset(null, null, null)
      return
    }

    const dataset = datasets.find((item) => item.id === datasetId)
    setSelectedDataset(datasetId, dataset?.file_path || null, dataset?.file_type || null)
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
            onChange={(e) => handleSelectDataset(e.target.value)}
            style={{ width: '100%' }}
          >
            <List
              dataSource={datasets}
              renderItem={(dataset) => (
                <List.Item
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleSelectDataset(dataset.id)}
                >
                  <List.Item.Meta
                    avatar={
                      <Radio value={dataset.id} />
                    }
                    title={
                      <Space>
                        <FileTextOutlined />
                        <span>{dataset.name}</span>
                      </Space>
                    }
                    description={
                      <Space size="large" wrap>
                        <Text type="secondary">
                          {dataset.record_count || dataset.row_count || 0} 条记录
                        </Text>
                        <Text type="secondary">
                          {dataset.size ? ((dataset.size || 0) / 1024).toFixed(1) : '0'} KB
                        </Text>
                        <Tag>{(dataset.format || 'jsonl').toUpperCase()}</Tag>
                        <Text code>{dataset.file_path}</Text>
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
          ) : <DatasetPreviewList records={previewData} />}
        </Card>
      )}
    </div>
  )
}
