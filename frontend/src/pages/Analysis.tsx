import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Select,
  Typography,
  Empty,
  Spin,
  Tabs,
  Alert,
  Tag,
  List,
  Divider,
} from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  DollarOutlined,
  ClockCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { Line, Column, Pie } from '@ant-design/plots'
import StatCard from '@/components/StatCard'
import { analysisApi } from '@/services/api'

const { Title, Text } = Typography

export default function Analysis() {
  const { id } = useParams<{ id: string }>()
  const [loading, setLoading] = useState(false)
  const [summary, setSummary] = useState<any>(null)
  const [timeseries, setTimeseries] = useState<any>(null)
  const [anomalies, setAnomalies] = useState<any>(null)
  const [selectedMetric, setSelectedMetric] = useState('latency')

  useEffect(() => {
    if (id) {
      fetchAnalysisData(id)
    }
  }, [id])

  const fetchAnalysisData = async (runId: string) => {
    setLoading(true)
    try {
      // Fetch summary
      const summaryData = await analysisApi.getSummary(runId) as any
      setSummary(summaryData)

      // Fetch timeseries
      try {
        const tsData = await analysisApi.getTimeseries(runId, selectedMetric) as any
        setTimeseries(tsData)
      } catch (e) {
        // Ignore
      }

      // Fetch anomalies
      try {
        const anomalyData = await analysisApi.getAnomalies(runId) as any
        setAnomalies(anomalyData)
      } catch (e) {
        // Ignore
      }
    } catch (error) {
      console.error('Failed to fetch analysis data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (!id) {
    return (
      <Card>
        <Empty description="请从任务列表选择要分析的任务" />
      </Card>
    )
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!summary) {
    return (
      <Card>
        <Empty description="未找到分析数据" />
      </Card>
    )
  }

  const executorColumns = [
    { title: 'Executor', dataIndex: 'executor_id', key: 'executor_id' },
    {
      title: '请求数',
      dataIndex: 'total_requests',
      key: 'total_requests',
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (v: number) => (
        <Tag color={v > 95 ? 'green' : v > 80 ? 'orange' : 'red'}>
          {v.toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: '平均延迟',
      dataIndex: 'avg_first_resp_time',
      key: 'avg_first_resp_time',
      render: (v: number) => `${v.toFixed(0)}ms`,
    },
    {
      title: 'P95延迟',
      dataIndex: 'p95_first_resp_time',
      key: 'p95_first_resp_time',
      render: (v: number) => `${v.toFixed(0)}ms`,
    },
    {
      title: '费用',
      dataIndex: 'total_cost',
      key: 'total_cost',
      render: (v: number) => `${v.toFixed(4)}`,
    },
  ]

  // Prepare executor data for table
  const executorData = summary.by_executor
    ? Object.entries(summary.by_executor).map(([id, data]: [string, any]) => ({
        executor_id: id,
        ...data,
      }))
    : []

  // Time series chart config
  const lineConfig = {
    data: timeseries?.metrics?.[selectedMetric]?.data || [],
    xField: 'timestamp',
    yField: 'value',
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: selectedMetric,
        value: datum.value,
      }),
    },
    smooth: true,
  }

  // Anomaly list
  const anomalyList = anomalies?.anomalies || []

  return (
    <div>
      <Title level={4}>数据分析</Title>
      <Text type="secondary">Run ID: {id}</Text>

      {/* Summary Stats */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="总请求数"
            value={summary.total_requests}
            prefix={<DashboardOutlined />}
            color="#1677ff"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="成功率"
            value={summary.success_rate}
            suffix="%"
            prefix={<ThunderboltOutlined />}
            color={summary.success_rate > 95 ? '#52c41a' : '#ff4d4f'}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="平均延迟"
            value={summary.avg_first_resp_time}
            suffix="ms"
            prefix={<ClockCircleOutlined />}
            color="#722ed1"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <StatCard
            title="总费用"
            value={summary.total_cost}
            prefix={<DollarOutlined />}
            color="#fa8c16"
          />
        </Col>
      </Row>

      {/* Detailed Stats */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="延迟分布">
            <Row gutter={24}>
              <Col span={6}>
                <Statistic title="P50 延迟" value={summary.p50_first_resp_time?.toFixed(0) || '-'} suffix="ms" />
              </Col>
              <Col span={6}>
                <Statistic title="P95 延迟" value={summary.p95_first_resp_time?.toFixed(0) || '-'} suffix="ms" />
              </Col>
              <Col span={6}>
                <Statistic title="P99 延迟" value={summary.p99_first_resp_time?.toFixed(0) || '-'} suffix="ms" />
              </Col>
              <Col span={6}>
                <Statistic title="平均吞吐" value={summary.avg_token_throughput?.toFixed(1) || '-'} suffix="tok/s" />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Executor Comparison */}
      {executorData.length > 0 && (
        <Card title="执行器对比" style={{ marginTop: 24 }}>
          <Table
            columns={executorColumns}
            dataSource={executorData}
            rowKey="executor_id"
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {/* Time Series */}
      <Card
        title="性能趋势"
        style={{ marginTop: 24 }}
        extra={
          <Select
            value={selectedMetric}
            onChange={(v) => {
              setSelectedMetric(v)
              // Refetch timeseries
              if (id) {
                analysisApi.getTimeseries(id, v).then((data: any) => {
                  setTimeseries(data)
                })
              }
            }}
            options={[
              { value: 'latency', label: '延迟' },
              { value: 'throughput', label: '吞吐量' },
              { value: 'error_rate', label: '错误率' },
              { value: 'cost', label: '费用' },
            ]}
            style={{ width: 120 }}
          />
        }
      >
        {timeseries?.metrics?.[selectedMetric]?.data?.length > 0 ? (
          <Line {...lineConfig} />
        ) : (
          <Empty description="暂无时序数据" />
        )}
      </Card>

      {/* Anomalies */}
      {anomalyList.length > 0 && (
        <Card
          title={
            <span>
              <WarningOutlined style={{ color: '#faad14', marginRight: 8 }} />
              异常检测
            </span>
          }
          style={{ marginTop: 24 }}
        >
          <List
            itemLayout="horizontal"
            dataSource={anomalyList.slice(0, 10)}
            renderItem={(item: any) => (
              <List.Item>
                <List.Item.Meta
                  avatar={
                    <Tag color={item.severity === 'high' ? 'red' : item.severity === 'medium' ? 'orange' : 'blue'}>
                      {item.severity}
                    </Tag>
                  }
                  title={item.anomaly_type}
                  description={item.description}
                />
              </List.Item>
            )}
          />
        </Card>
      )}
    </div>
  )
}
