import { useMemo } from 'react'
import { Card, Table, Empty, Spin, Tag, Space } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons'

interface ModelMetric {
  model: string
  requests: number
  successRate: number
  avgTtft: number
  p95Ttft: number
  avgTps: number
  totalCost: number
  currency: string
  avgTokens: number
}

interface ComparisonTableProps {
  data?: ModelMetric[]
  loading?: boolean
  title?: string
  baselineModel?: string
}

export default function ComparisonTable({
  data = [],
  loading = false,
  title = '模型指标对比',
  baselineModel
}: ComparisonTableProps) {
  const { tableData, baseline } = useMemo(() => {
    if (!data || data.length === 0) {
      return { tableData: [], baseline: null }
    }

    const baselineItem = baselineModel
      ? data.find(d => d.model === baselineModel) || data[0]
      : data[0]

    const tableDataWithDiff = data.map(item => {
      const isBaseline = item.model === baselineItem.model

      // 安全计算差异，处理除零和空值情况
      const safeDiff = (current: number, baseline: number): number | null => {
        if (baseline === 0 || current == null || baseline == null) return null
        return ((current - baseline) / baseline) * 100
      }

      return {
        ...item,
        diff: isBaseline ? null : {
          successRate: safeDiff(item.successRate, baselineItem.successRate),
          avgTtft: safeDiff(item.avgTtft, baselineItem.avgTtft),
          p95Ttft: safeDiff(item.p95Ttft, baselineItem.p95Ttft),
          avgTps: safeDiff(item.avgTps, baselineItem.avgTps),
          totalCost: safeDiff(item.totalCost, baselineItem.totalCost),
          avgTokens: safeDiff(item.avgTokens, baselineItem.avgTokens)
        }
      }
    })

    return { tableData: tableDataWithDiff, baseline: baselineItem }
  }, [data, baselineModel])

  const renderDiff = (value: number | null, reverse = false) => {
    if (value === null) return <Tag color="blue">基准</Tag>

    // 处理NaN情况
    if (isNaN(value)) return <Tag color="default">-</Tag>

    const threshold = 5
    const absValue = Math.abs(value)

    if (absValue < threshold) {
      return <Tag icon={<MinusOutlined />} color="default">±{absValue.toFixed(1)}%</Tag>
    }

    const isGood = reverse ? value < 0 : value > 0
    const color = isGood ? 'green' : 'red'
    const icon = isGood ? <ArrowUpOutlined /> : <ArrowDownOutlined />

    return (
      <Tag icon={icon} color={color}>
        {value > 0 ? '+' : ''}{value.toFixed(1)}%
      </Tag>
    )
  }

  const columns: ColumnsType<any> = [
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      fixed: 'left',
      width: 150,
      render: (text, record) => (
        <Space>
          <span style={{ fontWeight: record.model === baseline?.model ? 'bold' : 'normal' }}>
            {text}
          </span>
          {record.model === baseline?.model && <Tag color="blue">基准</Tag>}
        </Space>
      )
    },
    {
      title: '请求数',
      dataIndex: 'requests',
      key: 'requests',
      width: 100,
      sorter: (a, b) => a.requests - b.requests
    },
    {
      title: '成功率',
      dataIndex: 'successRate',
      key: 'successRate',
      width: 120,
      sorter: (a, b) => a.successRate - b.successRate,
      render: (value, record) => (
        <Space>
          <span>{value != null ? value.toFixed(2) : '-'}%</span>
          {renderDiff(record.diff?.successRate ?? null)}
        </Space>
      )
    },
    {
      title: '平均TTFT',
      dataIndex: 'avgTtft',
      key: 'avgTtft',
      width: 120,
      sorter: (a, b) => a.avgTtft - b.avgTtft,
      render: (value, record) => (
        <Space>
          <span>{value != null ? value.toFixed(0) : '-'}ms</span>
          {renderDiff(record.diff?.avgTtft ?? null)}
        </Space>
      )
    },
    {
      title: 'P95 TTFT',
      dataIndex: 'p95Ttft',
      key: 'p95Ttft',
      width: 120,
      sorter: (a, b) => a.p95Ttft - b.p95Ttft,
      render: (value, record) => (
        <Space>
          <span>{value != null ? value.toFixed(0) : '-'}ms</span>
          {renderDiff(record.diff?.p95Ttft ?? null)}
        </Space>
      )
    },
    {
      title: '平均TPS',
      dataIndex: 'avgTps',
      key: 'avgTps',
      width: 120,
      sorter: (a, b) => a.avgTps - b.avgTps,
      render: (value, record) => (
        <Space>
          <span>{value != null ? value.toFixed(1) : '-'}</span>
          {renderDiff(record.diff?.avgTps ?? null)}
        </Space>
      )
    },
    {
      title: '平均Token',
      dataIndex: 'avgTokens',
      key: 'avgTokens',
      width: 120,
      sorter: (a, b) => a.avgTokens - b.avgTokens,
      render: (value, record) => (
        <Space>
          <span>{value != null ? value.toFixed(0) : '-'}</span>
          {renderDiff(record.diff?.avgTokens ?? null)}
        </Space>
      )
    },
    {
      title: '总成本',
      dataIndex: 'totalCost',
      key: 'totalCost',
      width: 120,
      sorter: (a, b) => a.totalCost - b.totalCost,
      render: (value, record) => (
        <Space>
          <span>{record.currency === 'CNY' ? '¥' : '$'}{value != null ? value.toFixed(4) : '-'}</span>
          {renderDiff(record.diff?.totalCost ?? null)}
        </Space>
      )
    }
  ]

  if (loading) {
    return (
      <Card>
        <div style={{ height: 200, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    )
  }

  if (!data || data.length === 0) {
    return (
      <Card title={title}>
        <Empty description="暂无数据" />
      </Card>
    )
  }

  return (
    <Card title={title}>
      <div style={{ marginBottom: 12, fontSize: 12, color: '#999' }}>
        基准模型: <strong>{baseline?.model}</strong> | 百分比差异表示相对于基准模型的变化
      </div>
      <Table
        columns={columns}
        dataSource={tableData}
        rowKey="model"
        pagination={false}
        scroll={{ x: 'max-content' }}
        size="small"
      />
    </Card>
  )
}
