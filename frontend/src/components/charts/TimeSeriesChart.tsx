import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin, Select } from 'antd'
import type { SelectProps } from 'antd'

interface TimeSeriesDataPoint {
  timestamp: number
  value: number
}

interface MetricSeries {
  name: string
  data: TimeSeriesDataPoint[]
  unit?: string
  color?: string
}

interface TimeSeriesChartProps {
  data?: MetricSeries[]
  loading?: boolean
  title?: string
  height?: number
  showSelector?: boolean
  selectedMetric?: string
  onMetricChange?: (metric: string) => void
}

export default function TimeSeriesChart({
  data = [],
  loading = false,
  title = '时间序列',
  height = 400,
  showSelector = false,
  selectedMetric,
  onMetricChange
}: TimeSeriesChartProps) {
  const colors = ['#1677ff', '#52c41a', '#fa8c16', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96']

  const option = useMemo(() => {
    if (data.length === 0) {
      return {}
    }

    return {
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '10%',
        containLabel: true
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: (params: any) => {
          let result = params[0].axisValueLabel + '<br/>'
          params.forEach((param: any) => {
            const unit = data.find(d => d.name === param.seriesName)?.unit || ''
            result += `${param.marker} ${param.seriesName}: ${param.value}${unit}<br/>`
          })
          return result
        }
      },
      legend: {
        data: data.map(d => d.name),
        top: 0
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLabel: {
          formatter: (value: number) => {
            const date = new Date(value)
            return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`
          }
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => value.toFixed(2)
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100,
          height: 20,
          bottom: 10
        }
      ],
      series: data.map((series, index) => ({
        name: series.name,
        type: 'line',
        smooth: true,
        data: series.data.map(d => [d.timestamp, d.value]),
        itemStyle: {
          color: series.color || colors[index % colors.length]
        },
        areaStyle: {
          opacity: 0.1
        },
        sampling: 'lttb'
      }))
    }
  }, [data, colors])

  if (loading) {
    return (
      <Card>
        <div style={{ height, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
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

  const metricOptions: SelectProps['options'] = data.map(series => ({
    label: series.name,
    value: series.name
  }))

  return (
    <Card
      title={title}
      extra={
        showSelector && (
          <Select
            style={{ width: 150 }}
            value={selectedMetric}
            onChange={onMetricChange}
            options={metricOptions}
          />
        )
      }
    >
      <ReactECharts option={option} style={{ height }} opts={{ renderer: 'svg' }} />
    </Card>
  )
}
