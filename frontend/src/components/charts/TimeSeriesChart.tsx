import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin, Select } from 'antd'
import type { SelectProps } from 'antd'
import { useThemeStore } from '@/stores/themeStore'

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
  onMetricChange,
}: TimeSeriesChartProps) {
  const mode = useThemeStore((state) => state.mode)
  const isDark = mode === 'dark'

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
        containLabel: true,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        backgroundColor: isDark ? '#1f1f1f' : '#ffffff',
        borderColor: isDark ? '#424242' : '#d9d9d9',
        textStyle: {
          color: isDark ? '#ffffffd9' : '#1f1f1f',
        },
        formatter: (params: any) => {
          let result = params[0].axisValueLabel + '<br/>'
          params.forEach((param: any) => {
            const unit = data.find((d) => d.name === param.seriesName)?.unit || ''
            result += `${param.marker} ${param.seriesName}: ${param.value}${unit}<br/>`
          })
          return result
        },
      },
      legend: {
        data: data.map((d) => d.name),
        top: 0,
        textStyle: {
          color: isDark ? '#ffffffd9' : '#1f1f1f',
        },
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLabel: {
          color: isDark ? '#a6a6a6' : '#666666',
          formatter: (value: number) => {
            const date = new Date(value)
            return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`
          },
        },
        axisLine: {
          lineStyle: {
            color: isDark ? '#424242' : '#d9d9d9',
          },
        },
        splitLine: {
          lineStyle: {
            color: isDark ? '#303030' : '#f0f0f0',
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: isDark ? '#a6a6a6' : '#666666',
          formatter: (value: number) => value.toFixed(2),
        },
        axisLine: {
          lineStyle: {
            color: isDark ? '#424242' : '#d9d9d9',
          },
        },
        splitLine: {
          lineStyle: {
            color: isDark ? '#303030' : '#f0f0f0',
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          start: 0,
          end: 100,
          height: 20,
          bottom: 10,
          borderColor: isDark ? '#424242' : '#d9d9d9',
          fillerColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(22,119,255,0.1)',
          handleStyle: {
            color: isDark ? '#177ddc' : '#1677ff',
          },
          textStyle: {
            color: isDark ? '#a6a6a6' : '#666666',
          },
        },
      ],
      series: data.map((series, index) => ({
        name: series.name,
        type: 'line',
        smooth: true,
        data: series.data.map((d) => [d.timestamp, d.value]),
        itemStyle: {
          color: series.color || colors[index % colors.length],
        },
        areaStyle: {
          opacity: 0.1,
        },
        sampling: 'lttb',
      })),
    }
  }, [data, colors, isDark])

  if (loading) {
    return (
      <Card
        style={{
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        <div style={{ height, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    )
  }

  if (!data || data.length === 0) {
    return (
      <Card
        title={title}
        style={{
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        <Empty description="暂无数据" />
      </Card>
    )
  }

  const metricOptions: SelectProps['options'] = data.map((series) => ({
    label: series.name,
    value: series.name,
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
      style={{
        borderRadius: 12,
        border: '1px solid var(--color-border-secondary)',
      }}
    >
      <ReactECharts
        option={option}
        style={{ height }}
        opts={{ renderer: 'svg' }}
        theme={isDark ? 'dark' : undefined}
      />
    </Card>
  )
}
