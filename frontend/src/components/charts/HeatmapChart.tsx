import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin, Typography } from 'antd'
import { useThemeStore } from '@/stores/themeStore'

const { Text } = Typography

interface TimeFrameData {
  period: string
  metric: string
  value: number
  unit?: string
}

interface HeatmapChartProps {
  data?: TimeFrameData[]
  periods?: string[]
  metrics?: string[]
  loading?: boolean
  title?: string
  height?: number
}

const defaultPeriods = ['0-6点', '6-12点', '12-18点', '18-24点']
const defaultMetrics = ['TTFT', '总响应时间', '速率', '成功率']

export default function HeatmapChart({
  data = [],
  periods = defaultPeriods,
  metrics = defaultMetrics,
  loading = false,
  title = '时段热力图分析',
  height = 350,
}: HeatmapChartProps) {
  const mode = useThemeStore((state) => state.mode)
  const isDark = mode === 'dark'

  const { heatmapData, minValue, maxValue } = useMemo(() => {
    if (data.length === 0) {
      return { heatmapData: [], minValue: 0, maxValue: 100 }
    }

    const formattedData = data.map((d) => {
      const periodIndex = periods.indexOf(d.period)
      const metricIndex = metrics.indexOf(d.metric)
      return [periodIndex, metricIndex, d.value || 0]
    })

    const values = data.map((d) => d.value)
    return {
      heatmapData: formattedData,
      minValue: Math.min(...values),
      maxValue: Math.max(...values),
    }
  }, [data, periods, metrics])

  const option = useMemo(
    () => ({
      grid: {
        height: '70%',
        top: '15%',
      },
      tooltip: {
        position: 'top',
        backgroundColor: isDark ? '#1f1f1f' : '#ffffff',
        borderColor: isDark ? '#424242' : '#d9d9d9',
        textStyle: {
          color: isDark ? '#ffffffd9' : '#1f1f1f',
        },
        formatter: (params: any) => {
          const period = periods[params.value[0]]
          const metric = metrics[params.value[1]]
          const value = params.value[2]
          return `<div style="padding:4px 8px;">
            <div style="font-weight:500;">${period} - ${metric}</div>
            <div>值: ${value}</div>
          </div>`
        },
      },
      xAxis: {
        type: 'category',
        data: periods,
        splitArea: {
          show: true,
          areaStyle: {
            color: isDark ? ['#1f1f1f', '#141414'] : ['#fafafa', '#ffffff'],
          },
        },
        axisLabel: {
          rotate: 0,
          color: isDark ? '#a6a6a6' : '#666666',
        },
        axisLine: {
          lineStyle: {
            color: isDark ? '#424242' : '#d9d9d9',
          },
        },
      },
      yAxis: {
        type: 'category',
        data: metrics,
        splitArea: {
          show: true,
          areaStyle: {
            color: isDark ? ['#1f1f1f', '#141414'] : ['#fafafa', '#ffffff'],
          },
        },
        axisLabel: {
          color: isDark ? '#a6a6a6' : '#666666',
        },
        axisLine: {
          lineStyle: {
            color: isDark ? '#424242' : '#d9d9d9',
          },
        },
      },
      visualMap: {
        min: minValue,
        max: maxValue,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: isDark
            ? ['#162612', '#274916', '#3e6f1e', '#49aa19', '#d48806', '#a61d24']
            : ['#d4f4dd', '#52c41a', '#faad14', '#f5222d'],
        },
        text: ['高', '低'],
        textStyle: {
          color: isDark ? '#a6a6a6' : '#666666',
        },
      },
      series: [
        {
          name: '时段指标',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: (params: any) => params.value[2]?.toFixed(1) || '',
            color: isDark ? '#ffffffd9' : '#1f1f1f',
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    }),
    [heatmapData, periods, metrics, minValue, maxValue, isDark]
  )

  if (loading) {
    return (
      <Card
        style={{
          borderRadius: 12,
          border: '1px solid var(--color-border-secondary)',
        }}
      >
        <div
          style={{ height, display: 'flex', justifyContent: 'center', alignItems: 'center' }}
        >
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

  return (
    <Card
      title={title}
      style={{
        borderRadius: 12,
        border: '1px solid var(--color-border-secondary)',
      }}
    >
      <ReactECharts option={option} style={{ height }} opts={{ renderer: 'svg' }} />
      <div style={{ marginTop: 12, textAlign: 'center' }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          热力图显示不同时段各指标的表现，颜色越红表示数值越高，颜色越绿表示数值越低
        </Text>
      </div>
    </Card>
  )
}
