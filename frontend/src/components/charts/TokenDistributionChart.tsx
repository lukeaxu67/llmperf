import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin } from 'antd'
import { useThemeStore } from '@/stores/themeStore'

interface TokenData {
  model?: string
  values: number[]
}

interface TokenDistributionChartProps {
  data?: TokenData[]
  loading?: boolean
  title?: string
  height?: number
}

export default function TokenDistributionChart({
  data = [],
  loading = false,
  title = '输出Token分布',
  height = 400,
}: TokenDistributionChartProps) {
  const mode = useThemeStore((state) => state.mode)
  const isDark = mode === 'dark'

  const { seriesData } = useMemo(() => {
    if (data.length === 0) {
      return { seriesData: [] }
    }

    const allValues = data.flatMap((d) => d.values)
    const min = Math.min(...allValues)
    const max = Math.max(...allValues)

    // Create histogram bins for each model
    const binCount = 20
    const binSize = Math.ceil((max - min) / binCount) || 1

    const series = data.map((item) => {
      const histogram = new Array(binCount).fill(0)
      const binLabels: string[] = []

      item.values.forEach((v) => {
        const binIndex = Math.min(Math.floor((v - min) / binSize), binCount - 1)
        histogram[binIndex]++
      })

      for (let i = 0; i < binCount; i++) {
        const binStart = min + i * binSize
        const binEnd = min + (i + 1) * binSize
        binLabels.push(`${binStart}-${binEnd}`)
      }

      return {
        name: item.model || '全部',
        data: histogram.map((count, i) => ({
          value: [i, count],
          label: binLabels[i],
        })),
      }
    })

    return { seriesData: series }
  }, [data])

  const option = useMemo(() => {
    if (seriesData.length === 0) {
      return {}
    }

    const colors = ['#1677ff', '#52c41a', '#fa8c16', '#f5222d', '#722ed1']

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
          type: 'shadow',
        },
        backgroundColor: isDark ? '#1f1f1f' : '#ffffff',
        borderColor: isDark ? '#424242' : '#d9d9d9',
        textStyle: {
          color: isDark ? '#ffffffd9' : '#1f1f1f',
        },
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          const series = seriesData.find((s) => s.name === param.seriesName)
          const label = series?.data[param.dataIndex]?.label || ''
          return `<div style="padding:4px 8px;">
            <div style="font-weight:500;">${param.seriesName}</div>
            <div>Token数: ${label}</div>
            <div>频次: ${param.value[1]}</div>
          </div>`
        },
      },
      legend: {
        data: seriesData.map((s) => s.name),
        top: 0,
        textStyle: {
          color: isDark ? '#ffffffd9' : '#1f1f1f',
        },
      },
      xAxis: {
        type: 'category',
        name: '输出Token数',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: isDark ? '#a6a6a6' : '#666666',
        },
        axisLabel: {
          rotate: 45,
          interval: 0,
          color: isDark ? '#a6a6a6' : '#666666',
          formatter: (value: number) => {
            const label = seriesData[0]?.data[value]?.label || ''
            return label.includes('-') ? label.split('-')[0] : label
          },
        },
        axisLine: {
          lineStyle: {
            color: isDark ? '#424242' : '#d9d9d9',
          },
        },
      },
      yAxis: {
        type: 'value',
        name: '频次',
        nameTextStyle: {
          color: isDark ? '#a6a6a6' : '#666666',
        },
        axisLabel: {
          color: isDark ? '#a6a6a6' : '#666666',
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
      series: seriesData.map((series, index) => ({
        name: series.name,
        type: 'bar',
        data: series.data.map((d) => d.value),
        itemStyle: {
          color: colors[index % colors.length],
          borderRadius: [4, 4, 0, 0],
        },
        large: true,
      })),
    }
  }, [seriesData, isDark])

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

  const allValues = data.flatMap((d) => d.values)
  const avg = allValues.reduce((a, b) => a + b, 0) / allValues.length
  const sorted = [...allValues].sort((a, b) => a - b)
  const p50 = sorted[Math.floor(sorted.length * 0.5)]
  const p90 = sorted[Math.floor(sorted.length * 0.9)]

  return (
    <Card
      title={title}
      extra={
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          <span>平均: {avg.toFixed(0)}</span>
          <span style={{ marginLeft: 16 }}>P50: {p50.toFixed(0)}</span>
          <span style={{ marginLeft: 16 }}>P90: {p90.toFixed(0)}</span>
        </div>
      }
      style={{
        borderRadius: 12,
        border: '1px solid var(--color-border-secondary)',
      }}
    >
      <ReactECharts option={option} style={{ height }} opts={{ renderer: 'svg' }} />
    </Card>
  )
}
