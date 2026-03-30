import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin, Typography } from 'antd'
import { useThemeStore } from '@/stores/themeStore'

const { Text } = Typography

interface LatencyData {
  values: number[]
  p50?: number
  p90?: number
  p95?: number
  p99?: number
  mean?: number
  std?: number
  cv?: number
}

interface LatencyDistributionChartProps {
  data?: LatencyData
  loading?: boolean
  title?: string
}

export default function LatencyDistributionChart({
  data,
  loading = false,
  title = '延迟分布',
}: LatencyDistributionChartProps) {
  const mode = useThemeStore((state) => state.mode)
  const isDark = mode === 'dark'

  const { histogramData, stats } = useMemo(() => {
    if (!data || data.values.length === 0) {
      return {
        histogramData: [],
        stats: {
          p50: 0,
          p90: 0,
          p95: 0,
          p99: 0,
          mean: 0,
          std: 0,
          cv: 0,
        },
      }
    }

    const sorted = [...data.values].sort((a, b) => a - b)
    const percentile = (ratio: number) =>
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * ratio))]
    const mean =
      data.mean ??
      data.values.reduce((sum, value) => sum + value, 0) / data.values.length
    const variance =
      data.std != null
        ? data.std * data.std
        : data.values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
          data.values.length
    const std = data.std ?? Math.sqrt(variance)
    const cv = data.cv ?? (mean === 0 ? 0 : std / mean)

    const min = Math.min(...data.values)
    const max = Math.max(...data.values)
    const binCount = Math.min(50, Math.max(10, Math.floor(Math.sqrt(data.values.length))))
    const binSize = (max - min) / binCount || 1

    const histogram: number[][] = []
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize
      const isLastBin = i === binCount - 1
      const count = data.values.filter((value) => {
        if (isLastBin) {
          return value >= binStart && value <= binStart + binSize
        }
        return value >= binStart && value < binStart + binSize
      }).length
      histogram.push([binStart, count])
    }

    return {
      histogramData: histogram,
      stats: {
        p50: data.p50 ?? percentile(0.5),
        p90: data.p90 ?? percentile(0.9),
        p95: data.p95 ?? percentile(0.95),
        p99: data.p99 ?? percentile(0.99),
        mean,
        std,
        cv,
      },
    }
  }, [data])

  const option = useMemo(
    () => ({
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
          return `<div style="padding:4px 8px;">
            <div style="font-weight:500;">${param.name} ms</div>
            <div>请求数: ${param.value[1]}</div>
          </div>`
        },
      },
      xAxis: {
        type: 'category',
        name: '延迟 (ms)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: isDark ? '#a6a6a6' : '#666666',
        },
        axisLabel: {
          rotate: 45,
          interval: 'auto',
          color: isDark ? '#a6a6a6' : '#666666',
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
      series: [
        {
          name: '请求分布',
          type: 'bar',
          data: histogramData,
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
          },
          large: true,
        },
      ],
      visualMap: {
        show: false,
        min: 0,
        max: Math.max(0, ...histogramData.map((item) => item[1])),
        inRange: {
          color: isDark
            ? ['#111d2c', '#177ddc', '#1765ad']
            : ['#c6e2ff', '#1677ff', '#0958d9'],
        },
      },
    }),
    [histogramData, isDark]
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
          style={{ height: 400, display: 'flex', justifyContent: 'center', alignItems: 'center' }}
        >
          <Spin size="large" />
        </div>
      </Card>
    )
  }

  if (!data || data.values.length === 0) {
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

  const statItems = [
    { label: 'P50', value: stats.p50.toFixed(2), unit: 'ms', color: '#1677ff' },
    { label: 'P90', value: stats.p90.toFixed(2), unit: 'ms', color: '#52c41a' },
    { label: 'P95', value: stats.p95.toFixed(2), unit: 'ms', color: '#fa8c16' },
    { label: 'P99', value: stats.p99.toFixed(2), unit: 'ms', color: '#ff4d4f' },
    {
      label: '极端慢请求',
      value: (
        (data.values.filter((value) => value > stats.p95 * 2).length / data.values.length) *
        100
      ).toFixed(2),
      unit: '%',
      color: '#722ed1',
    },
  ]

  return (
    <Card
      title={title}
      extra={
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          <span>均值: {stats.mean.toFixed(2)}ms</span>
          <span style={{ marginLeft: 16 }}>标准差: {stats.std.toFixed(2)}ms</span>
          <span style={{ marginLeft: 16 }}>CV: {stats.cv.toFixed(2)}</span>
        </div>
      }
      style={{
        borderRadius: 12,
        border: '1px solid var(--color-border-secondary)',
      }}
    >
      <ReactECharts option={option} style={{ height: 400 }} opts={{ renderer: 'svg' }} />
      <div
        style={{
          marginTop: 16,
          display: 'flex',
          justifyContent: 'space-around',
          flexWrap: 'wrap',
        }}
      >
        {statItems.map((item) => (
          <div key={item.label} style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {item.label}
            </Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: item.color }}>
              {item.value}
              <span style={{ fontSize: 12, marginLeft: 2 }}>{item.unit}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
