import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin } from 'antd'

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
  title = '寤惰繜鍒嗗竷',
}: LatencyDistributionChartProps) {
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
    const percentile = (ratio: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * ratio))]
    const mean = data.mean ?? (data.values.reduce((sum, value) => sum + value, 0) / data.values.length)
    const variance = data.std != null
      ? data.std * data.std
      : data.values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / data.values.length
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
      const count = data.values.filter((value) => (
        value >= binStart && (isLastBin ? value <= binStart + binSize : value < binStart + binSize)
      )).length
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

  const option = useMemo(() => ({
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%',
      top: '15%',
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
      },
      formatter: (params: any) => {
        let result = `${params[0].name} ms<br/>`
        params.forEach((param: any) => {
          result += `${param.marker} ${param.seriesName}: ${param.value[1]}<br/>`
        })
        return result
      },
    },
    xAxis: {
      type: 'category',
      name: '寤惰繜 (ms)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        interval: 'auto',
      },
    },
    yAxis: [
      {
        type: 'value',
        name: '棰戞',
        position: 'left',
      },
    ],
    series: [
      {
        name: '璇锋眰鍒嗗竷',
        type: 'bar',
        data: histogramData,
        itemStyle: {
          color: '#5470c6',
        },
        large: true,
      },
    ],
    visualMap: {
      show: false,
      min: 0,
      max: Math.max(0, ...histogramData.map((item) => item[1])),
      inRange: {
        color: ['#c6e2ff', '#1677ff', '#0958d9'],
      },
    },
  }), [histogramData])

  if (loading) {
    return (
      <Card>
        <div style={{ height: 400, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    )
  }

  if (!data || data.values.length === 0) {
    return (
      <Card title={title}>
        <Empty description="鏆傛棤鏁版嵁" />
      </Card>
    )
  }

  return (
    <Card
      title={title}
      extra={(
        <div style={{ fontSize: 12, color: '#666' }}>
          <span>鍧囧€? {stats.mean.toFixed(2)}ms </span>
          <span style={{ marginLeft: 16 }}>鏍囧噯宸? {stats.std.toFixed(2)}ms </span>
          <span style={{ marginLeft: 16 }}>CV: {stats.cv.toFixed(2)}</span>
        </div>
      )}
    >
      <ReactECharts option={option} style={{ height: 400 }} opts={{ renderer: 'svg' }} />
      <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P50</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#1677ff' }}>{stats.p50.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P90</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#52c41a' }}>{stats.p90.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P95</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#fa8c16' }}>{stats.p95.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P99</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#f5222d' }}>{stats.p99.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>鏋佺鎱㈣姹傛瘮</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#722ed1' }}>
            {(data.values.filter((value) => value > stats.p95 * 2).length / data.values.length * 100).toFixed(2)}%
          </div>
        </div>
      </div>
    </Card>
  )
}
