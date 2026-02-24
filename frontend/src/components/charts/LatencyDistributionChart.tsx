import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin } from 'antd'

interface LatencyData {
  values: number[]
  p50: number
  p90: number
  p95: number
  p99: number
  mean: number
  std: number
  cv: number
}

interface LatencyDistributionChartProps {
  data?: LatencyData
  loading?: boolean
  title?: string
}

export default function LatencyDistributionChart({
  data,
  loading = false,
  title = '延迟分布'
}: LatencyDistributionChartProps) {
  const { histogramData } = useMemo(() => {
    if (!data || data.values.length === 0) {
      return { histogramData: [], boxPlotData: [] }
    }

    // Create histogram
    const min = Math.min(...data.values)
    const max = Math.max(...data.values)
    const binCount = Math.min(50, Math.max(10, Math.floor(Math.sqrt(data.values.length))))
    const binSize = (max - min) / binCount

    const histogram: number[][] = []
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize
      const count = data.values.filter(v => v >= binStart && v < binStart + binSize).length
      histogram.push([binStart, count])
    }

    return {
      histogramData: histogram
    }
  }, [data])

  const option = useMemo(() => ({
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%',
      top: '15%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: (params: any) => {
        let result = params[0].name + ' ms<br/>'
        params.forEach((param: any) => {
          result += `${param.marker} ${param.seriesName}: ${param.value[1]}<br/>`
        })
        return result
      }
    },
    xAxis: {
      type: 'category',
      name: '延迟 (ms)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        interval: 'auto'
      }
    },
    yAxis: [
      {
        type: 'value',
        name: '频次',
        position: 'left'
      }
    ],
    series: [
      {
        name: '请求分布',
        type: 'bar',
        data: histogramData,
        itemStyle: {
          color: '#5470c6'
        },
        large: true
      }
    ],
    visualMap: {
      show: false,
      min: 0,
      max: Math.max(...histogramData.map((d: any) => d[1])),
      inRange: {
        color: ['#c6e2ff', '#1677ff', '#0958d9']
      }
    }
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
        <Empty description="暂无数据" />
      </Card>
    )
  }

  return (
    <Card
      title={title}
      extra={
        <div style={{ fontSize: 12, color: '#666' }}>
          <span>均值: {data.mean.toFixed(2)}ms </span>
          <span style={{ marginLeft: 16 }}>标准差: {data.std.toFixed(2)}ms </span>
          <span style={{ marginLeft: 16 }}>CV: {data.cv.toFixed(2)}</span>
        </div>
      }
    >
      <ReactECharts option={option} style={{ height: 400 }} opts={{ renderer: 'svg' }} />
      <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P50</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#1677ff' }}>{data.p50.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P90</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#52c41a' }}>{data.p90.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P95</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#fa8c16' }}>{data.p95.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>P99</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#f5222d' }}>{data.p99.toFixed(2)}ms</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#666' }}>极端慢请求比</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#722ed1' }}>
            {(data.values.filter(v => v > data.p95 * 2).length / data.values.length * 100).toFixed(2)}%
          </div>
        </div>
      </div>
    </Card>
  )
}
