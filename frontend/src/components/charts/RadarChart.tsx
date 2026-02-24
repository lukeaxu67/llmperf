import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card, Empty, Spin } from 'antd'

interface RadarData {
  name: string
  value: number[]
  maxValue?: number[]
}

interface RadarChartProps {
  data?: RadarData[]
  indicators?: string[]
  loading?: boolean
  title?: string
  height?: number
}

const defaultIndicators = ['延迟', '吞吐', '成功率', '成本效率', '稳定性']

export default function RadarChart({
  data = [],
  indicators = defaultIndicators,
  loading = false,
  title = '模型综合评分对比',
  height = 400
}: RadarChartProps) {
  const option = useMemo(() => {
    if (data.length === 0) {
      return {}
    }

    const maxValues = data.reduce((acc, item) => {
      item.value.forEach((v, i) => {
        acc[i] = Math.max(acc[i] || 0, v)
      })
      return acc
    }, {} as Record<number, number>)

    const radarIndicators = indicators.map((name, i) => ({
      name,
      max: maxValues[i] || 100
    }))

    const colors = ['#1677ff', '#52c41a', '#fa8c16', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96']

    return {
      legend: {
        data: data.map(d => d.name),
        top: 5
      },
      radar: {
        indicator: radarIndicators,
        radius: '65%',
        splitNumber: 4,
        axisName: {
          fontSize: 12
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        },
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(22, 119, 255, 0.05)', 'rgba(22, 119, 255, 0.1)']
          }
        }
      },
      series: [
        {
          type: 'radar',
          data: data.map((item, index) => ({
            value: item.value,
            name: item.name,
            itemStyle: {
              color: colors[index % colors.length]
            },
            areaStyle: {
              opacity: 0.2
            }
          }))
        }
      ]
    }
  }, [data, indicators])

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

  return (
    <Card title={title}>
      <ReactECharts option={option} style={{ height }} opts={{ renderer: 'svg' }} />
    </Card>
  )
}
