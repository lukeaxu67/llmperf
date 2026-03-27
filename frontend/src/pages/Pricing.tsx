import { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  InputNumber,
  DatePicker,
  message,
  Popconfirm,
  Row,
  Col,
  Statistic,
  Tag,
  Select,
  Empty,
  Tabs,
} from 'antd'
import {
  PlusOutlined,
  DeleteOutlined,
  DollarOutlined,
  HistoryOutlined,
  BarChartOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { Line } from '@ant-design/plots'
import { pricingApi, PricingRecord, CostSummary, TotalCost } from '@/services/api'
import { EXECUTOR_TYPES } from '@/types/taskConfig'

const { Title, Text } = Typography

export default function Pricing() {
  const [loading, setLoading] = useState(false)
  const [pricingList, setPricingList] = useState<PricingRecord[]>([])
  const [costSummary, setCostSummary] = useState<CostSummary[]>([])
  const [totalCost, setTotalCost] = useState<TotalCost | null>(null)
  const [pricingHistory, setPricingHistory] = useState<any[]>([])
  const [modalVisible, setModalVisible] = useState(false)
  const [form] = Form.useForm()
  const [providerFilter, setProviderFilter] = useState<string | undefined>()
  const [providers, setProviders] = useState<string[]>([])
  const providerOptions = Array.from(
    new Set([
      ...EXECUTOR_TYPES.filter((item) => item.value !== 'mock').map((item) => item.value),
      ...providers,
    ]),
  ).map((provider) => ({ label: provider, value: provider }))

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    try {
      const [pricingRes, costRes, totalRes, historyRes] = await Promise.all([
        pricingApi.list() as any,
        pricingApi.getCostSummary() as any,
        pricingApi.getTotalCost() as any,
        pricingApi.getHistory() as any,
      ])
      setPricingList(pricingRes.items || [])
      setCostSummary(costRes || [])
      setTotalCost(totalRes)
      setPricingHistory(historyRes || [])

      // Extract unique providers
      const uniqueProviders = [...new Set((pricingRes.items || []).map((p: PricingRecord) => p.provider))] as string[]
      setProviders(uniqueProviders)
    } catch (error: any) {
      message.error(error.message || '获取数据失败')
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = () => {
    form.resetFields()
    setModalVisible(true)
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      const providerValue = Array.isArray(values.provider) ? values.provider[0] : values.provider
      const effectiveAt = values.effective_at
        ? Math.floor(values.effective_at.valueOf() / 1000)
        : undefined

      await pricingApi.add({
        provider: providerValue,
        model: values.model,
        input_price: values.input_price,
        output_price: values.output_price,
        cache_read_price: values.cache_read_price || 0,
        cache_write_price: values.cache_write_price || 0,
        effective_at: effectiveAt,
        note: values.note || '',
      })
      message.success('价格记录已添加')
      setModalVisible(false)
      fetchData()
    } catch (error: any) {
      message.error(error.message || '添加失败')
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await pricingApi.delete(id)
      message.success('已删除')
      fetchData()
    } catch (error: any) {
      message.error(error.message || '删除失败')
    }
  }

  const pricingColumns = [
    {
      title: '厂商',
      dataIndex: 'provider',
      key: 'provider',
      width: 100,
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      width: 200,
    },
    {
      title: '输入价格',
      dataIndex: 'input_price',
      key: 'input_price',
      width: 120,
      render: (v: number) => <Text>¥{v.toFixed(2)}/M</Text>,
    },
    {
      title: '输出价格',
      dataIndex: 'output_price',
      key: 'output_price',
      width: 120,
      render: (v: number) => <Text>¥{v.toFixed(2)}/M</Text>,
    },
    {
      title: '生效时间',
      dataIndex: 'effective_at',
      key: 'effective_at',
      width: 150,
      render: (v: number) => dayjs.unix(v).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '备注',
      dataIndex: 'note',
      key: 'note',
      ellipsis: true,
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_: any, record: PricingRecord) => (
        <Popconfirm title="确定删除?" onConfirm={() => handleDelete(record.id)}>
          <Button type="text" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ]

  const costColumns = [
    {
      title: '厂商',
      dataIndex: 'provider',
      key: 'provider',
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
    },
    {
      title: '请求数',
      dataIndex: 'request_count',
      key: 'request_count',
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: '输入Tokens',
      dataIndex: 'total_input_tokens',
      key: 'total_input_tokens',
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: '输出Tokens',
      dataIndex: 'total_output_tokens',
      key: 'total_output_tokens',
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: '总费用',
      dataIndex: 'total_cost',
      key: 'total_cost',
      render: (v: number) => <Tag color="red">¥{v.toFixed(4)}</Tag>,
    },
  ]

  // Chart config for pricing history
  const chartData = pricingHistory.flatMap((item) => [
    {
      provider_model: `${item.provider}/${item.model}`,
      type: '输入价格',
      value: item.input_price,
      effective_at: dayjs.unix(item.effective_at).format('MM-DD'),
    },
    {
      provider_model: `${item.provider}/${item.model}`,
      type: '输出价格',
      value: item.output_price,
      effective_at: dayjs.unix(item.effective_at).format('MM-DD'),
    },
  ])

  const lineConfig = {
    data: chartData,
    xField: 'effective_at',
    yField: 'value',
    seriesField: 'provider_model',
    colorField: 'type',
    legend: { position: 'top' as const },
    smooth: true,
    point: { size: 3 },
    yAxis: {
      title: { text: '价格 (¥/百万tokens)' },
    },
  }

  return (
    <div>
      <Title level={4}>成本监控</Title>
      <Text type="secondary">管理各厂商各模型的API调用价格，监控成本消耗</Text>

      {/* Total Cost Summary */}
      <Row gutter={16} style={{ marginTop: 24, marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="累计总费用"
              value={totalCost?.total_cost || 0}
              precision={4}
              prefix={<DollarOutlined />}
              suffix="元"
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="任务总数"
              value={totalCost?.run_count || 0}
              suffix="个"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="平均任务费用"
              value={totalCost?.run_count ? (totalCost.total_cost / totalCost.run_count) : 0}
              precision={4}
              prefix="¥"
            />
          </Card>
        </Col>
      </Row>

      <Tabs
        defaultActiveKey="pricing"
        items={[
          {
            key: 'pricing',
            label: (
              <span>
                <DollarOutlined />
                价格管理
              </span>
            ),
            children: (
              <Card
                title="价格记录"
                extra={
                  <Space>
                    <Select
                      placeholder="筛选厂商"
                      allowClear
                      style={{ width: 150 }}
                      value={providerFilter}
                      onChange={setProviderFilter}
                      options={providers.map((p) => ({ value: p, label: p }))}
                    />
                    <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>
                      添加价格
                    </Button>
                  </Space>
                }
              >
                <Table
                  columns={pricingColumns}
                  dataSource={providerFilter ? pricingList.filter((p) => p.provider === providerFilter) : pricingList}
                  rowKey="id"
                  loading={loading}
                  pagination={{ pageSize: 10 }}
                  size="small"
                />
              </Card>
            ),
          },
          {
            key: 'history',
            label: (
              <span>
                <HistoryOutlined />
                价格走势
              </span>
            ),
            children: (
              <Card title="价格历史走势 (¥/百万tokens)">
                {chartData.length > 0 ? (
                  <Line {...lineConfig} />
                ) : (
                  <Empty description="暂无价格历史数据" />
                )}
              </Card>
            ),
          },
          {
            key: 'cost',
            label: (
              <span>
                <BarChartOutlined />
                费用统计
              </span>
            ),
            children: (
              <Card title="各模型费用统计 (近30天)">
                <Table
                  columns={costColumns}
                  dataSource={costSummary}
                  rowKey={(r) => `${r.provider}-${r.model}`}
                  loading={loading}
                  pagination={false}
                  summary={(data) => {
                    const total = data.reduce((acc, cur) => acc + cur.total_cost, 0)
                    return (
                      <Table.Summary.Row>
                        <Table.Summary.Cell index={0} colSpan={5}>
                          <Text strong>总计</Text>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={1}>
                          <Tag color="red">¥{total.toFixed(4)}</Tag>
                        </Table.Summary.Cell>
                      </Table.Summary.Row>
                    )
                  }}
                />
              </Card>
            ),
          },
        ]}
      />

      {/* Add Pricing Modal */}
      <Modal
        title="添加价格记录"
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
        width={500}
      >
        <Form form={form} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="provider"
                label="厂商"
                rules={[{ required: true, message: '请输入厂商名称' }]}
              >
                <Select
                  mode="tags"
                  options={providerOptions}
                  tokenSeparators={[',', ' ']}
                  placeholder="例如: openai, zhipu"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="model"
                label="模型"
                rules={[{ required: true, message: '请输入模型名称' }]}
              >
                <Input placeholder="例如: gpt-4o, glm-4" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="input_price"
                label="输入价格 (¥/百万tokens)"
                rules={[{ required: true, message: '请输入输入价格' }]}
              >
                <InputNumber
                  min={0}
                  step={0.01}
                  precision={2}
                  style={{ width: '100%' }}
                  placeholder="例如: 10.00"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="output_price"
                label="输出价格 (¥/百万tokens)"
                rules={[{ required: true, message: '请输入输出价格' }]}
              >
                <InputNumber
                  min={0}
                  step={0.01}
                  precision={2}
                  style={{ width: '100%' }}
                  placeholder="例如: 30.00"
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="cache_read_price" label="缓存读取价格 (¥/百万tokens)">
                <InputNumber
                  min={0}
                  step={0.01}
                  precision={2}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="cache_write_price" label="缓存写入价格 (¥/百万tokens)">
                <InputNumber
                  min={0}
                  step={0.01}
                  precision={2}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="effective_at" label="生效时间">
            <DatePicker showTime style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="note" label="备注">
            <Input.TextArea rows={2} placeholder="可选备注" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
