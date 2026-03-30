import { useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Layout, Menu, Button, Typography, Tooltip, Avatar, Space } from 'antd'
import {
  DashboardOutlined,
  RocketOutlined,
  DollarOutlined,
  DatabaseOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SunOutlined,
  MoonOutlined,
} from '@ant-design/icons'
import type { MenuProps } from 'antd'
import { useThemeStore } from '@/stores/themeStore'

const { Header, Sider, Content } = Layout
const { Text } = Typography

const menuItems: MenuProps['items'] = [
  {
    key: '/dashboard',
    icon: <DashboardOutlined />,
    label: '系统概览',
  },
  {
    key: '/tasks',
    icon: <RocketOutlined />,
    label: '任务管理',
  },
  {
    key: '/pricing',
    icon: <DollarOutlined />,
    label: '成本监控',
  },
  {
    key: '/datasets',
    icon: <DatabaseOutlined />,
    label: '数据管理',
  },
  {
    key: '/settings',
    icon: <SettingOutlined />,
    label: '系统设置',
  },
]

const pageTitles: Record<string, string> = {
  '/dashboard': '系统概览',
  '/tasks': '任务管理',
  '/tasks/create': '创建任务',
  '/pricing': '成本监控',
  '/datasets': '数据管理',
  '/settings': '系统设置',
}

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { mode, toggleTheme } = useThemeStore()

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key)
  }

  const getSelectedKey = () => {
    const path = location.pathname
    if (path.startsWith('/tasks/') && path !== '/tasks/create') {
      return '/tasks'
    }
    return path
  }

  const title = pageTitles[getSelectedKey()] || 'LLMPerf'

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        style={{
          background: 'var(--color-bg-container)',
          borderRight: '1px solid var(--color-border-secondary)',
          boxShadow: collapsed ? 'none' : '2px 0 8px rgba(0, 0, 0, 0.04)',
        }}
      >
        {/* Logo Area */}
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid var(--color-border-secondary)',
            margin: '0 12px',
          }}
        >
          <Space size={8}>
            <Avatar
              size={32}
              style={{
                background: 'var(--gradient-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <RocketOutlined style={{ color: '#fff', fontSize: 16 }} />
            </Avatar>
            {!collapsed && (
              <Text
                strong
                style={{
                  fontSize: 18,
                  background: 'var(--gradient-primary)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                LLMPerf
              </Text>
            )}
          </Space>
        </div>

        {/* Menu */}
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{
            borderRight: 0,
            background: 'transparent',
            marginTop: 8,
            padding: '0 8px',
          }}
        />

        {/* Bottom Actions */}
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '16px 12px',
            borderTop: '1px solid var(--color-border-secondary)',
            display: 'flex',
            justifyContent: collapsed ? 'center' : 'space-between',
            alignItems: 'center',
          }}
        >
          {!collapsed && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {mode === 'dark' ? '暗色模式' : '亮色模式'}
            </Text>
          )}
          <Tooltip title={collapsed ? (mode === 'dark' ? '切换亮色' : '切换暗色') : ''} placement="right">
            <Button
              type="text"
              icon={mode === 'dark' ? <SunOutlined /> : <MoonOutlined />}
              onClick={toggleTheme}
              style={{
                color: 'var(--color-text-secondary)',
              }}
            />
          </Tooltip>
        </div>
      </Sider>

      <Layout style={{ background: 'var(--color-bg-layout)' }}>
        <Header
          style={{
            padding: '0 24px',
            background: 'var(--color-bg-container)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid var(--color-border-secondary)',
            height: 64,
          }}
        >
          <Space size="middle">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{
                fontSize: 16,
                color: 'var(--color-text-secondary)',
              }}
            />
            <Text
              strong
              style={{
                fontSize: 16,
                color: 'var(--color-text)',
              }}
            >
              {title}
            </Text>
          </Space>
        </Header>

        <Content
          style={{
            margin: 24,
            padding: 24,
            background: 'var(--color-bg-container)',
            borderRadius: 12,
            minHeight: 'calc(100vh - 112px)',
            overflow: 'auto',
            boxShadow: 'var(--shadow-card)',
          }}
          className="fade-in"
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
