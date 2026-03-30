import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ConfigProvider, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import MainLayout from './layouts/MainLayout'
import Dashboard from './pages/Dashboard'
import Tasks from './pages/Tasks'
import TaskDetail from './pages/TaskDetail'
import CreateTask from './pages/CreateTask'
import Pricing from './pages/Pricing'
import Datasets from './pages/Datasets'
import Settings from './pages/Settings'
import { useThemeStore, initializeTheme } from './stores/themeStore'
import { lightTheme, darkTheme } from './styles/themes'

function App() {
  const mode = useThemeStore((state) => state.mode)
  const currentTheme = mode === 'dark' ? darkTheme : lightTheme

  useEffect(() => {
    initializeTheme()
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode)
  }, [mode])

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        ...currentTheme,
        algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
      }}
    >
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="tasks" element={<Tasks />} />
            <Route path="tasks/create" element={<CreateTask />} />
            <Route path="tasks/:id" element={<TaskDetail />} />
            <Route path="pricing" element={<Pricing />} />
            <Route path="datasets" element={<Datasets />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  )
}

export default App
