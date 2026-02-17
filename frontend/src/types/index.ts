// Type declarations for custom modules
declare module 'react-monaco-editor' {
  import { ComponentType } from 'react'

  interface MonacoEditorProps {
    width?: number | string
    height?: number | string
    value?: string
    defaultValue?: string
    language?: string
    theme?: string
    options?: any
    onChange?: (value: string) => void
    editorWillMount?: (monaco: any) => void
    editorDidMount?: (editor: any, monaco: any) => void
    editorWillUnmount?: (editor: any, monaco: any) => void
  }

  const MonacoEditor: ComponentType<MonacoEditorProps>
  export default MonacoEditor
}
