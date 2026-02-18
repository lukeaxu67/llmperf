import { useRef } from 'react'
import MonacoEditor from 'react-monaco-editor'
import { Alert } from 'antd'

interface YamlEditorProps {
  value: string
  onChange: (value: string) => void
  height?: number | string
  readOnly?: boolean
  error?: string
}

export default function YamlEditor({
  value,
  onChange,
  height = 400,
  readOnly = false,
  error,
}: YamlEditorProps) {
  const editorRef = useRef<any>(null)

  const editorDidMount = (editor: any) => {
    editorRef.current = editor
    editor.focus()
  }

  const options = {
    selectOnLineNumbers: true,
    roundedSelection: false,
    readOnly,
    cursorStyle: 'line' as const,
    automaticLayout: true,
    fontSize: 13,
    lineNumbers: 'on' as const,
    scrollBeyondLastLine: false,
    minimap: { enabled: false },
    folding: true,
    renderLineHighlight: 'all' as const,
    tabSize: 2,
    wordWrap: 'on' as const,
    theme: 'vs-light',
  }

  return (
    <div>
      {error && (
        <Alert
          message="配置错误"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 12 }}
        />
      )}
      <div className="monaco-editor-container">
        <MonacoEditor
          height={height}
          language="yaml"
          value={value}
          options={options}
          onChange={onChange}
          editorDidMount={editorDidMount}
        />
      </div>
    </div>
  )
}
