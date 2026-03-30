import type { ThemeConfig } from 'antd'

export const darkTheme: ThemeConfig = {
  algorithm: undefined, // Will use theme.darkAlgorithm in App.tsx
  token: {
    colorPrimary: '#177ddc',
    colorSuccess: '#49aa19',
    colorWarning: '#d89614',
    colorError: '#a61d24',
    colorInfo: '#177ddc',
    colorBgContainer: '#141414',
    colorBgElevated: '#1f1f1f',
    colorBgLayout: '#0a0a0a',
    colorBgSpotlight: '#262626',
    colorText: '#ffffffd9',
    colorTextSecondary: '#a6a6a6',
    colorTextTertiary: '#737373',
    colorTextQuaternary: '#595959',
    colorBorder: '#424242',
    colorBorderSecondary: '#303030',
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 4,
    fontSize: 14,
    fontSizeHeading1: 38,
    fontSizeHeading2: 30,
    fontSizeHeading3: 24,
    fontSizeHeading4: 20,
    fontSizeHeading5: 16,
    fontFamily: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
      'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
      'Noto Color Emoji'`,
  },
  components: {
    Card: {
      borderRadiusLG: 12,
      boxShadowTertiary: '0 1px 2px rgba(0, 0, 0, 0.2), 0 1px 6px -1px rgba(0, 0, 0, 0.3)',
    },
    Menu: {
      itemBg: 'transparent',
      itemSelectedBg: '#111b26',
      itemSelectedColor: '#177ddc',
      darkItemBg: 'transparent',
      darkItemSelectedBg: '#111b26',
    },
    Table: {
      headerBg: '#1f1f1f',
      rowHoverBg: '#1f1f1f',
    },
    Layout: {
      siderBg: '#141414',
      headerBg: '#141414',
      bodyBg: '#0a0a0a',
    },
    Button: {
      borderRadius: 8,
      controlHeight: 36,
      defaultColor: '#ffffffd9',
      defaultBg: '#1f1f1f',
      defaultBorderColor: '#424242',
      primaryColor: '#ffffff',
    },
    Input: {
      borderRadius: 8,
    },
    Select: {
      borderRadius: 8,
    },
  },
}
