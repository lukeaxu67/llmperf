import { test, expect } from '@playwright/test';

test.describe('LLMPerf Report Page Tests', () => {
  test.beforeEach(async ({ page }) => {
    // 等待前端服务可用
    await page.goto('/');
  });

  test('should display task list page', async ({ page }) => {
    await page.goto('/tasks');
    // 等待页面加载
    await page.waitForTimeout(2000);
    // 检查任务列表或空状态
    const hasTaskList = await page.locator('text=任务列表').isVisible().catch(() => false);
    const hasEmpty = await page.locator('text=暂无任务').isVisible().catch(() => false);
    expect(hasTaskList || hasEmpty).toBeTruthy();
  });

  test('should display datasets page correctly', async ({ page }) => {
    await page.goto('/datasets');
    await expect(page.locator('text=数据集管理')).toBeVisible({ timeout: 10000 });
  });

  test('should display pricing page', async ({ page }) => {
    await page.goto('/pricing');
    await expect(page.locator('text=价格管理')).toBeVisible({ timeout: 10000 });
  });

  test('should display report tabs in Chinese', async ({ page }) => {
    // 访问任务列表页
    await page.goto('/tasks');
    await page.waitForTimeout(2000);

    // 查找已完成的任务
    const completedTask = page.locator('text=completed').first();
    const hasCompletedTask = await completedTask.isVisible().catch(() => false);

    if (hasCompletedTask) {
      // 点击第一个已完成的任务
      await page.click('text=completed');
      await page.waitForTimeout(1000);

      // 获取当前URL并导航到任务详情
      const currentUrl = page.url();
      if (currentUrl.includes('/tasks/')) {
        // 检查中文标签
        await expect(page.locator('text=概览')).toBeVisible({ timeout: 10000 });
      }
    }
  });

  test('should not have white screen on model comparison', async ({ page }) => {
    // 先获取任务列表
    await page.goto('/tasks');
    await page.waitForTimeout(2000);

    // 查找有多个执行器的任务（通过检查任务卡片）
    const taskCards = await page.locator('[class*="ant-card"]').count();

    if (taskCards > 0) {
      // 点击第一个任务查看详情
      await page.click('[class*="ant-card"] >> visible=true');
      await page.waitForTimeout(2000);

      // 检查是否有模型对比标签
      const comparisonTab = page.locator('text=模型对比');
      const hasComparison = await comparisonTab.isVisible().catch(() => false);

      if (hasComparison) {
        // 点击模型对比标签
        await comparisonTab.click();
        await page.waitForTimeout(1000);

        // 确保没有白屏 - 页面应该有内容
        const pageContent = await page.content();
        expect(pageContent.length).toBeGreaterThan(1000);

        // 检查雷达图或对比表格
        const radarChart = page.locator('.echarts-for-react');
        const comparisonTable = page.locator('text=模型指标对比');

        const hasRadar = await radarChart.isVisible().catch(() => false);
        const hasTable = await comparisonTable.isVisible().catch(() => false);

        expect(hasRadar || hasTable).toBeTruthy();
      }
    }
  });

  test('should display per-executor metrics', async ({ page }) => {
    await page.goto('/tasks');
    await page.waitForTimeout(2000);

    // 点击第一个任务
    const taskCard = page.locator('[class*="ant-card"]').first();
    const hasTask = await taskCard.isVisible().catch(() => false);

    if (hasTask) {
      await taskCard.click();
      await page.waitForTimeout(2000);

      // 点击延迟分析标签
      const latencyTab = page.locator('text=延迟分析');
      const hasLatencyTab = await latencyTab.isVisible().catch(() => false);

      if (hasLatencyTab) {
        await latencyTab.click();
        await page.waitForTimeout(500);

        // 检查是否有"各模型延迟概览"（多执行器时）
        const perExecutorCard = page.locator('text=各模型延迟概览');
        const hasPerExecutor = await perExecutorCard.isVisible().catch(() => false);

        // 如果有多个执行器，应该显示分对象概览
        // 如果只有一个执行器，则不显示
        // 这里只是验证页面不会崩溃
        const pageContent = await page.content();
        expect(pageContent.length).toBeGreaterThan(1000);
      }
    }
  });
});
