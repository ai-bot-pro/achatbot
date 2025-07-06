// vite.config.ts
import { defineConfig } from 'vite';

// 使用 defineConfig 辅助函数，提供类型支持
export default defineConfig({
    // 确保环境变量以 VITE_ 开头的会被暴露给客户端代码
    envPrefix: 'VITE_',

    // 如果需要，可以在这里添加其他 Vite 配置
    server: {
        port: 3000,
        open: true,
    },

    // 构建配置
    build: {
        outDir: 'dist',
        sourcemap: true,
    },
});