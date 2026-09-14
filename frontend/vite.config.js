import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

/**
 * 步骤：01
 * SSE 关掉代理缓冲，避免 token 攒到请求结束才一次性到达浏览器。
 */
function disableSseBuffer_step_01(proxy) {
  proxy.on("proxyRes", (proxyRes, _req, res) => {
    const ct = String(proxyRes.headers["content-type"] || "");
    if (!ct.includes("text/event-stream")) return;
    proxyRes.headers["cache-control"] = "no-cache, no-transform";
    proxyRes.headers["x-accel-buffering"] = "no";
    delete proxyRes.headers["content-encoding"];
    if (typeof res.flushHeaders === "function") res.flushHeaders();
  });
}

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: disableSseBuffer_step_01,
      },
    },
  },
});
