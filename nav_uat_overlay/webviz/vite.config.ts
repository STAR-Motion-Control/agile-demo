import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  base: "/viz/",
  plugins: [vue()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/viz/api": "http://127.0.0.1:8008",
      "/viz/ws": {
        target: "ws://127.0.0.1:8008",
        ws: true,
      },
    },
  },
});
