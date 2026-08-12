import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../web",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          map: ["maplibre-gl"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
    },
  },
});
