import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // No proxy needed — the built frontend is served by FastAPI (same origin).
    // For local hot-reload development against a separate uvicorn process, add:
    //   proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } }
  }
})
