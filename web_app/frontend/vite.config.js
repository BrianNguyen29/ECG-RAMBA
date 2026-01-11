import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Development Server Configuration
  server: {
    port: 5173,           // Fixed port for Frontend
    strictPort: true,     // Fail if port is already in use
    host: true,           // Expose to network (for Docker/remote testing)
    
    // Proxy API requests to Backend (avoids CORS in development)
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8003',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  
  // Preview (Production Build) Server
  preview: {
    port: 5173,
    strictPort: true,
  },
  
  // Test Configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
  },
})
