import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // Allow serving files from one level above web-ui (the monorepo root)
      allow: [path.resolve(__dirname, "..")],
    },
  },
});