import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  // plugins: [
  //   vue({
  //     template: {
  //       compilerOptions: {
  //         isCustomElement: (tag) => ['zingchart'].includes(tag),
  //       }
  //     }
  //   })
  // ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
