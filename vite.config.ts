import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import topLevelAwait from 'vite-plugin-top-level-await';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
    base: '/webgpu-embed-demo/',
    build: {
      outDir: 'dist',
      minify: false,
      rollupOptions: {
        input: {
          main: 'index.html',
          // sw: 'sw.ts'
        },
        output: {
          assetFileNames: (assetInfo) => {
            if (assetInfo.name?.endsWith('.wasm')) {
              return 'assets/[name][extname]';
            }
            return 'assets/[name].[hash][extname]';
          }
        }
        //   entryFileNames: (chunk) => {
        //     if (chunk.name === 'sw') {
        //       return 'sw.js';
        //     }
        //     return '[name].[hash].js';
        //   }
        // }
      }
    },
    optimizeDeps: {
      exclude: ['@xenova/transformers']
    },
    plugins: [
      wasm(), topLevelAwait(),
        viteStaticCopy({
          targets: [
            {
              src: 'node_modules/onnxruntime-web/dist/*.wasm',
              dest: 'wasm'
            }
          ]
        })
      ],
    server: {
      fs: {
        deny: ['models/*', 'models/Xenova/bge-m3/config.json', '/models/Xenova/bge-m3/config.json'],
      },
      }
});
