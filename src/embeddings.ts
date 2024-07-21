import { env, pipeline, Tensor } from '@xenova/transformers';
// import ortWasmThreaded from 'onnxruntime-web/dist/ort-wasm-simd-threaded.mjs';

// https://github.com/xenova/transformers.js/issues/735

let extractor: any = null;
let isExtractionRunning = false;

env.backends.onnx.backend = 'wasm';
env.backends.onnx.wasm.proxy = false;
// env.backends.onnx.backend = 'cpu';
// env.backends.onnx.backend = 'cpu';
// env.backends.onnx.backend = 'webgpu';
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = navigator.maxTouchPoints <= 1;
env.backends.onnx.wasm.wasmPaths = "/webgpu-embed-demo/assets/";
// env.backends.onnx.wasm.wasmPaths = "http://localhost:5173/node_modules/@xenova/transformers/dist/";
// env.backends.onnx.wasm.transformer = ortWasmThreaded;
env.allowLocalModels = true;
env.allowRemoteModels = true;
// env.backends.onnx.debug = true;
// env.backends.onnx.logLevel = 'verbose';

export async function fetchWithProgress(url: string, onProgress: Function) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body!.getReader();
  const contentLength = +(response.headers?.get('Content-Length') ?? 0);
  let receivedLength = 0;

  const stream = new ReadableStream({
    start(controller) {
      function push() {
        reader.read().then(({ done, value }) => {
          if (done) {
            controller.close();
            return;
          }

          receivedLength += value.length;
          onProgress(receivedLength, contentLength);
          controller.enqueue(value);
          push();
        });
      }

      push();
    }
  });

  return new Response(stream);
}

// function interceptXHR() {
//   const originalOpen = XMLHttpRequest.prototype.open;
//   const originalSend = XMLHttpRequest.prototype.send;
//
//   XMLHttpRequest.prototype.open = function (method, url) {
//     this._url = url;
//     originalOpen.apply(this, arguments);
//   };
//
//   XMLHttpRequest.prototype.send = function () {
//     this.addEventListener('progress', (event) => {
//       if (event.lengthComputable && this._url.includes('path-to-your-model.onnx')) {
//         const progress = (event.loaded / event.total) * 100;
//         updateProgressBar(progress);
//       }
//     });
//
//     this.addEventListener('load', () => {
//       if (this._url.includes('path-to-your-model.onnx')) {
//         completeProgress();
//       }
//     });
//
//     originalSend.apply(this, arguments);
//   };
// }

export async function loadExtractor(progress_callback?: Function) {
    if (!extractor) {
        console.log('Loading extractor...');
        extractor = await pipeline('feature-extraction', 'Xenova/bge-m3', {
            progress_callback,
            session_options: {
                executionProviders: [
                  'wasm',
                // 'webgpu',
                // 'webgl',
                    // 'cpu'
                ]
            }
        });
        console.log('Extractor loaded.');
    }
    return extractor;
}

export async function computeEmbedding(text: string): Promise<number[] | null> {
    if (isExtractionRunning) {
        console.warn("Extraction already running. Please wait.");
        return null;
    }

    try {
        isExtractionRunning = true;
        const extractor = await loadExtractor();
        const start = +new Date();
        const embedding = await extractor(text, { pooling: 'mean', normalize: true });
        const embeddingArray = (embedding as Tensor).tolist()[0];
        console.log(`Computed Embedding (${+new Date - start}ms):`, embedding);
        return embeddingArray;
    } catch (error) {
        console.error("Failed to compute embedding: ", error);
        return null;
    } finally {
        isExtractionRunning = false;
    }
}

