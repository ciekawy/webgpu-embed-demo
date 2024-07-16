import { env, pipeline, Tensor } from '@xenova/transformers';
// import ortWasmThreaded from 'onnxruntime-web/dist/ort-wasm-simd-threaded.mjs';

// https://github.com/xenova/transformers.js/issues/735

let extractor: any = null;
let isExtractionRunning = false;

env.backends.onnx.backend = 'wasm';
// env.backends.onnx.backend = 'cpu';
// env.backends.onnx.backend = 'cpu';
// env.backends.onnx.backend = 'webgpu';
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.wasmPaths = "http://localhost:5173/node_modules/@xenova/transformers/dist/";
// env.backends.onnx.wasm.transformer = ortWasmThreaded;
env.allowLocalModels = true;
env.allowRemoteModels = false;
// env.backends.onnx.debug = true;
// env.backends.onnx.logLevel = 'verbose';

export async function loadExtractor() {
    if (!extractor) {
        console.log('Loading extractor...');
        extractor = await pipeline('feature-extraction', 'Xenova/bge-m3', {
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

