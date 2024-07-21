### Experiment: Running a Quantized BGE-M3 Embedding Model in the Browser for Full Privacy

This experiment aims to demonstrate running a quantized BGE-M3 embedding model directly in the browser, supporting multilingual and cross-lingual similarities while ensuring full privacy. The model runs entirely on the client side, avoiding any data transmission to external servers.

The implementation uses the `@xenova/transformers` and `onnxruntime-web` packages with a
BGE-M3 model that is 570MB in size after being 8-bit quantized.
