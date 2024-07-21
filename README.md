### Experiment: Running a Quantized BGE-M3 Embedding Model in the Browser for Full Privacy

This experiment aims to demonstrate running a quantized BGE-M3 embedding model directly in the browser (unfortunately
model is to big for mobile browsers), supporting multilingual and cross-lingual similarities while ensuring full
privacy. The model runs entirely on the client side, avoiding any data transmission to external servers.

Results are not ideal but are good enough for many use cases. I started this after taking many notes on some
broad topic and was not sure if some thought was already touched on previously. The original version was
implemented in Python, but later I got interested in seeing how it can be done in the browser.

The implementation uses the `@xenova/transformers` and `onnxruntime-web` packages with a
BGE-M3 model that is 570MB in size after being 8-bit quantized.

