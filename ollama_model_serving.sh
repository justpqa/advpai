#!/usr/bin/env bash

echo "FROM $(pwd)/qwen2.5-3b-instruct-q8/qwen2.5-3b-instruct-q8_0.gguf" > Modelfile
ollama create qwen2.5-3b-instruct-q8 -f Modelfile
ollama run qwen2.5-3b-instruct-q8
ollama serve
# port 11434