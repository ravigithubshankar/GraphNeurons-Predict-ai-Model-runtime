# GraphNeurons-Predict-ai-Model-runtime

## Google - Fast or Slow? Predict AI Model Runtime

### Predict how fast an AI model runs



Timeline

Start: August 30, 2023

Close: November 18, 2023

## Description

## AI Compiler Background

An AI model can be represented as a graph:

  - **Nodes:** Tensor operations (e.g., matrix multiplication, convolution, etc.).
  - **Edges:** Tensors.


A compilation configuration controls how the compiler transforms the graph for specific optimization passes. Alice can control two types of configurations:

  - **Layout Configuration:** Specifies the dimension order of each input and output of an operation node in physical memory.
  - **Tile Configuration:** Controls the tile size of each fused subgraph.


By predicting the optimal configuration for a given graph, you can:

Improve the compiler's heuristic to select the best configuration automatically.

Enable AI models to run more efficiently, consuming less time and resources.

## Dataset

The dataset, TpuGraphs, contains performance prediction data on XLA HLO graphs running on TPUs v3. There are five data collections:

layout:xla:random

layout:xla:default

layout:nlp:random

layout:nlp:default

tile:xla
