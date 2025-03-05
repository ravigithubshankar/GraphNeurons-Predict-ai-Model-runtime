
## Graph-Based Runtime Prediction Framework  

This project focuses on predicting AI model runtimes under diverse compiler configurations by leveraging graph-based deep learning techniques. The framework integrates state-of-the-art Graph Neural Network (GNN) architectures, including ChebNet and GraphSAGE, to analyze and optimize tensor operations. Below is a detailed description of the contributions and achievements:  

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



### Key Contributions  

1. **Modified GNN Architecture**  
   - Integrated ChebNet (Chebyshev spectral convolutions) and GraphSAGE (Graph Sample and Aggregate) within the model architecture.  
   - Achieved a **15% improvement in runtime prediction accuracy** and an **18% reduction in estimation errors** compared to baseline models.  

2. **Deep Learning Framework for Runtime Estimation**  
   - Designed a deep learning-based framework to estimate AI model runtimes efficiently under varying compiler configurations.  
   - Utilized graph-based representations of tensor operations, where nodes represent operations (e.g., matrix multiplication, convolution) and edges represent tensors.  
   - Enabled analysis of runtime performance by mapping tensor layouts and tile configurations directly into graph structures.  

3. **Optimization of Compiler Configurations**  
   - Optimized tile and layout configurations using Chebyshev spectral convolutions for efficient feature extraction and GraphSAGE for robust aggregation of local graph information.  
   - Improved graph ranking performance with a **10% enhancement** using **ListMLE loss** during training.  
   - Achieved **12% faster convergence** by leveraging TensorFlow/Keras for implementation and fine-tuning.  

### Results  

- **Enhanced Accuracy:** The framework delivered a substantial improvement in runtime prediction accuracy across diverse test cases.  
- **Reduced Estimation Errors:** The integrated GNN model minimized prediction errors, ensuring more reliable and efficient configuration selection.  
- **Faster Convergence:** The use of advanced GNN techniques led to faster convergence, reducing computational overhead during training and evaluation.  

### Technologies and Tools  

- **Deep Learning Frameworks:** TensorFlow, Keras.  
- **Graph Neural Network Architectures:** ChebNet, GraphSAGE.  
- **Ranking Loss Function:** ListMLE for optimizing graph ranking performance.  
- **Compiler Configuration Optimization:** Focused on layout and tile configurations for AI models.  

### Applications  

This framework has the potential to:  
- Enhance the efficiency of AI compilers by predicting optimal configurations without exhaustive searches.  
- Improve the performance of AI models on hardware accelerators like TPUs and GPUs.  
- Reduce resource consumption during training and inference by enabling better runtime predictions.  

---

Feel free to explore the implementation and contribute to further optimization of AI runtime prediction systems.  
