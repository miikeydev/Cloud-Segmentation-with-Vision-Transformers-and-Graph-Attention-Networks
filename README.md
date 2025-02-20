# Cloud-Segmentation-with-Vision-Transformers-and-Graph-Attention-Networks

This repository contains a Kaggle notebook and Python code for cloud segmentation using a Vision Transformer (ViT) backbone to extract patch embeddings, a Graph Attention Network (GAT) module to refine features via a fully connected graph, and a decoder for upsampling to the original resolution. The code is modular and highly configurable via hyperparameters.

## Table of Contents

- [Usage](#usage)
- [Architecture Overview](#architecture-overview)
- [Experimental Results](#experimental-results)
- [Future Work](#future-work)
- [License](#license)

## Usage

The main pipeline is implemented in the Kaggle notebook. It includes:
- Data loading and preprocessing from pickle files.
- A modular segmentation model that combines a ViT backbone with a GAT module and an upsampling decoder.
- A training loop with mixed precision (AMP) and persistent progress bars.
- Evaluation and visualization of overlay predictions with a label legend.

To run the training pipeline, simply execute the Kaggle notebook.

## Architecture Overview

<div align="center">
  <img src="https://github.com/user-attachments/assets/65d7cce0-7d75-44c7-b414-c76b27f1d3fe" alt="Segmentation Architecture" width="80%">
</div>


This project is divided into two main components: the **Encoder** and the **Decoder**.

### Encoder

**Input & Preprocessing**  
The encoder receives raw images and prepares them for processing. It starts by normalizing the pixel values and extracting the required channels (using channels 3, 2, and 1 for RGB). The images have the shape (B, 3, IMG_H, IMG_W).

**Vision Transformer Backbone**  
The images are then passed through a patch embedding module that divides each image into fixed-size patches and projects each patch into an embedding space. These initial patch embeddings capture local information and are later saved as skip connections. Next, a Transformer Encoder refines these patch embeddings using multi-head self-attention, incorporating global context while preserving the spatial relationships among patches.

### Graph Construction & GAT Module

After the encoder, the refined patch embeddings are transformed into a graph representation. In this graph, each patch is treated as a node, and every node is connected to every other node (a fully connected graph). The similarity between any two nodes is computed using a dot product, which results in an affinity matrix that represents the pairwise relationships between patches.

The Graph Attention Network (GAT) module further refines these node features. It begins by linearly projecting each node’s feature into a higher-dimensional space. Then, it calculates attention scores between nodes using a small neural network (an MLP with LeakyReLU activation), normalizes these scores with a softmax function, and finally aggregates the features of neighboring nodes by weighting them according to the attention scores. This results in refined node features that capture more meaningful relationships.

### Decoder/Upsampling

The decoder reconstructs the segmentation map from the refined graph features and the skip connections from the encoder:

1. **Reshape to Grid:**  
   The refined node features from the GAT module, originally in a flat format, are reshaped into a 2D grid that mirrors the original patch layout.

2. **Reshape Skip Connections:**  
   The intermediate features (saved for skip connections) from the encoder are similarly reshaped into a grid.

3. **Feature Fusion & Convolution:**  
   The grid-form GAT features and the grid-form skip connection features are concatenated along the channel dimension. A convolutional layer (with a 3×3 kernel and ReLU activation) is then applied to fuse these features.

4. **Upsampling:**  
   The fused feature map is upsampled using bilinear interpolation to match the original image resolution.

5. **Final 1×1 Convolution:**  
   A final 1×1 convolution is applied to reduce the number of channels to the number of segmentation classes, resulting in the final segmentation map.

### Final Output

The final segmentation map is a tensor with shape (B, NUM_CLASSES, IMG_H, IMG_W), where each pixel is assigned a class score that can be used to generate a semantic segmentation mask.

## Experimental Results

The current model shows promising performance on cloud segmentation tasks:
- **Training Loss:** Converges over epochs with some fluctuations.
- **Validation Loss:** Generally decreases over time.
- **Qualitative Results:** The model correctly identifies thick clouds, thin clouds, and cloud shadows. Some misclassifications occur (for example, confusing clouds with similar ground features), which will be addressed in future iterations.

*Note:* Experiments are conducted on limited GPU resources (such as NVIDIA T4) and are considered experimental. Further improvements may include additional data augmentation, hyperparameter tuning, or architectural modifications.

## Future Work

- **Enhanced Decoder:** Experiment with more upsampling stages and additional residual connections to improve segmentation details.
- **Intermediate Skip Connections:** Extract and fuse more intermediate features from the ViT backbone.
- **Data Augmentation & Sampling:** Increase data sampling and augmentation techniques to improve generalization.
- **Benchmarking:** Compare the model against state-of-the-art segmentation architectures.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
