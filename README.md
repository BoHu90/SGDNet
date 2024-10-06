# SGDNet

[1] Hu, Bo, Jia Zheng, Leida Li, Ke Gu, Shuaijian Wang, Weisheng Li, and Xinbo Gao. "Blind Image Quality Index with High-Level Semantic Guidance and Low-Level Fine-Grained Representation." Neurocomputing 600 (2024): 128151.

# Abstract

Image Quality Assessment (IQA) has received unprecedented attention due to the extensive applications in benchmarking image processing algorithms and systems. Despite great progress in IQA, most previous frameworks either utilized only high-level semantic features or simply stacked multi-level features to account for the distortions, resulting in limited performance. The initial visual perception is formed after that the information is processed by the primary visual cortex and the advanced visual cortex. However, the information transmission of the cerebral cortex is not unidirectional, that is, the higher brain area can affect the primary brain area in turn, regulate its sensitivity and preference, so as to help the brain to understand the external world more finely. Inspired by this, we propose a novel blind image quality index with high-level Semantic Guidance and low-level fine-grained Representation (SGRNet). First, the versatile backbone, called pyramid vision Transformer, is used to simulate the multilevel information processing in the brain, generating multilevel feature representation. Second, we propose a novel high-level semantic guidance module to simulate the feedback mechanism between levels of the visual cortex. Third, a simple but effective fine-grained feature extraction module is proposed for high-level information compensation and lower-level content perception. After this, an attention mechanism-based enhancement module is proposed to further learn the above two features respectively. Finally, a bilinear pooling-based regression module is proposed to integrate the enhanced features of the two parts and map them to the quality score. Extensive experiments on six challenging public datasets show that the proposed SGRNet can deal well with both the simulated and authentic distortions and achieves state-of-the-art performance.

# Requirement

- numpy
- openpyxl
- pandas
- Pillow
- scipy
- timm
- torch
- torchvision

More information please check the requirements.txt.