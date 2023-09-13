### Security Analysis of Localty-Sensitive Hashing-based Biometric Template Protection Schemes

- Seunghun Paik, Sunpill Kim and Jae Hong Seo
- Accepted at BMVC 2023 (Oral)

### 1. Prerequisite
- PyTorch
- CVXOPT
- Numpy
- etc. (e.g., matplotlib, tqdm, ...)

### 2. NbNet
We modified the NbNet according to the technique introduced in the supplementary material of IronMask, including perceptual loss based on angular distance. Implementation code can be found in the folder entitled `NbNet`.

Also, you can train the NbNet by referring to `KD` and `Train NbNet`. Prior to training NbNet, you should do knowledge distillation for constructing a face recognition model for calculating the perceptual loss.

### 3. Proposed Attack

In `Security Analysis on LSH-based BTPs`, we provide an examplar notebook for doing protection & attack on demo images.