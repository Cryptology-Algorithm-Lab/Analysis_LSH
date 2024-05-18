### Locality-Sensitive Hashing-based Biometric Template Protection Schemes are Fully Reversible!

- Seunghun Paik, Chanwoo Hwang, Sunpill Kim and Jae Hong Seo
- An extended version of `Security Analysis on LSH-based BTPs` Accepted at BMVC 2023 (Oral).

This repository provides the source code for our extended paper entitled "Locality-Sensitive Hashing-based Biometric Template Protection Schemes are Fully Reversible!", which is currently under review.

### Introduction & Summary of Changes

In BMVC'23, we presented a novel attack to break the irreversibility of some biometric template protections (BTPs) from locality-sensitive hashing (LSH). That is, with the modified NbNet, we successfully recovered faces from the protected templates that resemble the original ones used for enrollment. Although our attack is known to be superior to previous attack methodologies because we directly utilize the structural properties shared by several LSH-based BTPs, its applicability to other classes of LSH-based BTPs is still open.

In this extended version, we formalized and extended this attack by extensively investigating the structural properties of other known LSH-based BTP proposals. At last, we identified the class of LSH-based BTPs vulnerable to our attack called the `PMA-LSHs`. Thanks to this, we successfully cryptanalyzed several other locality-sensitive hashing-based face template protections that did not appear in BMVC'23.

The following schemes are the target LSH-based BTPs we successfully cryptanalyzed. The highlighted ones were not presented in the earlier version.
- BioHashing [1]
- GRP/URP-IoM [2]
- ABH [3]
- **Deep IoM** [4]
- **IMM** [5]
- **Jiang et al.** [6]

In addition, we also evaluated the *transferability* of the recovered faces to other face recognition systems. More precisely, we tested whether the recovered faces and the corresponding original ones were also recognized as the same person in other independently trained face recognition systems. If possible, then it would be strong evidence that the adversary can recover the representative faces of the target identities from the corresponding protected templates. More importantly, the adversary would exploit these recovered faces to impersonate victims' identities enrolled in other face recognition systems.

The following open-sourced and commercial APIs are used for evaluating the transferability. Note that this repository provides evaluation codes for open-sourced recognition models only. Experimental results for commercial APIs are solely presented in our submitted manuscript at this moment and will be released upon paper acceptance.

###### Open-Sourced Face Recognition Models 
- InsightFace [7]
- OpenSphere [8]
- Face-Transformer [9]

###### Commercial APIs
- AWS CompareFace [10]
- Tencent CompareFace [11]
 
### Prerequisite

Our source code is all written in PyTorch. See `requirements.txt` for checking all prerequisite libraries.

### Usage

Overall, this repository consists of the following five parts in the form of `ipynb` notebooks. You can reproduce all experimental results by successively running each notebook.
- Training modified NbNet
- Instantiating LSHs & Enrollment
- Proposed Attack
- Transfererability Analysis
- Visualization Results

We note that some parts of the source codes were forked from the above open-sourced face recognition models. In particular, many parts of utilities, e.g., loading the datasets, came from some parts of `InsightFace`.

Throughout this repository, we will mainly utilize two publicly available datasets, MS1M-V3 [12] dataset and the LFW [13] dataset, for training and evaluation, respectively. These datasets can be found in the following link:
- https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view

#### 1. Modified NbNet.ipynb

Throughout this paper, we utilized a variant of NbNet [14], which is a famous method for reconstructing faces from the corresponding feature vectors. More precisely, we slightly modified the original NbNet by (1) utilizing the structure of the StyleGAN, (2) using real faces (MS1M-V3) directly during training, and (3) replacing VGG16 in the perceptual loss with other face recognition models. For more information, we recommend checking the codes in `INV`.

The notebook `1. Modified NbNet.ipynb` contains an example code of training and evaluating the modified NbNet.

**REMARK** Since the reconstruction attack in our method can be replaced in a plug-and-play manner, it is possible to replace other recent reconstruction models, such as Vec2Face [15], Shahreza & Marcel [16], or Arc2Face [17].

#### 2. LSHs.ipynb

In this notebook, you can extract features through pre-trained face recognition models, initialize LSH-based BTPs through our implementation codes, enroll templates through these LSHs, and evaluate their performance.

Implementation codes for targeted LSH schemes are in `LSH.py`. All codes are written by following pseudocodes in their original papers. In addition, codes for performance evaluations are in `eval_tools.py`.

This notebook reproduces our experimental results in the first row of Table 3 in the submitted manuscript.

**Remark** To make use of *Deep IoM*, we need to train a simple neural network that serves a role as an LSH. To this end, we provide a training script and a simple evaluation code in the folder `DIOM`. Make sure that the DIoM network is well-trained before running this notebook!

**Also Remark** Because of the randomness during enrollment, the reproduced results would slightly differ from the reported values.

#### 3. Attack.ipynb

This notebook provides implementations of the proposed attack and a previous genetic algorithm-based attack [18] for comparison. All attack algorithms and techniques introduced in the manuscript are implemented in `eval_attack.py`.

This notebook reproduces our experimental results in the remaining rows of Table 3, with Table 4, in the submitted manuscript.

#### 4. Transfer Attack.ipynb

This notebook provides our *transferability* analyses on the recovered faces. We measured the ratio of faces recognized as the same person over the whole face pairs in the LFW dataset from the same identity, replacing one side of each pair with the recovered one.

Because of the availability of the recognition model, we only provide the results from aforementioned open-sourced face recognition models.

This notebook reproduces our experimental results in Table 5 and Table 6, excluding results for commercial APIs in the submitted manuscript.

#### 5. Demo.ipynb

In this notebook, you can visualize the recovered faces from the stolen template protected by various LSH-based BTPs. For a comparison, we also provide the recovered faces from [18]. Faces in this notebook are randomly sampled from the LFW dataset. 

This notebook reproduces Figure 2 in the submitted manuscript.

### References

[1] A. T. B. Jin, D. N. C. Ling, and A. Goh, “Biohashing: two factor authentication featuring fingerprint data and tokenised random number,” Pattern recognition, vol. 37, no. 11, pp. 2245–2255, 2004.

[2] Z. Jin, J. Y. Hwang, Y.-L. Lai, S. Kim, and A. B. J. Teoh, “Rankingbased locality sensitive hashing-enabled cancelable biometrics: Indexof-max hashing,” IEEE Transactions on Information Forensics and Security, vol. 13, no. 2, pp. 393–407, 2017.

[3] Y. Lai, Z. Jin, K. Wong, and M. Tistarelli, “Efficient known-sample attack for distance-preserving hashing biometric template protection schemes,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 3170–3185, 2021.

[4] J. Cui and A. B. J. Teoh, “Deep index-of-maximum hashing for face template protection,” in 2020 5th International Conference on Computer and Communication Systems (ICCCS). IEEE, 2020, pp. 413–418.

[5] Y. Li, L. Pang, H. Zhao, Z. Cao, E. Liu, and J. Tian, “Indexing-min–max hashing: Relaxing the security–performance tradeoff for cancelable fingerprint templates,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 10, pp. 6314–6325, 2022.

[6] Y. Jiang, P. Shen, L. Zeng, X. Zhu, D. Jiang, and C. Chen, “Cancelable biometric schemes for euclidean metric and cosine metric,” Cybersecurity, vol. 6, no. 1, pp. 1–20, 2023

[7] J. Guo, J. Deng, X. An, J. Yu, and B. Gecer, “Insightface: 2d and 3d face analysis project,” [Online]. Available: https://github.com/deepinsight/insightface

[8] W. Liu and Y. Wen, “Opensphere,” [Online]. Available: https://github.com/ydwen/opensphere

[9] Y. Zhong and W. Deng, “Face transformer for recognition,” arXiv preprint arXiv:2103.14803, 2021

[10] "Amazon Rekognition," [Online]. Available: https://aws.amazon.com/rekognition/

[11] "Tencent Cloud," [Online]. Available: https://www.tencentcloud.com/

[12] J. Deng, J. Guo, Y. Zhou, J. Yu, I. Kotsia, and S. Zafeiriou, “Retinaface: Single-stage dense face localisation in the wild,” arXiv preprint arXiv:1905.00641, 2019

[13] G. B. Huang, M. Mattar, T. Berg, and E. Learned-Miller, “Labeled faces in the wild: A database forstudying face recognition in unconstrained environments,” in Workshop on faces in’Real-Life’Images: detection, alignment, and recognition, 2008.

[14] G. Mai, K. Cao, P. C. Yuen, and A. K. Jain, “On the reconstruction of face images from deep face templates,” IEEE transactions on pattern analysis and machine intelligence, vol. 41, no. 5, pp. 1188–1202, 2018.

[15] C. N. Duong, T.-D. Truong, K. Luu, K. G. Quach, H. Bui, and K. Roy, “Vec2face: Unveil human faces from their blackbox features in face recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 6132–6141.

[16] H. O. Shahreza and S. Marcel, “Face reconstruction from facial templates by learning latent space of a generator network,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[17] Papantoniou, F. P., Lattas, A., Moschoglou, S., Deng, J., Kainz, B., & Zafeiriou, S. (2024). Arc2Face: A Foundation Model of Human Faces. arXiv preprint arXiv:2403.11641.

[18] X. Dong, Z. Jin, and A. T. B. Jin, “A genetic algorithm enabled similarity-based attack on cancellable biometrics,” in 2019 IEEE 10th International Conference on Biometrics Theory, Applications and Systems (BTAS). IEEE, 2019, pp. 1–8.
