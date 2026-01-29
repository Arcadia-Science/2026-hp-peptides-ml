# Stage 1 Lit Review: DetaNet + ELoRA + equivariant GNNs + Distributed Shampoo

## DetaNet (papers)
- DetaNet combines E(3)-equivariant tensor features with self-attention to predict molecular spectra and scalar/vector/tensor properties, demonstrated on QM9S and multiple spectra types. (Nature Computational Science, 2023)
  - https://www.nature.com/articles/s43588-023-00550-y
- TL-DetaNet applies transfer learning to scale IR/Raman spectra prediction to large polypeptides/proteins using datasets of amino acids/dipeptides/tripeptides. (J. Phys. Chem. Lett., 2025)
  - https://pubs.acs.org/doi/10.1021/acs.jpclett.5c00169

## DetaNet (local code in this repo)
- Core model: `capsule-3259363/code/detanet_model/detanet.py` (e3nn irreps + spherical harmonics; radius graph edges).
- Message passing: attention over scalar features + radial basis, then tensor products with SH to produce equivariant messages.
- Update: residual aggregation + tensor-product attention gate.
- Outputs: scalar/dipole/2nd- & 3rd-order tensors/R2 and derivative targets (forces/Hessians/polarizability).

## LoRA + ELoRA
- LoRA inserts low-rank adapters into linear layers while freezing base weights to reduce trainable parameters and memory. (Hu et al., 2021)
  - https://arxiv.gg/abs/2106.09685
- ELoRA extends low-rank adaptation to SO(3)-equivariant GNNs while preserving equivariance via a path-dependent decomposition; reports improvements on rMD17 and inorganic datasets. (ICML 2025)
  - Paper PDF: https://openreview.net/pdf/52a7ee49bb54485f26c4b012de71dd437a2d52f4.pdf
  - Poster page: https://icml.cc/virtual/2025/poster/44404
  - Code: https://github.com/hyjwpk/ELoRA

## Equivariant GNNs (relevant baselines)
- EGNN: E(n)-equivariant message passing without higher-order tensor reps; equivariant to rotations/translations/reflections/permutations. (ICML 2021)
  - https://proceedings.mlr.press/v139/satorras21a
- PaiNN: rotationally equivariant message passing for tensorial properties and spectra; improved data efficiency. (ICML 2021)
  - https://proceedings.mlr.press/v139/schutt21a.html
- SE(3)-Transformer: SE(3)-equivariant attention for 3D point clouds/graphs. (NeurIPS 2020)
  - https://fabianfuchsml.github.io/se3transformer/

## Contrastive / representation learning (optional auxiliary)
- GraphCL introduces contrastive pretraining with graph augmentations for robust GNN representations. (NeurIPS 2020)
  - https://neurips.cc/virtual/2020/poster/18375

## Distributed Shampoo optimizer
- Distributed Shampoo is a data-parallel PyTorch implementation of Shampoo (AdaGrad-family, Kronecker-factored block-diagonal preconditioners), using DTensor + AllGather for multi-GPU scaling. (arXiv 2309.06497)
  - https://huggingface.co/papers/2309.06497
- Reference implementation (Meta): https://github.com/facebookresearch/optimizers
