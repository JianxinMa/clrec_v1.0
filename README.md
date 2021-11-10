# CLRec: Contrastive Learning for RECommendation

Implementation in TensorFlow 1.x
of **[Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems, KDD 2021](https://arxiv.org/abs/2005.12964)**
.

## Source Code - Single-Vector CLRec

### Running on a Single Machine

[clrec_v1_local_sasrec](clrec_v1_local_sasrec) is the version that we use for conducting the experiments on the public
datasets, which runs on a single machine. Note that this version uses the same neural architecture
as [SASRec](https://github.com/kang205/SASRec) in order to investigate the effect of the contrastive loss itself.

### Running on a Distributed Cluster

[clrec_v1_distributed](clrec_v1_distributed) is the version that we use for conducting the large-scale experiments in
our real-world production environment, which runs on the distributed clusters provided by our company's infrastructure.
This version uses the neural architecture described in the appendix of our paper.

## Source Code - Multi-Vector CLRec

[multi_interest_clrec](multi_interest_clrec/multclr_v2) contains the implementation of the multi-interest sequence
encoder, which can produce multiple vectors of a user for capturing the user's diverse interests.

## Datasets

### Public Datasets

The public datasets used in our paper are pre-processed and provided by [SASRec](https://github.com/kang205/SASRec).

### Sampled Data from Our Production Environment

Please find the description of the new dataset and its download link [here](clrec_v1_distributed/data). We could only
release a sampled anonymized subset of our production environment data, while we conducted the experiments in our paper
on the non-sampled large-scale data.
