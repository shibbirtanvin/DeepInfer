# DeepInfer
This repository contains the reproducibility package, source code, benchmark, and results for the paper - "Inferring Data Preconditions from Deep Learning Models for Trustworthy Prediction in Deployment", which appeared in ICSE’2024: 2024 IEEE/ACM 46th International Conference on Software Engineering (ICSE ’24), April 14–20, 2024, Lisbon, Portugal.

The structure of this repository is below:

## Index
> 1. Models
> 2. Source Codes
  >> * Data precondition inference with a trained DNN model (inferDataPrecondition.py)
  >> * Unseen data prediction with DeepInfer (unseenPrediction.py)
  >> * Source codes of PIMA diabetes, House Price, German Credit, Bank Customer model prediction
  >> * ReadMe file to execute codes
> 3. Appendix with the derivation of wp rules

For reproducibility, we intend to release the complete artifact soon.

### Cite the paper as
```
@inproceedings{ahmed24deepinfer,
  author = {Shibbir Ahmed and Hongyang Gao and Hridesh Rajan},
  title = {Inferring Data Preconditions from Deep Learning Models for Trustworthy Prediction in Deployment},
  booktitle = {ICSE'2024: The 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  location = {Lisbon, Portugal},
  month = {April 14–20},
  year = {2024},
  entrysubtype = {conference},
  abstract = {
    Deep learning models are trained with certain assumptions about the data during the development stage and then used for prediction in the deployment stage. It is important to reason about the trustworthiness of the model’s predictions with unseen data during deployment. Existing methods for specifying and verifying traditional software are insufficient for this task, as they cannot handle the complexity of DNN model architecture and expected outcomes. In this work, we propose a novel technique that uses rules derived from neural network computations to infer data preconditions for a DNN model to determine the trustworthiness of its predictions. Our approach, DeepInfer involves introducing a novel abstraction for a trained DNN model that enables weakest precondition reasoning using Dijkstra’s Predicate Transformer Semantics. By deriving rules over the inductive type of neural network abstract representation, we can overcome the matrix dimensionality issues that arise from the backward non-linear computation from the output layer to the input layer. We utilize the weakest precondition computation using rules of each kind of activation function to compute layer-wise precondition from the given postcondition on the final output of a deep neural network. We extensively evaluated DeepInfer on 29 real-world DNN models using four different datasets collected from five different sources and demonstrated the utility, effectiveness, and performance improvement over closely related work. DeepInfer efficiently detects correct and incorrect predictions of high-accuracy models with high recall (0.98) and high F-1 score (0.84) and has significantly improved over the prior technique, SelfChecker. The average runtime overhead of DeepInfer is low, 0.22 sec for all the unseen datasets. We also compared runtime overhead using the same hardware settings and found that DeepInfer is 3.27 times faster than SelfChecker, the state-of-the-art in this area.
  }
}
```
