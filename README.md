# Peptimizer

This package consists of works from multiple publications based on supervised and unsupervised deep learning models for  optimizing peptides. The code is agnostic of the monomer, and can be used for a broad range of polymers with minimal changes.

## Publications
### Interpretable deep learning for <i>de novo</i> design of cell-penetrating abiotic polymers <a href='https://www.biorxiv.org/content/10.1101/2020.04.10.036566v1'> (link) </a>
The work done as a part of this paper is denoted with the prefix/suffix <u>cpp</u> for respective files. 
#### Usage
<b>Tutorial_CPP.ipynb</b> (Jupyter notebook) covers the training and usage of major functions (Generator, Predictor and Optimizer) and analysis of graident activations using a pre-trained model. This file can be used as-is with the current and customized datasets by providing appropriate filepath and parameters.

### Deep Learning for Prediction and Optimization of Rapid Flow Peptide Synthesis <a href='https://github.mit.edu/MLMat/peptimizer'> (coming soon) </a>
The work done as a part of this paper is denoted with the prefix/suffix <u>synthesis</u> for respective files. 
#### Usage
<b>Tutorial_Synthesis.ipynb</b> (Jupyter notebook) covers the training and usage of major functions (FeatureTransformation, Predictor and Optimizer) and analysis of graident activations using a pre-trained model. This file can be used as-is with the current and customized datasets by providing appropriate filepath and parameters.

#### For advanced users
You may change the model architecture and other parameters in the files under utils.

### Prerequisites
The package requires:
* <a href='https://www.tensorflow.org/'>Tensorflow 2.x</a>
* <a href='https://www.rdkit.org/'>RDKit </a>

## License
MIT License
