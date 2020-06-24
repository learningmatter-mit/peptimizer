# Peptimizer

Peptimizer is a repository based on machine learning algorithms for the optimization of peptides, and functional polymers in general. The codebase has been designed to be used for optimization of functionality and synthetic accessibility.

## Tools
### Optimization of functionality
Based on our work on generating novel and highly efficient cell-penetrating peptides <a href='https://www.biorxiv.org/content/10.1101/2020.04.10.036566v1'> (link)</a>, we provide a generator-predictor-optimizer framework for the discovery of novel functional polymers. A tutorial notebook demonstrating the usage is presented in <b>Tutorial_CPP.ipynb</b>.<br> 
<ul>
  <li><b>Generator</b> is trained on the library of polymer sequences in an unsupervised fashion using recurrent neural networks, and is used to sample similar-looking polymers.</b> 
  <li><b>Predictor</b> is a convolutional neural network model trained on sequence-activity relationships. The sequences are represented as matrices of monomer fingerprint bit-vectors. Each fingerprint bti-vector is a topological exploration of the monomer graph, where atoms are treated as nodes and bonds as edges. This is used to estimate the activities for unknown sequences.</li>
  <li><b>Optimizer</b> is based on genetic algorithms, and optimizes sequences by evaluating single-residue and multi-residue mutations against an objective function.
</ul>

<img src="https://github.com/learningmatter-mit/peptimizer/blob/master/figures/CPP.svg" width="100%" height="400"><br>

### Optimization of synthetic accessibility
Based on our work on optimization of synthetic accessibility for polymers synthesized using flow chemistry<a href=''> (coming soon)</a>, we provide a predictor-optimizer framework. A tutorial notebook demonstrating the usage is presented in <b>Tutorial_Synthesis.ipynb</b>.<br>
<ul>
  <li><b>Predictor</b> is trained over experimental synthesis parameters such as pre-synthesized chain, incoming monomer, temperature, flow rate and catalysts. The different variables are represented as fingerprint, continuous and categorical features.<br>
  <li><b>Optimizer</b> for synthesis is a brute-force optimization code that evaluates single-point mutants of the wild-type sequence for higher theoretical yield.<br>
</ul>

<img src="https://github.com/learningmatter-mit/peptimizer/blob/master/figures/Synthesis.svg" width="100%" height="600"><br>

### Interpretability of models
Using <b>gradient activation maps</b>, we provide monomer and sub-structure level insight into the functionality of different sequences. For example, in the case of functionality-based models, this enables to find the specific monomers (and their substructures) which contribute positively or negatively to the activity.

## Dependencies
The package requires:
* <a href='https://www.tensorflow.org/'>Tensorflow 2.x</a>
* <a href='https://www.rdkit.org/'>RDKit </a>

## How to cite
Optimization of functionality codebase - 
```
@article{Schissel2020,
author = {Schissel, Carly K and Mohapatra, Somesh and Wolfe, Justin M and Fadzen, Colin M and Bellovoda, Kamela and Wu, Chia-Ling and Wood, Jenna A. and Malmberg, Annika B. and Loas, Andrei and G{\'{o}}mez-Bombarelli, Rafael and Pentelute, Bradley L.},
doi = {10.1101/2020.04.10.036566},
file = {:Users/somesh/Downloads/Articles/2020.04.10.036566v1.full.pdf:pdf},
journal = {bioRxiv},
title = {{Interpretable Deep Learning for De Novo Design of Cell-Penetrating Abiotic Polymers}},
url = {https://www.biorxiv.org/content/10.1101/2020.04.10.036566v1},
year = {2020}
}
```

Optimization of synthetic accessibility codebase -
```
@article{Mohapatra2020,
author = {Mohapatra, Somesh and Hartrampf, Nina and Poskus, Mackenzie and Loas, Andrei and Gomez-Bombarelli, Rafael and Pentelute, Bradley L.},
journal = {Under Preparation},
title = {Deep Learning for Prediction and Optimization of Rapid Flow Peptide Synthesis},
year = {2020}
}
```

## License
MIT License
