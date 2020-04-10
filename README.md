# Interpretable deep learning for <i>de novo</i> design of cell-penetrating abiotic polymers

This package couples supervised and unsupervised deep learning models to sample peptides and other functional polymers, predict and infer their activity, and optimize for a given set of parameters.

The package is accompanied by a <b>Tutorial.ipynb</b> (Jupyter notebook) that covers the training and usage of major functions (generator, predictor and optimizer) and analysis of graident activations using a pre-trained model.

Paper: <a href='https://github.mit.edu/MLMat/Mach_CPP/'> Interpretable deep learning for <i>de novo</i> design of cell-penetrating abiotic polymers </a>

Cite:
```
To be updated.
```

## Prerequisites
The package requires:
* <a href='http://faroit.com/keras-docs/2.1.4/'>Keras 2.1.4 </a>
* <a href='https://www.tensorflow.org/'>Tensorflow 1.5.0 </a>
* <a href='https://www.rdkit.org/'>RDKit </a>
* <a href='https://github.com/titu1994/Nested-LSTM'>Nested LSTM </a>

If you are new to Python, please consider using a package manager, preferably <a href='https://www.anaconda.com/'>anaconda</a>, to manage the libraries for the implementation of the package.

After installing anaconda, use the <b>mach_cpp.yml</b> file to create the environment - <br>
```
conda env create -f mach_cpp.yml python==3.6
```

## Usage
The <b>Tutorial.ipynb</b> can be used as-is with the current and customized datasets by providing appropriate filepath.

#### For advanced users
You may change the model architecture and other parameters in the generator, predictor and optimizer files.

## Data
There are two datasets -
* <b>generator_dataset.csv</b> - Used for the training of the generator
* <b>predictor_dataset.csv</b> - Used for the training of the predictor

## Pre-trained models
The pre-trained models are available for download -
* <b><a href='https://www.dropbox.com/s/jrtghjd5fvtrbl9/generator.hdf5?dl=0'>generator</a></b>
* <b><a href='https://www.dropbox.com/s/lc0edfl51ppln75/predictor.hdf5?dl=0'>predictor</a></b>

## Authors
The package was primarily written by <a href='https://www.linkedin.com/in/pikulsomesh/'>Somesh Mohapatra</a> who was advised by <a href='https://dmse.mit.edu/people/rafael-gomez-bombarelli'>Prof. Rafael Gómez-Bombarelli </a>.

The wet-lab experiments were performed by Carly K. Schissel, Dr. Justin M. Wolfe, Dr. Colin M. Fadzen, Kamela Bellovoda, Dr. Leslie Wu and Dr. Annika Malmberg, under the supervision of <a href='https://chemistry.mit.edu/profile/bradley-l-pentelute/'>Prof. Bradley Pentelute</a>.

## License
MIT License
