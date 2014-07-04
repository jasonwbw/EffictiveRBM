EffictiveRBM
============

EffictiveRBM is a python version restricted boltzmann machine tools for solve problems or build more complex model. The effictive means this is a fast rbm tools, it accelerate by cython and multi-process. It has been used to build the autoencoder model which descript by Hinton.

# How to Use
First, initialize an RBM with the desired number of visible, hidden units and process number.  
Then train the rbm model with Iterator and params which descript in rbm.py
```python
rbm = ParallelRBM(self.converter.dimensionality, 1000, 5)
rbm.train(DefaultBatchIterator(100, self.converter.train_images), max_epochs = 20, batch = 100)
```

## Dependencies
- [NumPy](http://www.numpy.org): normal computing
- [Scipy](http://www.scipy.org/): faster sigmoid computing

## Optional dependencies
- [Cython](http://www.cython.org): Only necessary to run the faster version. Version 0.17.1 or higher.
- [matplotlib](http://matplotlib.sourceforge.net/): For plotting some data

# Reference

* [A Practical guide to training restricted Boltzmann machines](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), by Geoffrey Hinton.
* [Reducing the Dimensionality of data with neural networks](http://www.cs.toronto.edu/~hinton/science.pdf), by Geoffrey Hinton.
* [Learning Deep Architectures for AI](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf), by Yoshua Bengio.

## License
The MIT License (MIT)

