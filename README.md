## MultiLayerPerceptron

This project is about implementing a basic perceptron and then followed by a multi-layer perceptron using [NumPy](https://numpy.org). The framework for multi-layer perceptron is then used to solve the "Handwritten Digit Recognition" problem that the MNIST database can solve.

## Motivation
This project was a part of Assignment 1 of the "Machine Learning" course at USI, Lugano taken by [Dr. Micheal Wand](https://people.idsia.ch/~michael/).

## Tech used
<b>Built with</b>
- [Python3](https://www.python.org)
- [NumPy](https://numpy.org)

## Features
The project includes implementation of the following concepts from scratch using NumPy as opposed to using traditional Deep Learning libraries like [PyTorch](https://pytorch.org).
- The Perceptron (A very simple single layer network)
- Activation functions and their derivatives (*tanh*, *softmax*)
- Forward and Backward pass for the linear layer
- Forward and Backward pass for the cross-entropy layer
- Sequential Layer

Some basic training concepts like the following were also implemented on the MNIST problem:
- Splitting the data into a training and a validation set
- Functions for the training loop and to calculate validation and test accuracies
- Early stopping

## File descriptions
1. `perceptron.py`

In this script the concepts of perceptron are used to find the weights of a polynomial defined by a 
$X \in \mathbb{R}^{3 \times 10}$ and $t \in \mathbb{R}$.

2. `framework.py`

In this script the various DL concepts are formulated in the form of Python class definitions, methods and attributes. Some of these include:
- Linear Layer
- Activation functions (*tanh*, *softmax*)
- Cross Entropy Loss function
- Sequential Layer
- Gradient Descent

3. `mnist.py`

In this script, the MNIST data is imported and trained using functions and methods defined in `framework.py`.

## Credits
Two articles helped me through the creation of this project:
- [Backpropagation through a fully-connected layer by Eli Bendersky](https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/) 
- [A simple neural net in numpy by Silvian Gugger](https://sgugger.github.io/a-simple-neural-net-in-numpy.html)