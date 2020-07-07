import numpy as np
import numba

@numba.jit(nopython=True)
def activation(x):
    """Sigmoid activation function."""
    return(1.0 / (1.0 + np.exp(-x)))

@numba.jit(nopython=True)
def activation_prime(x):
    """Derivative of the sigmoid activation function."""
    return x * (1 - x)

@numba.jit(nopython=True)
def node_activation(inputs, weights):
    """
    Returns the weighted sum of the inputs and weights (and bias).

    inputs and weights should be vectors, and weights should have 1 more
    element than inputs.
    """
    return np.dot(inputs, weights[:-1]) + weights[-1]

@numba.jit(nopython=True)
def forward_prop(inputs, weights):
    """Runs one iteration of forward propagation through the neural net and
    returns activations."""
    activations = [np.empty(weights[i].shape[0]) \
        for i in range(len(weights))]
    prev_activation = inputs

    for i in range(len(weights)):
        curr_activation = np.empty(weights[i].shape[0])

        for j in range(weights[i].shape[0]):
            if i > 0:
                curr_activation[j] = activation(node_activation( \
                    prev_activation, weights[i][j]))
            else:
                curr_activation[j] = inputs[j]

        prev_activation = curr_activation
        activations[i] = prev_activation

    return activations

@numba.jit(nopython=True)
def backprop(inputs, expected, weights):
    """Runs one iteration of forward propagation through the neural net, then
    backpropagates and returns updated weights."""
    for i in reversed(range(len(weights))):
        for j in range(weights[i].shape[0]):
            pass