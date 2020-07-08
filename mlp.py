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
def backprop(inputs, expected, activations, weights, learning_rate=0.1):
    """Runs one iteration of forward propagation through the neural net, then
    backpropagates and returns updated weights."""
    weight_delta = [np.empty(i.shape) for i in weights]
    dstream_error_term = np.empty(1)

    for _ in range(len(weights)):
        i = len(weights) - _ - 1
        if i == 0:
            break

        curr_error_term = np.empty(weights[i].shape[0])

        for j in range(weights[i].shape[0]):
            ustream_activation = np.append(activations[i-1], 1)

            if i == len(weights) - 1:
                activation_error = activations[-1][j] - expected[j]
            else:
                activation_error = np.sum(np.multiply(dstream_error_term, \
                    weights[i]))

            error_term = activation_prime(activations[i][j]) * activation_error

            weight_delta[i][j] = learning_rate * \
                error_term * ustream_activation

            curr_error_term[j] = error_term
        dstream_error_term = curr_error_term

    return [weights[i] - weight_delta[i] for i in range(len(weights))]