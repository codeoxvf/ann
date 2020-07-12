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
    result = [np.empty(weights[i].shape[0]) \
        for i in range(len(weights))]

    for i in range(len(weights)):
        for j in range(weights[i].shape[0]):
            result[i][j] = activation(node_activation(inputs, weights[i][j]))

    return result

#@numba.jit(nopython=True)
def backprop(inputs, expected, activations, weights, learning_rate=0.1):
    """Runs one iteration of forward propagation through the neural net, then
    backpropagates and returns updated weights."""
    weight_delta = [np.empty(i.shape) for i in weights]
    dstream_error_term = np.empty(1)

    for _ in range(len(weights)):
        i = len(weights) - _ - 1

        curr_error_term = np.empty(weights[i].shape[0])

        for j in range(weights[i].shape[0]):
            if i == 0:
                ustream_activation = inputs
            else:
                ustream_activation = np.append(activations[i-1], 1)

            if i == len(weights) - 1:
                activation_error = activations[-1][j] - expected[j]
            else:
                # FIXME
                activation_error = 0.0
                for k in range(dstream_error_term.shape[0]):
                    activation_error += np.sum(dstream_error_term[k] * \
                        weights[i][k])
                activation_error *= activation_prime(activations[i][j])

            error_term = activation_prime(activations[i][j]) * activation_error

            weight_delta[i][j] = learning_rate * \
                error_term * ustream_activation

            curr_error_term[j] = error_term
        dstream_error_term = curr_error_term

    return [weights[i] - weight_delta[i] for i in range(len(weights))]