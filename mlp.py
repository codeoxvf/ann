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
def forward_prop(inputs, weights):
    """Runs one iteration of forward propagation through the neural net and
    returns activations."""
    activations = []
    result = inputs

    for i in range(len(weights)):
        result = activation(np.sum(weights[i] * np.append(result, 1), axis=1))
        activations.append(result)

    return activations

@numba.jit(nopython=True)
def backprop(inputs, expected, activations, weights, learning_rate=0.1):
    """Runs one iteration of forward propagation through the neural net, then
    backpropagates and returns updated weights."""
    weight_delta = [np.empty(i.shape) for i in weights]

    node_count = [i.shape[0] for i in weights]
    layer_count = len(node_count)
    dstream_error_term = None

    for i in range(layer_count):
        i = layer_count - i - 1

        if i == layer_count - 1:
            error_term = activations[-1] - expected
        else:
            # each node's error term is the sum of the product of all the
            # upstream nodes' error terms and connecting weights
            # TODO: don't use for loop
            error_term = np.empty(node_count[i])
            for j in range(node_count[i]):
                error_term[j] = np.sum(dstream_error_term * weights[i+1].T[j])

        error_term = error_term * activation_prime(activations[i])

        if i > 0:
            ustream_activation = activations[i-1]
        else:
            ustream_activation = inputs

        ustream_activation = np.append(ustream_activation, 1)

        # TODO: don't use for loop
        for j in range(error_term.shape[0]):
            weight_delta[i][j] = learning_rate * \
                error_term[j] * ustream_activation

        dstream_error_term = error_term

    return [weights[i] - weight_delta[i] for i in range(len(weights))]