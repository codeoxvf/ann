import numpy as np
import numba
import mlp

# TODO: object-oriented?

@numba.jit(nopython=True)
def print_results(inputs, weights):
    print('Weights: ', weights)
    print('Results:', [mlp.forward_prop(i, weights)[-1] for i in inputs])
    print()

epochs = 10000

node_count = (2, 1)
layer_count = len(node_count)

inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
expected = np.array([[0.0], [1.0], [1.0], [0.0]])

weights = numba.typed.List()
for i in range(layer_count):
    if i > 0:
        weights.append(np.random.uniform(low=-1.0, high=1.0, \
            size=(node_count[i], node_count[i-1] + 1)))
        continue
    weights.append(np.random.uniform(low=-1.0, high=1.0, \
        size=(node_count[i], inputs[0].shape[0] + 1)))

print_results(inputs, weights)

for _ in range(epochs):
    for curr_input in range(inputs.shape[0]):
        a = mlp.forward_prop(inputs[curr_input], weights)
        activations = numba.typed.List()
        [activations.append(i) for i in a]

        w = mlp.backprop(inputs[curr_input], expected[curr_input], \
            activations, weights)
        weights = numba.typed.List()
        [weights.append(i) for i in w]

print_results(inputs, weights)