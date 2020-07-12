# TODO: object-oriented?
import numpy as np
import numba
import mlp

def print_results(inputs, weights):
    print("Weights: ", weights)
    print("Results:")
    [print(mlp.forward_prop(i, weights)) for i in inputs]
    print()

epochs = 1

input_count = 2
node_count = (2, 1)
layer_count = len(node_count)

inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
expected = np.array([[0.0], [1.0], [1.0], [0.0]])

weights = numba.typed.List()
# TODO: dynamic weight array generation
for i in node_count:
    pass
[weights.append(np.random.uniform(low=-1.0, high=1.0, size=(i))) \
    for i in node_count]

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