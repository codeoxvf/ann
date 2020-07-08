# TODO: object-oriented?
import numpy as np
import numba
import mlp

epochs = 10000
learning_rate = 0.5

node_count = (2, 2, 1)
layer_count = len(node_count)

inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
expected = np.array([[0.0], [1.0], [1.0], [0.0]])

weights = numba.typed.List()
[weights.append(np.random.uniform(low=-1.0, high=1.0, size=(i, 3))) \
    for i in node_count]

print(weights)
print()
for i in inputs:
    r = mlp.forward_prop(i, weights)
    for j in r:
        print(j)
    print()

for _ in range(epochs):
    for curr_input in range(inputs.shape[0]):
        activations = numba.typed.List()
        a = mlp.forward_prop(inputs[curr_input], weights)
        [activations.append(i) for i in a]

        w = mlp.backprop(inputs[curr_input], expected[curr_input], \
            activations, weights)
        weights = numba.typed.List()
        [weights.append(i) for i in w]

print(weights)
print()
for i in inputs:
    r = mlp.forward_prop(i, weights)
    for j in r:
        print(j)
    print()