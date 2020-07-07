# TODO: object-oriented?
import numpy as np
import mlp

epochs = 1
learning_rate = 0.1

node_count = (2, 2, 1)
layer_count = len(node_count)

inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
expected = np.array([[0.0], [1.0], [1.0], [0.0]])

weights = [np.random.uniform(low=-1.0, high=1.0, size=(i, 3)) \
    for i in node_count]

[print(mlp.forward_prop(i, weights)) for i in inputs]

for _ in range(epochs):
    for curr_input in inputs:
        activations = mlp.forward_prop(inputs[curr_input], weights)

        weights = mlp.backprop(inputs[curr_input], \
            expected[curr_input], weights)