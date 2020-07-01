import ann
import json
import random
import copy

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [[0], [1], [1], [0]]

with open('weights.json', 'r') as f:
    data=f.read()
weights = json.loads(data)

n = ann.FFNN(weights)

print(weights)
print()
for i in inputs:
    print(n.forward_prop(i))
print()

[n.backprop(inputs[i], expected[i], learning_rate=0.1) \
        for i in range(len(inputs)) \
        for j in range(10000)]

print(weights)
print()
for i in inputs:
    print(n.forward_prop(i))
