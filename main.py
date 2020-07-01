import ann
import random
import json

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [[0], [1], [1], [0]]

n = ann.FFNN(weights)

print(weights)
print()
for i in inputs:
    print(n.forward_prop(i))
print()

for j in range(10000):
    for i in range(len(inputs)):
        n.backprop(inputs[i], expected[i], learning_rate=0.1)

print(weights)
print()
for i in inputs:
    print(n.forward_prop(i))
