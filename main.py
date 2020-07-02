import ann
import random

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [[0], [1], [1], [0]]

def rand_weight():
    return random.randint(-10, 10) / 10.0

# first element inputs, last element outputs
node_count = [2, 2, 1]
weights = []

for i in range(len(node_count)):
    if i == 0:
        continue

    layer = []

    for j in range(node_count[i]):
        layer.append([rand_weight() for k in range(node_count[i - 1] + 1)])

    weights.append(layer)

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
