import ann
import random

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [[0], [1], [1], [0]]

def rand_weight():
    return random.randint(-10, 10) / 10.0

weights = [[[rand_weight() for i in range(3)] for j in range(2)], \
        [[rand_weight() for i in range(3)]]]

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
