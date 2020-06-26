import ann
import json
import random

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [[0], [1], [1], [0]]

#with open('weights.json', 'r') as f:
#    data=f.read()
#weights = json.loads(data)

random.seed(a=926835)

weights = [[[random.randrange(-10, 10) / 10.0 for k in range(3)] for j in range(2)]]
print(weights)
weights.append([[random.randrange(-10, 10) / 10.0 for i in range(3)]])
print(weights)

with open('weights.json', 'w') as f:
    json.dump(weights, f)

n = ann.FFNN(weights)

print("Initial result: ", [n.forward_prop(i) for i in inputs])
print("Expected: ", expected)
print("Weights: ", weights)
print("")

for j in range(100000):
    for i in range(len(inputs)):
        n.backprop(inputs[i], expected[i], learning_rate=0.01)

print("Updated result: ", [n.forward_prop(i) for i in inputs])
print("Weights: ", weights)
