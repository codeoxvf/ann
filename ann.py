import activation

class NNNode:
    def __init__(self, weights):
        self.weights = weights

    def forward_prop(self, inputs):
        """Calculates weighted sum of inputs and bias."""
        self.activation = activation.activation( \
                sum([inputs[i] * self.weights[i] \
                for i in range(len(inputs))]) + self.weights[-1])

        return self.activation

class NNLayer:
    def __init__(self, weights):
        self.node_count = len(weights)
        self.nodes = [NNNode(weights[i]) for i in range(self.node_count)]

    def forward_prop(self, inputs):
        """Gets activations of each node in the layer."""
        self.output = [self.nodes[i].forward_prop(inputs) \
                for i in range(self.node_count)]

        return self.output

class FFNN:
    def __init__(self, weights):
        self.layer_count = len(weights)
        self.layers = [NNLayer(weights[i]) for i in range(self.layer_count)]

    def forward_prop(self, inputs):
        """Feeds inputs forward through layers and gets activations."""
        self.activations = inputs
        for i in range(self.layer_count):
            self.activations = self.layers[i].forward_prop(self.activations)

        return self.activations

    def backprop(self, inputs, expected, learning_rate=1):
        """Runs one iteration of training and updating weights."""
        self.forward_prop(inputs)

        layer_error = 0

        for i in reversed(range(self.layer_count)):
            curr_error = 0

            for j in range(self.layers[i].node_count):
                for k in range(len(self.layers[i].nodes[j].weights)):
                    # upstream node's activation
                    if k == len(self.layers[i].nodes[j].weights) - 1:
                        # bias
                        y_j = 1
                    elif i == 0:
                        # input layer
                        y_j = inputs[k]
                    else:
                        y_j = self.layers[i - 1].nodes[k].activation

                    # calculate error
                    if i == self.layer_count - 1:
                        # output layer
                        error = (expected[j] - \
                                    self.layers[i].nodes[j].activation) * \
                                activation.activation_prime( \
                                    self.layers[i].nodes[j].activation)
                        curr_error = sum([ \
                                ((expected[i] - \
                                    self.layers[-1].nodes[i].activation) \
                                    ** 2) / 2 for i in range(len(expected))])

                    else:
                        error = layer_error * \
                                sum([self.layers[i + 1].nodes[n].weights[m] \
                                    for n in range( \
                                        self.layers[i + 1].node_count) \
                                    for m in range(len( \
                                    self.layers[i + 1].nodes[n].weights))]) * \
                                activation.activation_prime( \
                                    self.layers[i].nodes[j].activation)

                        curr_error += error

                    self.layers[i].nodes[j].weights[k] -= \
                            learning_rate * error * y_j

            layer_error = curr_error
            curr_error = 0
