import activation

class NNNode:
    def __init__(self, weights):
        self.weights = weights
        self.weight_delta = [0 for i in weights]
        self.error_term = 0

    def forward_prop(self, inputs):
        """Calculates weighted sum of inputs and bias."""
        self.activation = activation.activation( \
                sum([inputs[i] * self.weights[i] \
                for i in range(len(inputs))]) + self.weights[-1])

        return self.activation

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weight_delta[i]

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

    def backprop(self, inputs, expected, learning_rate=0.1):
        """Runs one iteration of training and updating weights."""
        self.forward_prop(inputs)

        for i in reversed(range(self.layer_count)):
            for j in range(self.layers[i].node_count):
                for k in range(len(self.layers[i].nodes[j].weights)):
                    # upstream node's activation
                    # bias
                    if k == len(self.layers[i].nodes[j].weights) - 1:
                        ustream_out = 1
                    # input layer
                    elif i == 0:
                        ustream_out = inputs[k]
                    # hidden layers
                    else:
                        ustream_out = self.layers[i - 1].nodes[k].activation

                    # output layer errors
                    if i == self.layer_count - 1:
                        output_error = (self.layers[i].nodes[j].activation - \
                                expected[j])
                    # hidden layer errors
                    else:
                        # I feel like this could be less loopy
                        output_error = sum([n.error_term * m \
                                for n in self.layers[i + 1].nodes \
                                for m in n.weights])
                        output_error = sum([n.error_term * n.weights[j] \
                                for n in self.layers[i + 1].nodes])

                    activation_prime = activation.activation_prime( \
                            self.layers[i].nodes[j].activation)

                    error_term = output_error * activation_prime
                    self.layers[i].nodes[j].error_term = error_term

                    self.layers[i].nodes[j].weight_delta[k] = \
                            learning_rate * error_term * ustream_out

        # another loop??
        # adjust weights
        for i in self.layers:
            for j in i.nodes:
                j.update_weights()
