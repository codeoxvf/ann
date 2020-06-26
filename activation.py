import math

def sigmoid(x):
    """Sigmoid activation function"""
    return(1.0 / (1.0 + (math.e ** -x)))

def sigmoid_prime(x):
    """Derivative of the sigmoid activation function"""
    return x * (1 - x)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return math.tanh(x)

def tanh_prime(x):
    """Derivative of the hyperbolic tangent activation function"""
    return 1 - (tanh(x) ** 2)

def relu(x):
    """ReLU (rectified linear unit) activation function"""
    return max(0, x)

def relu_prime(x):
    """Derivative of the ReLU activation function"""
    return 1 if x > 0 else 0

activation = sigmoid
activation_prime = sigmoid_prime
