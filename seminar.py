import numpy as np

class Layer():
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def num_params(self):
        num = 0
        for layer in self.layers:
            num += layer.num_params()
        return num
        

class LinearLayer():
    def __init__(self, input_size, output_size, W=None, b=None):
        if W is None:
            # инициализация случайными весами
            W = np.random.randn(output_size, input_size) * 0.01
        if b is None:
            b = np.zeros(output_size)
        self.W = W
        self.b = b
        
    def forward(self, x):
        return np.dot(x, self.W.T) + self.b
    
    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return self.W.size + self.b.size


# ======== Нелинейности ========

class ReLU():
    def forward(self, x):
        return np.maximum(0, x)

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return 0


class Sigmoid():
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return 0


class Softmax():
    def forward(self, x):
        # Численно стабильная версия
        exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return 0


# ======== Простая свёртка ========

class SimpleConv():
    """
    Простейшая 2D свёртка без padding, stride=1.
    x.shape = (H, W)
    kernel.shape = (kH, kW)
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, x):
        H, W = x.shape
        kH, kW = self.kernel.shape
        out_H = H - kH + 1
        out_W = W - kW + 1
        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = x[i:i+kH, j:j+kW]
                output[i, j] = np.sum(region * self.kernel)
        return output

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return self.kernel.size


# ======== Пример использования ========

if __name__ == "__main__":
    # Пример нейросети
    neural_network = NeuralNetwork(layers=[
        LinearLayer(784, 392),
        ReLU(),
        LinearLayer(392, 196),
        Sigmoid(),
        LinearLayer(196, 10),
        Softmax()
    ])

    x = np.random.rand(1, 784)
    y = neural_network(x)
    print("Выход сети:", y)
    print("Сумма softmax =", np.sum(y))

    # Пример свёртки
    img = np.random.rand(5, 5)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    conv = SimpleConv(kernel)
    out = conv(img)
    print("Выход свёртки:\n", out)