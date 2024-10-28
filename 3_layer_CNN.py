import numpy as np
from dataloader import Dataloader

class Flatten():
    def __init__(self):
        self.in_shape = None
    def forward(self, input):
        self.in_shape = input.shape
        return input.reshape(input.shape[0], -1)
    def backward(self, output_gradient):
        output_gradient = np.reshape(output_gradient, self.in_shape)
        return output_gradient

class Linear():
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.empty((in_features, out_features))
        if bias:
            self.bias = np.empty((out_features, 1))
        self.initialize_params()
        self.input = None

    def initialize_params(self):
        self.weight = np.random.uniform(low=-1.0, high=1.0, size=(self.in_features, self.out_features))
        if self.bias is not None:
            self.bias = np.random.uniform(low=-1.0, high=1.0, size=(1, self.out_features))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weight) + self.bias
    
    def backward(self, output_gradient, lr):
        input_gradient = np.dot(output_gradient, self.weight.T)
        weight_gradient = np.dot(self.input.T, output_gradient)
        self.weight = self.weight - lr * weight_gradient
        if self.bias is not None:
            bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
            self.bias = self.bias - lr * bias_gradient
        return input_gradient
    


class Conv2d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernnel_size = kernel_size
        self.weight = np.empty((out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = np.empty((1, 1, out_channels))
        self.initialize_params()
        self.input = None

    def initialize_params(self):
        self.weight = np.random.uniform(low=-1.0, high=1.0, size=(self.out_channels, self.kernnel_size, self.kernnel_size))
        if self.bias is not None:
            self.bias = np.random.uniform(low=-1.0, high=1.0, size=(self.out_channels, 1, 1))

    def forward(self, input):
        self.input = input
        w_iter = np.shape(input)[-1] - self.kernnel_size + 1
        h_iter = np.shape(input)[-2] - self.kernnel_size + 1
        output = np.empty((np.shape(input)[0], self.out_channels, h_iter, w_iter))
        for i in range(h_iter):
            for j in range(w_iter):
                input_arr = input[:, :, i:i+3, j:j+3]
                out_arr = input_arr*self.weight
                out_arr = np.sum(out_arr, axis=(2, 3), keepdims=True)
                output[:, :, i:i+1, j:j+1] = out_arr
        output += self.bias
        return output
    
    def backward(self, output_gradient, lr):
        batch_size, out_channels, out_h, out_w = output_gradient.shape
        in_channels, in_h, in_w = self.input.shape[1:]

        input_gradient = np.zeros_like(self.input)
        weight_gradient = np.zeros_like(self.weight)
        bias_gradient = np.zeros_like(self.bias)

        for i in range(out_h):
            for j in range(out_w):
                for oc in range(out_channels):
                    output_gradient_slice = output_gradient[:, oc, np.newaxis, i, j, np.newaxis, np.newaxis]
                    input_gradient[:, :, i:i+self.kernnel_size, j:j+self.kernnel_size] += self.weight[oc] * output_gradient_slice

                    input_slice = self.input[:, :, i:i+self.kernnel_size, j:j+self.kernnel_size]
                    weight_gradient[oc] += np.sum(input_slice * output_gradient_slice, axis=(0, 1))

        if self.bias is not None:
            bias_gradient = np.sum(output_gradient, axis=(0, 2, 3), keepdims=True) / batch_size
            self.bias = self.bias - lr * bias_gradient

        weight_gradient = weight_gradient / batch_size 
        self.weight = self.weight - lr * weight_gradient

        return input_gradient

    
    
class MaxPool2d():
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size
        self.in_shape = None
        self.argmax_indices = None

    def forward(self, input):
        self.in_shape = input.shape
        batch_size, channels, height, width = input.shape

        out_height = height // self.kernel_size
        out_width = width // self.kernel_size

        output = np.zeros((batch_size, channels, out_height, out_width))
        self.argmax_indices = np.zeros((batch_size, channels, out_height, out_width), dtype=int)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.kernel_size
                h_end = h_start + self.kernel_size
                w_start = j * self.kernel_size
                w_end = w_start + self.kernel_size

                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                flat_input_slice = input_slice.reshape(batch_size, channels, -1)
                max_indices = np.argmax(flat_input_slice, axis=2)
                self.argmax_indices[:, :, i, j] = max_indices

                max_values = np.max(flat_input_slice, axis=2)
                output[:, :, i, j] = max_values

        return output

    def backward(self, grad_output):
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros(self.in_shape)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.kernel_size
                h_end = h_start + self.kernel_size
                w_start = j * self.kernel_size
                w_end = w_start + self.kernel_size

                for batch in range(batch_size):
                    for channel in range(channels):
                        max_index = self.argmax_indices[batch, channel, i, j]
                        max_h = max_index // self.kernel_size
                        max_w = max_index % self.kernel_size
                        grad_input[batch, channel, h_start + max_h, w_start + max_w] = grad_output[batch, channel, i, j]

        return grad_input
 
    

    
class ReLU():
    def __init__(self):
        self.input = None
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient):
        if output_gradient.shape != self.input.shape:
            output_gradient = np.reshape(output_gradient, self.input.shape)
        grad = self.input > 0
        return grad * output_gradient
    

class Softmax():
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, input):
        input = input - np.max(input, axis=-self.dim, keepdims=True)

        e_x = np.exp(input)
        
        softmax_output = e_x / (np.sum(e_x, axis=-self.dim, keepdims=True) + 1e-15)
        return softmax_output

    
class CrossEntropyLoss():
    def __init__(self):
        pass

    def CrossEntropyLoss(self, output, label):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        log_output = np.log(output)
        cross_entropy = -np.sum(log_output*label) / label.shape[0]
        return cross_entropy
    
# -------------------------------- load Data
path = 'dataset'
train_loader = Dataloader(path, is_train=True, batch_size=128, shuffle=True)
test_loader = Dataloader(path, is_train=False, batch_size=128, shuffle=False)

# -------------------------------- model

class NeuralNetwork():
    def __init__(self):
        self.conv_layer_1 = Conv2d(1, 32, 3)
        self.relu_1 = ReLU()
        self.maxpool_1 = MaxPool2d(2)
        self.conv_layer_2 = Conv2d(32, 32, 3)
        self.relu_2 = ReLU()
        self.maxpool_2 = MaxPool2d(2)
        self.Flatten = Flatten()
        self.linear_layer = Linear(5*5*32, 10)
        self.Softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.maxpool_1.forward(x)
        x = self.conv_layer_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.maxpool_2.forward(x)
        x = self.Flatten.forward(x)
        x = self.linear_layer.forward(x)
        x = self.Softmax.forward(x)
        return x
    
    def backward(self, output, label, lr):
        grad_loss = output - label
        grad_output = self.linear_layer.backward(grad_loss, lr)
        grad_output = self.Flatten.backward(grad_output)
        grad_output = self.maxpool_2.backward(grad_output)
        grad_output = self.relu_2.backward(grad_output)
        grad_output = self.conv_layer_2.backward(grad_output, lr)
        grad_output = self.maxpool_1.backward(grad_output)
        grad_output = self.relu_1.backward(grad_output)
        grad_output = self.conv_layer_1.backward(grad_output, lr)
        return grad_output



model = NeuralNetwork()

# --------------------------------- training

loss_fn = CrossEntropyLoss()

epochs = 4
lr = 0.0001


train_losses = []
test_losses = []
count = 0
for epoch in range(epochs):
    for i, (input, label) in enumerate(train_loader):
        count += 1
        output = model.forward(input)
        loss = loss_fn.CrossEntropyLoss(output, label)
        model.backward(output, label, lr)
        if count % 3 == 0:
            train_losses.append(loss)
        
        running_loss_test = 0.0
        for i, (input, labels) in enumerate(test_loader):
            outputs = model.forward(input)
            test_loss = loss_fn.CrossEntropyLoss(outputs, labels)
            break
        if count%3 == 0:
            test_losses.append(test_loss)
    

# --------------------------------- test
cor_pred = 0
total_pred = 0
y_preds = []
y_labels = []

y_top_probs = []
y_images = []
for i, (input, label) in enumerate(test_loader):
    prediction = model.forward(input)
    for i in range(len(prediction)):
        if np.argmax(prediction[i]) == np.argmax(label[i]):
            cor_pred += 1
        total_pred += 1
    predicted = np.argmax(prediction, 1)

    y_preds.append(predicted)
    y_labels.append(np.argmax(label, 1))
    
    y_top_probs.append(prediction)
    y_images.append(input)


accuracy = cor_pred / total_pred * 100
print(f"accuracy: {accuracy}%")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("3_layer_cnn_lossgraph.png")


y_labels = np.concatenate(y_labels)
y_preds = np.concatenate(y_preds)

cm = confusion_matrix(y_labels, y_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.savefig("3_layer_cnn_cm.png")




y_top_probs = np.concatenate(y_top_probs)
y_images = np.concatenate(y_images)
top3_indices = np.argsort(y_top_probs, axis=0)[-3:][::-1]


figure = plt.figure(figsize=(8, 14))
cols, rows = 3, 10
width, height = 0, 0
for i in range(1, cols * rows + 1):
    img = y_images[top3_indices[height][width]]
    index = top3_indices[height][width]
    prob = y_top_probs[index][width]
    figure.add_subplot(rows, cols, i)
    plt.title(f"{prob*100:.2f} | index:{index}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

    height += 1
    if height == 3:
        height = 0
        width += 1
    
    
plt.savefig("3_layer_cnn_top3.png")

