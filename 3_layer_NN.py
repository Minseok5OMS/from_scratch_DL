import numpy as np
from dataloader import Dataloader

class Flatten():
    def forward(self, input):
        size = np.size(input[0])
        return input.reshape(-1, size)


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
    
class ReLU():
    def __init__(self):
        self.input = None
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient):
        grad = self.input > 0
        return grad * output_gradient
    


class Softmax():
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, input):
        input_stable = input - np.max(input, axis=self.dim, keepdims=True)
        e_x = np.exp(input_stable)
        softmax_output = e_x / np.sum(e_x, axis=self.dim, keepdims=True)
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
train_loader = Dataloader(path, is_train=True, batch_size=64, shuffle=True)
test_loader = Dataloader(path, is_train=False, batch_size=64, shuffle=False)

# -------------------------------- model

class NeuralNetwork():
    def __init__(self):
        self.Flatten = Flatten()
        self.linear_layer_1 = Linear(28*28, 512)
        self.relu_1 = ReLU()
        self.linear_layer_2 = Linear(512, 256)
        self.relu_2 = ReLU()
        self.linear_layer_3 = Linear(256, 10)
        self.Softmax = Softmax(dim=1)

    def forward(self, x):
        
        x = self.Flatten.forward(x)
        x = self.linear_layer_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.linear_layer_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.linear_layer_3.forward(x)
        x = self.Softmax.forward(x)
        return x
    
    def backward(self, output, label, lr):
        grad_loss = output - label
        grad_output = self.linear_layer_3.backward(grad_loss, lr)
        grad_output = self.relu_2.backward(grad_output)
        grad_output = self.linear_layer_2.backward(grad_output, lr)
        grad_output = self.relu_1.backward(grad_output)
        grad_output = self.linear_layer_1.backward(grad_output, lr)
        pass



model = NeuralNetwork()

# --------------------------------- train

loss_fn = CrossEntropyLoss()

epochs = 2
lr = 0.001


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
plt.savefig("3_layer_nn_lossgraph.png")


y_labels = np.concatenate(y_labels)
y_preds = np.concatenate(y_preds)

cm = confusion_matrix(y_labels, y_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.savefig("3_layer_nn_cm.png")




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
    
    
plt.savefig("3_layer_nn_top3.png")

