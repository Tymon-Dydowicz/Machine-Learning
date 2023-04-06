from sklearn.datasets import fetch_california_housing
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

dataset = fetch_california_housing()

X = torch.tensor(dataset['data'], dtype=torch.float)
y = torch.tensor(dataset['target'], dtype=torch.float)
training, validation, test = data.random_split(data.TensorDataset(X, y), [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))

print("Training:", len(training))
print("Validation:", len(validation))
print("Test:", len(test))

regressor = nn.Linear(8, 1) # 8 input features, 1 output feature
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        
    def forward(self, prediction, target):
        return ((prediction-target)**2).mean()

mse = MSE()

opt = optim.Adam(regressor.parameters())

mse_values = []
batch_size = 100
n_epoch = 10
for epoch in range(n_epoch):    
    mse_epoch = []
    for X_batch, y_batch in data.DataLoader(training, batch_size=batch_size, shuffle=True):        
        opt.zero_grad()
        y_pred = regressor(X_batch).reshape((-1,))
        mse_value = mse(y_pred, y_batch)
        mse_value.backward()
        opt.step()
        mse_epoch.append(mse_value.item())
    mse_values.append(np.mean(mse_epoch))

_ = sns.relplot(x=range(len(mse_values)), y=mse_values, kind="line")

def train_linear_regression(training: data.Dataset, 
                            validation: data.Dataset, 
                            no_improvement: int = 10,
                            batch_size: int = 128,
                            max_epochs: int = 100_000):
                            
    n_features = training[0][0].shape[0]
    regressor = nn.Linear(n_features, 1) # n_features input features, 1 output feature
    opt = optim.Adam(regressor.parameters())

    training_mses = []
    validation_mses = []
    noImprov = 0
    bestMSE = float('inf')
    for epoch in range(max_epochs):
        regressor.train()
        mse_epoch = []
        for X_batch, y_batch in data.DataLoader(training, batch_size=batch_size, shuffle=True):
            opt.zero_grad()
            y_pred = regressor(X_batch).reshape((-1,))
            mse_value = mse(y_pred, y_batch)
            mse_value.backward()
            opt.step()
            mse_epoch.append(mse_value.item())
        training_mses.append(np.mean(mse_epoch))

        
        regressor.eval()
        y_pred = regressor(validation[:][0]).reshape((-1,))
        mse_value = mse(y_pred, validation[:][1])
        validation_mses.append(mse_value.item())
        if mse_value.item() <= bestMSE:
            bestMSE = mse_value.item()
            noImprov = 0
            print(mse_value.item(), epoch)
        else:
            noImprov += 1

        if noImprov == no_improvement:
            break
        


    return regressor, validation_mses

regressor, validation_mses = train_linear_regression(training, validation)

print(len(validation_mses))
print(validation_mses[-40:])
print(min(validation_mses[-40:]), validation_mses[-40:].index(min(validation_mses[-40:])))
_ = sns.relplot(x=range(1, 41), y=validation_mses[-40:], kind="line")

regressor.eval()
y_pred = regressor(test[:][0]).reshape((-1,))
print("MSE on the test set:", mse(y_pred, test[:][1]).item())

mnist_sklearn = fetch_openml('mnist_784', version=1, as_frame=False)

print(mnist_sklearn.DESCR)

X = torch.tensor(mnist_sklearn.data, dtype=torch.float)
y = torch.tensor([int(v) for v in mnist_sklearn.target])
mnist = data.TensorDataset(X, y)
mnist_conv = data.TensorDataset(X.reshape((-1, 1, 28, 28)), y)

fig = plt.figure(figsize = (4,4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(16):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(mnist[i][0].reshape(28, 28),cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0, 7, str(mnist[i][1].item()))

training, validation, test = data.random_split(mnist, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))
trainingConv, validationConv, testConv = data.random_split(mnist_conv, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))

print("Training:", len(training))
print("Validation:", len(validation))
print("Test:", len(test))

p = 28*28
k = 10
model = nn.Linear(p, k)

loss_function = torch.nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

def compute_acc(logits, expected):
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

loss_values = []
acc_values = []
batch_size = 512
n_epoch = 10

loader = data.DataLoader(training, batch_size=batch_size, shuffle=True)    

for epoch in range(n_epoch):    
    model.train()    
    epoch_loss = []
    for X_batch, y_batch in loader:
        opt.zero_grad()
        logits = model(X_batch)
        
        loss = loss_function(logits, y_batch)
        loss.backward()
        opt.step()        
        epoch_loss.append(loss.detach())
    model.eval()
    loss_values.append(torch.tensor(epoch_loss).mean())
    logits = model(validation[:][0])
    acc = compute_acc(logits, validation[:][1]).item()
    print("Epoch:", epoch, "accuracy:", acc)
    acc_values.append(acc)

plt.title("Loss on the training set")
plt.plot(loss_values)
plt.show()
plt.title("Accuracy on the validation set")
plt.plot(acc_values)
plt.show()

def train_classifier(model: nn.Module,        
              training: data.Dataset, 
              validation: data.Dataset,
              no_improvement: int = 10,
              batch_size: int = 128,
              max_epochs: int = 100_000):
    n_features = training[0][0].shape[0]
    classifier = model
    loss_function = torch.nn.CrossEntropyLoss()
    opt = optim.Adam(classifier.parameters())

    training_losses = []
    validation_losses = []
    accs = []
    noImprov = 0
    bestACC = float('-inf')
    for epoch in range(max_epochs):
        classifier.train()
        epoch_loss = []
        for X_batch, y_batch in data.DataLoader(training, batch_size=batch_size, shuffle=True):
            opt.zero_grad()
            logits = classifier(X_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.detach())
        training_losses.append(torch.tensor(epoch_loss).mean())

        classifier.eval()
        logits = classifier(validation[:][0])
        loss_value = loss_function(logits, validation[:][1])
        validation_losses.append(loss_value.item())
        acc = compute_acc(logits, validation[:][1]).item()
        accs.append(acc)

        if acc > bestACC:
            indexBest = epoch
            bestACC = acc
            noImprov = 0
            print(acc, epoch)
        else:
            noImprov += 1

        if noImprov >= no_improvement:
            break
    return accs

model = nn.Linear(28*28, 10)
accuracies = train_classifier(model, training, validation)

_ = sns.relplot(x=range(len(accuracies)), y=accuracies, kind="line")

model = nn.Sequential(nn.Linear(28*28, 300), nn.ReLU(), nn.Dropout(0.5), nn.Linear(300,10))

accuracies = train_classifier(model, training, validation)
print("The best accurac on the validation set:", max(accuracies))

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 4 * 64, 10)


    def forward(self, x):

        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        x = self.fc1(x)

        return x



class MultiLinearLayerNetWithDroupout(nn.Module):
    def __init__(self) -> None:
        super(MultiLinearLayerNetWithDroupout, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28*28, 420)
        self.fc2 = nn.Linear(420, 69)
        self.fc3 = nn.Linear(69, 30)
        self.fc4 = nn.Linear(30, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.softmax(x)
        x = self.fc4(x)

        
        return x



class MultiLinearLayerNetWithoutActivations(nn.Module):
    def __init__(self) -> None:
        super(MultiLinearLayerNetWithoutActivations, self).__init__()
        self.dropout = nn.AlphaDropout(0.5)
        self.fc1 = nn.Linear(28*28, 420)
        self.fc2 = nn.Linear(420, 69)
        self.fc3 = nn.Linear(69, 30)
        self.fc4 = nn.Linear(30, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x



class MultiLinearWithBottleneck(nn.Module):
    def __init__(self) -> None:
        super(MultiLinearWithBottleneck, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(x)
        x = self.fc6(x)

        return x



class MultiLinearLayerNetWithoutDroupout(nn.Module):
    def __init__(self) -> None:
        super(MultiLinearLayerNetWithoutDroupout, self).__init__()
        self.fc1 = nn.Linear(28*28, 420)
        self.fc2 = nn.Linear(420, 69)
        self.fc3 = nn.Linear(69, 30)
        self.fc4 = nn.Linear(30, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x



class DifferentActivationLinearLayerNet(nn.Module):
    def __init__(self, function) -> None:
        super(DifferentActivationLinearLayerNet, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 10)
        self.function = function

    def forward(self, x):
        x = self.function(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class MultiLinearLayerWithPowersOf2(nn.Module):
    def __init__(self) -> None:
        super(MultiLinearLayerWithPowersOf2, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        
        return x

Score = {}

model = ConvNet()#.to(device)
accuracies = train_classifier(model, trainingConv, validationConv)
acc = compute_acc(model(testConv[:][0]), testConv[:][1]).item()
print("The best accuracy on the test set:", acc)
Score['CNN'] = acc
# torch.save(model.state_dict(), 'ConvNetWeights.pth')

model = MultiLinearLayerNetWithoutDroupout()#.to(device)
accuracies = train_classifier(model, training, validation)
acc = compute_acc(model(test[:][0]), test[:][1]).item()
print("The accuracy on the test set:", acc)
Score['MLLND'] = acc
# torch.save(model.state_dict(), 'MultiLayerWeights.pth')

functions = [F.elu, F.selu, F.gelu, torch.tanh, torch.sigmoid, F.hardshrink, F.hardswish]
mem = float('-inf')
for fn in functions:
    model = DifferentActivationLinearLayerNet(fn)#.to(device)
    accuracies = train_classifier(model, training, validation)
    acc = compute_acc(model(test[:][0]), test[:][1]).item()
    print(fn, "The accuracy on the test set:", acc)
    if acc > mem:
        mem = acc
Score['DALLN'] = mem

model = MultiLinearLayerNetWithDroupout()#.to(device)
accuracies = train_classifier(model, training, validation)
acc = compute_acc(model(test[:][0]), test[:][1]).item()
print("The accuracy on the test set:", acc)
Score['MLLN'] = acc

model = MultiLinearLayerNetWithoutActivations()#.to(device)
accuracies = train_classifier(model, training, validation)
acc = compute_acc(model(test[:][0]), test[:][1]).item()
print("The best accuracy on the validation set:", acc)
Score['NoActiv'] = acc

model = MultiLinearWithBottleneck()#.to(device)
accuracies = train_classifier(model, training, validation)
acc = compute_acc(model(test[:][0]), test[:][1]).item()
print("The best accuracy on the validation set:", acc)
Score['BottleNeck'] = acc

model = MultiLinearLayerWithPowersOf2()#.to(device)
accuracies = train_classifier(model, training, validation)
acc = compute_acc(model(test[:][0]), test[:][1]).item()
print("The best accuracy on the validation set:", acc)
Score['PowersOf2'] = acc

plt.scatter(Score.values(), Score.keys())
print(Score)
