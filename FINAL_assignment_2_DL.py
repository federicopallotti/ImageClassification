#!/usr/bin/python3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)

#for training:
batch_size = 32
learning_rate = 0.001
momentum = 0.9

"""1.1 Dataset"""

#DATASETS
train_set = torchvision.datasets.CIFAR10(root='./data', train = True, transform = transforms.ToTensor(), download = True)

test_set = torchvision.datasets.CIFAR10(root='./data', train = False, transform = transforms.ToTensor())

#Dataloaders

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)

"""1) few images visualization"""

for i in range(5):
  plt.imshow( train_set[i][0].permute(1,2,0))
  plt.show(block = False)

"""2) dataset normalization and standardization"""

#compute means and std devs to feed the normalization func for training set
means = train_set.data.mean(axis = (0,1,2))/255

std_dev=train_set.data.std(axis = (0,1,2))/255

means

std_dev

transform_train = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(means, std_dev)
 ])

#upload normalized train set
new_train_set = torchvision.datasets.CIFAR10(root='./data', train = True, transform = transform_train, download = True)
new_train_set
test_set = torchvision.datasets.CIFAR10(root='./data', train = False, transform = transform_train, download = True)

for i in range(5):
  plt.imshow( new_train_set[i][0].permute(1,2,0))
  plt.show(block = False)

#compute means and std devs to feed the normalization func for test set
#means = test_set.data.mean(axis = (0,1,2))/255

#std_dev=test_set.data.std(axis = (0,1,2))/255

transform_test = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(means, std_dev)
 ])

#upload normalized test set
new_test_set  = torchvision.datasets.CIFAR10(root='./data', train = False, transform = transform_test)

#Dataloaders

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle = False)

"""3) Validation set initialization

"""

idx = np.arange(len(train_set))
id_train = idx[:-1000]
id_val = idx[50000 - 1000:]

train_sampler = torch.utils.data.SubsetRandomSampler(id_train)
val_sampler = torch.utils.data.SubsetRandomSampler(id_val)

train_loader = torch.utils.data.DataLoader(new_train_set, batch_size= batch_size,  sampler = train_sampler, num_workers = 2 )

val_loader= torch.utils.data.DataLoader(new_train_set, batch_size = batch_size, sampler =val_sampler, num_workers = 2)

"""1.2 Model"""

class ConvNet(nn.Module):
    def __init__(self, dropout_val):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)

        self.conv2 = nn.Conv2d(32,32,3)
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32,64,3)

        self.conv4 = nn.Conv2d(64,64,3)

        self.pool2 = nn.MaxPool2d(2)
        

        self.fc1 = nn.Linear(64*5*5, 512)
        
        self.dropout = torch.nn.Dropout(p=dropout_val)
        
        self.fc2 = nn.Linear(512,10)
      
    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.conv4(F.relu(self.conv3(x))))
      
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(-1, 5*5*64)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        out = self.fc2(x)
        # softmax is computed by cross entropy loss.
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""1.3 Training

4) plot the evolution of loss and accuracy for training e validation set
"""

def train_model(num_epochs,sampling_size, dropout_val,learning_rate, momentum):
  best_val = 0
  counter = 0

  training_loss = []
  validation_loss = []

  training_accuracy = []
  validation_accuracy = []

  #define the model
  model = ConvNet(dropout_val)
  model = model.to(device)

  # Create loss and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


  for epoch in range(1, num_epochs+1):
      running_loss = 0.0
      running_total = 0
      running_correct = 0
      run_step = 0
      train_loss_epoch = 0
      train_correct_epoch = 0
      train_total_epoch = 0
      count = 0
      
      for i, (images, labels) in enumerate(train_loader):
          model.train()
          images = images.to(device)
          labels = labels.to(device)  
          outputs = model(images)  # shape (B, 10).
          loss = loss_fn(outputs, labels)
          optimizer.zero_grad()  # reset gradients.
          loss.backward()  # compute gradients.
          optimizer.step()  # update parameters.

          running_loss += loss.item()
          running_total += labels.size(0)
          train_loss_epoch += loss.item()
          train_total_epoch += labels.size(0)
          count += 1

          with torch.no_grad():
              _, predicted = outputs.max(1)
          running_correct += (predicted == labels).sum().item()
          train_correct_epoch += (predicted == labels).sum().item()
          run_step += 1
          
          if i % sampling_size == 0:
              counter += 1
              # check accuracy.
              print(f'epoch: {epoch}, steps: {i}, '
                    f'train_loss: {running_loss / run_step :.3f}, '
                    f'running_acc: {100 * running_correct / running_total:.1f} %')
              #validation_loss.append(val_loss)
              #validation_accuracy.append(val_acc)
                    ###
              running_loss = 0.0

              running_total = 0
              running_correct = 0
              run_step = 0

      #appending for the training plots
      training_accuracy.append(100 * train_correct_epoch / train_total_epoch)
      training_loss.append(train_loss_epoch / count)

      # validate
      correct = 0
      total = 0
      model.eval()
      count = 0
      v_loss = 0
      with torch.no_grad():
          for data in val_loader:
              images, labels = data
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              ###
              v_loss += loss_fn(outputs, labels)
              count+= 1
              ###
              _, predicted = outputs.max(1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              
              
      #appending for val
      val_acc = 100 * correct / total
      val_loss = v_loss.item()/count
      validation_loss.append(val_loss)
      validation_accuracy.append(val_acc)

      if val_acc > best_val:
        best_val = val_acc
        best_epoch = epoch

      print(f'Validation accuracy: {val_acc} %')
      print(f'Validation error rate: {100 - val_acc: .2f} %')
      print(f'Validation loss:{val_loss} ')

      print(f'Best Validation accuracy:{best_val} ')
      print(f'Best epoch:{best_epoch} ')
  
  #print losses
  
  step = 1
  x_train = np.arange(1,num_epochs+1, step)
  plt.title("Loss")
  plt.plot(x_train, training_loss, label = 'training loss')

  x_val = np.arange(1,num_epochs + 1)
  plt.plot(x_val, validation_loss, label = 'validation loss')

  plt.legend()
  plt.show()

  #print accuracies
  step = 1
  plt.title("Accuracy")
  plt.plot(x_train, training_accuracy, label = 'training accuracy')
  x_val = np.arange(1,num_epochs + 1)
  plt.plot(x_val, validation_accuracy, label = 'validation accuracy')
  plt.legend()
  plt.show()


  print('Finished Training')
  return x_val, (validation_loss, validation_accuracy)

train_model(num_epochs = 20,sampling_size = 100, dropout_val = 0, learning_rate = 0.001, momentum = 0.9)

"""5) results with droput

"""

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.1, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.2, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.3, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.4, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.5, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.6, learning_rate = 0.001, momentum = 0.9)

train_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.7, learning_rate = 0.001, momentum = 0.9)

def hype():
  e = 50 # numbers of epochs
  d = 0.4 #best droput value
  learns = [0.001, 0.002, 0.004]
  moments = [ 0.85, 0.9, 0.95]
  losses = []
  accuracies = []
  names = []
  x_val, x_train = 0, 0
  for l in learns:
    for m in moments:
      print(f'number of epochs:{e}, dropout :{d}, learning rate:{l}, momentum:{m}')
      names.append(f'dr: {d}, lr: {l}, mom:{m}')
      x_val, (loss, acc) = train_model(num_epochs = e,sampling_size = 1000000, dropout_val = d, learning_rate = l, momentum = m)
      losses.append(loss)
      accuracies.append(acc)

  for i, l in enumerate(losses):
    plt.plot(x_val, l, label=names[i])

  plt.legend()
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.title('Model losses')
  plt.show()


  for i, a in enumerate(accuracies):
    plt.plot(x_val, a, label=names[i])

  plt.legend()
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.title('Model accuracies')
  plt.show()

hype()

def train_return_model(num_epochs,sampling_size, dropout_val,learning_rate, momentum):
  best_val = 0
  counter = 0

  training_loss = []
  validation_loss = []

  training_accuracy = []
  validation_accuracy = []

  #define the model
  model = ConvNet(dropout_val)
  model = model.to(device)

  # Create loss and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs+1):
      running_loss = 0.0
      running_total = 0
      running_correct = 0
      run_step = 0
      train_loss_epoch = 0
      train_correct_epoch = 0
      train_total_epoch = 0
      count = 0
      
      for i, (images, labels) in enumerate(train_loader):
          model.train()
          images = images.to(device)
          labels = labels.to(device)  
          outputs = model(images)  # shape (B, 10).
          loss = loss_fn(outputs, labels)
          optimizer.zero_grad()  # reset gradients.
          loss.backward()  # compute gradients.
          optimizer.step()  # update parameters.

          running_loss += loss.item()
          running_total += labels.size(0)
          train_loss_epoch += loss.item()
          train_total_epoch += labels.size(0)
          count += 1

          with torch.no_grad():
              _, predicted = outputs.max(1)
          running_correct += (predicted == labels).sum().item()
          train_correct_epoch += (predicted == labels).sum().item()
          run_step += 1
          
          if i % sampling_size == 0:
              counter += 1
              # check accuracy.
                    ###
              running_loss = 0.0

              running_total = 0
              running_correct = 0
              run_step = 0

      #appending for the training plots
      training_accuracy.append(100 * train_correct_epoch / train_total_epoch)
      training_loss.append(train_loss_epoch / count)

      # validate
      correct = 0
      total = 0
      model.eval()
      count = 0
      v_loss = 0
      with torch.no_grad():
          for data in val_loader:
              images, labels = data
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              ###
              v_loss += loss_fn(outputs, labels)
              count+= 1
              ###
              _, predicted = outputs.max(1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              
      #appending for val
      val_acc = 100 * correct / total
      val_loss = v_loss.item()/count
      validation_loss.append(val_loss)
      validation_accuracy.append(val_acc)

      if val_acc > best_val:
        best_val = val_acc
        best_epoch = epoch

  
  #print losses
  
  step = 1
  x_train = np.arange(1,num_epochs+1, step)
  plt.title("Loss")
  plt.plot(x_train, training_loss, label = 'training loss')

  x_val = np.arange(1,num_epochs + 1)
  plt.plot(x_val, validation_loss, label = 'validation loss')

  plt.legend()
  plt.show()

  #print accuracies
  step = 1
  plt.title("Accuracy")
  plt.plot(x_train, training_accuracy, label = 'training accuracy')
  x_val = np.arange(1,num_epochs + 1)
  plt.plot(x_val, validation_accuracy, label = 'validation accuracy')
  plt.legend()
  plt.show()


  print('Finished Training')
  return model

#training with the best parameters
final_model = train_return_model(num_epochs = 50,sampling_size = 100, dropout_val = 0.4, learning_rate = 0.004, momentum = 0.75)

def test(model):
  with torch.no_grad():
    correct = 0
    total = 0
    model.eval() # Set model in eval mode. Don’t forget!
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device) # shape (B)
        outputs = model(images) # shape (B, num_classes)
        # ’outputs’ are logits (unnormalized log prob).
        # Model prediction is the class which has the highest 
        # probability according to the model ,
        # i.e. the class which has the highest logit value:
        _, predicted = outputs.max(dim=1)
        # predicted.shape: (B)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f'Test accuracy: {test_acc} %')
    print(f'Test error rate: {100 - 100 * correct / total: .2f} %')

test(final_model)