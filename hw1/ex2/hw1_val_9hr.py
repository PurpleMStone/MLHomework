import sys
import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

print(pd.__version__)
data = pd.read_csv('../train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day*24:(day+1)*24] = raw_data[18*(20*month+day):18*(20*month+day+1)]
    month_data[month] = sample

x = np.empty([12*471, 18*9], dtype=float)  # training data
y = np.empty([12*471, 1], dtype=float)     # training set
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471+day*24+hour, :] = month_data[month][:, day*24+hour:day*24+hour+9].reshape(1,-1)
            y[month*471+day*24+hour, 0] = month_data[month][9, day*24+hour+9]   #value

# Normalization
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# Split training data into "train_set" and "validation_set"
x_train_set = x[:math.floor(len(x)*0.8), :]
y_train_set = y[:math.floor(len(y)*0.8), :]
x_validation = x[math.floor(len(x)*0.8):, :]
y_validation = y[math.floor(len(y)*0.8):, :]

# training
dim = 18*9 + 1
w = np.zeros([dim, 1])
x_train_set = np.concatenate((np.ones([math.floor(12*471*0.8), 1]), x_train_set), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([x_validation.shape[0], 1]), x_validation), axis = 1).astype(float)
print(x_train_set.shape)
learning_rate = 100
iter_time = 15000
adagrad = np.zeros([dim, 1])
eps = 1e-10

training_loss = []
val_loss = []
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/math.floor(12*471*0.8)) # rmse
    valloss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/math.floor(12*471*0.2)) # rmse

    if t % 1000 == 999:
        training_loss.append(loss)
        val_loss.append(valloss)
        print(str(t)+":"+str(loss))
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

np.save('weight.npy', w)

val_final_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/math.floor(12*471*0.2))
print('val_loss: ' + str(val_final_loss))

# Plot loss curve
plt.plot(training_loss)
plt.plot(val_loss)
plt.title('Loss')
plt.legend(['train', 'val'])
plt.grid(True)
plt.savefig('nine_for_ten_loss.png')
plt.show()

# Save loss data
fileObject = open('train_loss_9.txt', 'w')
for ip in training_loss:
    fileObject.write(str(ip))
    fileObject.write('\n')
fileObject.close()

fileObject = open('val_loss_9.txt', 'w')
for ip in val_loss:
    fileObject.write(str(ip))
    fileObject.write('\n')
fileObject.close()
