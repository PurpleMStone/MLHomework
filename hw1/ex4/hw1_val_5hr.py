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
x_train_set = np.concatenate((np.ones([x_train_set.shape[0], 1]), x_train_set), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([x_validation.shape[0], 1]), x_validation), axis = 1).astype(float)
print(x_train_set.shape)
lr_w = 100
iter_time = 20000
adagrad = np.zeros([dim, 1])
eps = 1e-10

training_loss = []
val_loss = []
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/x_train_set.shape[0]) # rmse
    valloss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/x_validation.shape[0]) # rmse
    
    if t > 300:
        training_loss.append(loss)
        val_loss.append(valloss)
    if t % 200 == 199:
        print('epoch: '+ str(t)+':  '+ str(loss))
    if t > 10000:
        lr_w = 10
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) # dim*1
    adagrad += gradient ** 2
    w = w - lr_w * gradient / np.sqrt(adagrad + eps)

np.save('weight.npy', w)

val_final_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/x_validation.shape[0])
print('val_loss: ' + str(val_final_loss))

# Plot loss curve
plt.plot(training_loss)
plt.plot(val_loss)
plt.title('Loss')
plt.legend(['train', 'val'])
plt.grid(True)
plt.savefig('five_for_ten_loss.png')
plt.show()

# Save loss data
fileObject = open('train_loss_5.txt', 'w')
for ip in training_loss:
    fileObject.write(str(ip))
    fileObject.write('\n')
fileObject.close()

fileObject = open('val_loss_5.txt', 'w')
for ip in val_loss:
    fileObject.write(str(ip))
    fileObject.write('\n')
fileObject.close()


# testing
testdata = pd.read_csv('../test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18*i:18*(1+i), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

#Save Prediction to CSV file
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
