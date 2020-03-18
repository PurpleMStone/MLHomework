import matplotlib.pyplot as plt

def readfile(filename):
    mylist = []
    with open(filename, 'r') as f:
        data = f.readlines()
        for line in data:
            element = line.split()
            element2 = list(map(float, element))
            mylist.append(element2)
    return mylist

train_loss_9 = readfile('train_loss_9.txt')
train_loss_5 = readfile('train_loss_5.txt')
val_loss_9 = readfile('val_loss_9.txt')
val_loss_5 = readfile('val_loss_5.txt')


plt.plot(train_loss_9)
plt.plot(train_loss_5)
plt.plot(val_loss_9)
plt.plot(val_loss_5)
plt.title('loss')
plt.legend(['train: 9hr','train: 5~9hr','val: 9hr', 'val: 5~9hr'])
plt.grid(True)
plt.savefig('compare.png')
plt.show()
