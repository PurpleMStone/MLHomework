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

train_loss_more = readfile('train_loss_more.txt')
train_loss_less = readfile('train_loss_less.txt')
val_loss_more = readfile('val_loss_more.txt')
val_loss_less = readfile('val_loss_less.txt')


plt.plot(train_loss_more)
plt.plot(train_loss_less)
plt.plot(val_loss_more)
plt.plot(val_loss_less)
plt.title('loss')
plt.legend(['train: 18 features','train: 1 feature','val: 18 features', 'val: 1 feature'])
plt.grid(True)
plt.savefig('compare.png')
plt.show()
