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

lr_loss_01 = readfile('lr_loss_0.1.txt')
lr_loss_10 = readfile('lr_loss_10.txt')
lr_loss_100 = readfile('lr_loss_100.txt')
lr_loss_1000 = readfile('lr_loss_1000.txt')


plt.plot(lr_loss_01)
plt.plot(lr_loss_10)
plt.plot(lr_loss_100)
plt.plot(lr_loss_1000)
plt.title('loss')
plt.legend(['lr=0.1','lr=10','lr=100', 'lr=1000'])
plt.grid(True)
plt.savefig('loss.png')
plt.show()
