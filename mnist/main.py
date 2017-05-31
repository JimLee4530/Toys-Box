from mnist import *
from get_data import *
from solver import *
import matplotlib.pyplot as plt

X_train = loadImageSet(0)
y_train = loadLabelSet(0)
X_test = loadImageSet(1)
y_test = loadLabelSet(1)

# print X_train[0]

X_train = X_train/255.0
X_test = X_test/255.0

# print X_train[0]

# print X_train.shape
data={}
N,W,H = X_train.shape
X_train=X_train.reshape((N,1,W,H))
N,W,H = X_test.shape
X_test=X_test.reshape((N,1,W,H))

data['X_train'] = X_train
data['y_train'] = y_train
data['X_val'] = X_test
data['y_val'] = y_test

# print X_train[0].shape
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(X_train[0][0], cmap='gray')
plt.show()
print y_train[0]

model = mnist(reg=0.0005)

solver = Solver(model, data,
                num_epochs=4, batch_size=50,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 7e-8,
                  'momentum': 0.9
                },
                gamma = 0.0001,
                power = 0.75,
                verbose=True, print_every=100)
solver.train()

params = file('params.txt','w')
params.write(str(model.params))
params.close()