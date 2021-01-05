import numpy as np
import scipy.io as scio


data = scio.loadmat('1.mat')
net = data['Net']
idx = data['true_idx4']
print(net)
data1 = 'snd(o)_solved'
scio.savemat(data1, {'net':net, 'idx':idx})
print('finished')


