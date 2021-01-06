import scipy.io as scio

dataset = scio.loadmat('./data/1.mat')
print(dataset)
net = dataset['Net']
net1 = net[0]
data = []
for i in range(len(net1)):
    data.append(net1[i])
# print(len(data))
# print()
# print(data)
dataname = './data/new_snd.mat'
lable = dataset['true_idx3']

print(lable)
true_idx = []
for value in lable:
    true_idx.append(value[0])
print(true_idx)
scio.savemat(dataname, {'net': data, 'true_idx': true_idx})
print('finished')
print(true_idx)
