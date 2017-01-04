import numpy as np

np.random.seed(9999)

#IBEX perform 2 matrices multiplication?
'''
data:
[[[0 1]  [0 1]  [0 1]]

 [[0 2]  [0 2]  [0 2]]]

weight:
[[[0 1]  [0 1]]

 [[0 2]  [0 2]]

 [[0 3]  [0 3]]]

out:
[[[0 6]  [0 6]]

 [[0 12]  [0 12]]]

'''

data = np.asarray([0,1,0,1,0,1,0,2,0,2,0,2]).reshape(2,3,2)
weight = np.asarray([0,1,0,1,0,2,0,2,0,3,0,3]).reshape(3,2,2)
top = 


# test only for inner product layer
net.blobs['data'].data[...]=np.asarray([1,1,1,2,2,2]).reshape(2,3)
net.params['ip1'][0].data[...]=np.transpose(np.asarray([1,1,2,2,3,3]).reshape(3,2))
net.forward()
