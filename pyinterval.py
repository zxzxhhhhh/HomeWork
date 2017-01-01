from interval import interval, inf, imath
import numpy as np

np.random.seed(9999)

#>>> interval[1, 2]
#interval([1.0, 2.0])

#>>> interval(1, 2)
#interval([1.0], [2.0])

# initialization of the weights
a = np.random.rand(20*1*5*5)
b = np.random.rand(20*1*5*5)
w = np.asarray([(w1,w2) for w1,w2 in zip(a,b)], dtype=float)
w = w.reshape(20,1,5,5,2)
# a list of tuples


def pool_2d(input, ds)
'''
input  – Input images. Max pooling will be done over the 2 last dimensions.
ds (tuple of length 2) – Factor by which to downscale (vertical ds, horizontal ds). (2,2) will halve the image in each dimension.
'''
	
def conv_2d(input, filter, input_shpe, filter_shape)
'''

    This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).
 Parameters
    ----------
    input: symbolic 4D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input rows, input columns).
        See the optional parameter ``input_shape``.
    filters: symbolic 4D tensor
        Set of filters used in CNN layer of shape
        (output channels, input channels, filter rows, filter columns).
        See the optional parameter ``filter_shape``.
    input_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
    filter_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
'''
#########################################################
def inter_mul(x, y):
	l = [x[0]*y[0], x[1]*y[0], x[0]*y[1], x[1]*y[1]]
	_max = T.max(l)
	_min = T.min(l)
	return (_min, _max)


x = T.vector()
y = T.vector()
out = inter_mul(x,y)
f = theano.function([x,y], out)
f([1,2], [3,4])

#########################################################
def inter_add(x, y):
	return [x[0]+y[0], x[1]+y[1]]

x = T.vector()
y = T.vector()
out = inter_add(x,y)
f = theano.function([x,y], out)
f([1,2], [3,4])
##############################################################
# x,y: (w, h, 2) tensor3?


def inter_gemm(x, y):
	xshape = T.shape(x)
	yshape = T.shape(y)
	out = T.tensor3()
	for i in T.arange(xshape[0]):
		for m in T.arange(yshape[1]):
			for j in T.arange(xshape[1]):	
				inter_add(out[i, m], inter_mul(x[i,j], y[j,m]))
	return out

x = T.tensor3()
y = T.tensor3()
out = inter_gemm(x,y)
f = theano.function([x,y], out)

a = np.asarray(range(8)).reshape(2,2,2)
b = np.asarray(range(4)).reshape(2,1,2)
f([1,2], [3,4])


	
