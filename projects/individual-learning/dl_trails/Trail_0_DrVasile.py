#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import random, zeros
from random import rand
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[2]:


from numpy import random, zeros
from random import rand()
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[3]:


from numpy import random, zeros
from random import rand
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[4]:


from numpy import random, zeros
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = random.rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[5]:


from numpy import random, zeros
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = random.rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('General dot product', (toc - tic) * 1e3))


# In[6]:


from numpy import dot, outer, multiply

### VECTORIZED DOT PRODUCT OF VECTORS
tic = process_time()

dot = dot(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### VECTORIZED OUTER PRODUCT
tic = process_time()

outer = outer(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### VECTORIZED ELEMENTWISE MULTIPLICATION
tic = process_time()

mul = multiply(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('elementwise multiplication', (toc - tic) * 1e3))


### VECTORIZED GENERAL DOT PRODUCT
tic = process_time()

dot = dot(W, x1)

toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[7]:


from numpy import dot, outer, multiply

### VECTORIZED DOT PRODUCT OF VECTORS
tic = process_time()

dot_product = dot(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### VECTORIZED OUTER PRODUCT
tic = process_time()

outer_product = outer(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### VECTORIZED ELEMENTWISE MULTIPLICATION
tic = process_time()

multiplication = multiply(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('elementwise multiplication', (toc - tic) * 1e3))


### VECTORIZED GENERAL DOT PRODUCT
tic = process_time()

general_dot_product = dot(W, x1)

toc = process_time()

print('"{}" computation time = {} ms'.format('(general) dot product', (toc - tic) * 1e3))


# In[8]:


from numpy import dot, outer, multiply

### VECTORIZED DOT PRODUCT OF VECTORS
tic = process_time()

dot_product = dot(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### VECTORIZED OUTER PRODUCT
tic = process_time()

outer_product = outer(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### VECTORIZED ELEMENTWISE MULTIPLICATION
tic = process_time()

multiplication = multiply(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('elementwise multiplication', (toc - tic) * 1e3))


### VECTORIZED GENERAL DOT PRODUCT
tic = process_time()

general_dot_product = dot(W, x1)

toc = process_time()

print('"{}" computation time = {} ms'.format('General dot product', (toc - tic) * 1e3))


# In[9]:


import numpy

def softmax(x):
    
    s = exp(x) / numpy.sum(exp(x), axis=1, keepdims=True)
    
    return s


# In[10]:


x = array([[9, 2, 5, 0, 0], [7, 5, 0, 0 ,0]])

print(softmax(x))
# Expected output: [
#      [9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04 1.21052389e-04]
#      [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04 8.01252314e-04]
# ]


# In[11]:


test_string = 'Hello World'

print("Print test string :", test_string)


# In[12]:


from math import exp

def basic_sigmoid(x):
    
    s = 1 / (1 + exp(-1.0 * x))
    
    return s


# In[13]:


basic_sigmoid(4.2)
# Expected output: 0.9852259683067269


# In[14]:


from numpy import exp, array

x = array([1, 2, 3])

print(exp(x)) # The exp() function is being applied for each element of the array.


# In[15]:


def sigmoid(x):
    
    s = 1 / (1 + exp(-1.0 * x))
    
    return s


# In[16]:


x = array([1, 2, 3])

sigmoid(x)
# Expected output: array([0.73105858, 0.88079708, 0.95257413])


# In[17]:


def sigmoid_derivative(x):
    
    s = 1 / (1 + exp(-1.0 * x))
    ds = s * (1 - s)
    
    return ds


# In[18]:


x = array([1, 2, 3])

print(sigmoid_derivative(x))
# Expected output: [0.19661193 0.10499359 0.04517666]


# In[19]:


def image2vector(image):
    
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v


# In[20]:


# The shape of the input image is (3, 3, 2).

image = array([[[ 0.67826139,  0.29380381],
                   [ 0.90714982,  0.52835647],
                   [ 0.4215251 ,  0.45017551]],
                  [[ 0.92814219,  0.96677647],
                   [ 0.85304703,  0.52351845],
                   [ 0.19981397,  0.27417313]],
                  [[ 0.60659855,  0.00533165],
                   [ 0.10820313,  0.49978937],
                   [ 0.34144279,  0.94630077]]])

vector = image2vector(image)
print(vector, '\n')
print(vector.shape)

# Expected output with shape (18, 1): [
#      [0.67826139], [0.29380381], [0.90714982], [0.52835647], [0.4215251], [0.45017551], [0.92814219], [0.96677647], 
#      [0.85304703], [0.52351845], [0.19981397], [0.27417313], [0.60659855], [0.00533165], [0.10820313], [0.49978937],
#      [0.34144279], [0.94630077]
# ]


# In[21]:


from numpy import linalg

def normalizeRows(x):
    
    x_norm = linalg.norm(x, ord=2, axis = 1, keepdims = True)
    x = x / x_norm

    return x


# In[22]:


x = array([[0, 3, 4], [1, 6, 4]])

print(normalizeRows(x))

# Expected output: [
#      [0.0        0.6        0.8       ]
#      [0.13736056 0.82416338 0.54944226]
# ]


# In[23]:


import numpy

def softmax(x):
    
    s = exp(x) / numpy.sum(exp(x), axis=1, keepdims=True)
    
    return s


# In[24]:


x = array([[9, 2, 5, 0, 0], [7, 5, 0, 0 ,0]])

print(softmax(x))
# Expected output: [
#      [9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04 1.21052389e-04]
#      [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04 8.01252314e-04]
# ]


# In[25]:


from numpy import random, zeros
from time import process_time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = random.rand(3, len(x1))

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION
tic = process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### CLASSIC OUTER PRODUCT IMPLEMENTATION
tic = process_time()
outer = zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = process_time()
mul = zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
    
toc = process_time()

print('"{}" computation time = {} ms'.format('Elementwise multiplication', (toc - tic) * 1e3))


### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
tic = process_time()
gdot = zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
        
toc = process_time()

print('"{}" computation time = {} ms'.format('General dot product', (toc - tic) * 1e3))


# In[26]:


from numpy import dot, outer, multiply

### VECTORIZED DOT PRODUCT OF VECTORS
tic = process_time()

dot_product = dot(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Dot product', (toc - tic) * 1e3))


### VECTORIZED OUTER PRODUCT
tic = process_time()

outer_product = outer(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('Outer product', (toc - tic) * 1e3))


### VECTORIZED ELEMENTWISE MULTIPLICATION
tic = process_time()

multiplication = multiply(x1, x2)

toc = process_time()

print('"{}" computation time = {} ms'.format('elementwise multiplication', (toc - tic) * 1e3))


### VECTORIZED GENERAL DOT PRODUCT
tic = process_time()

general_dot_product = dot(W, x1)

toc = process_time()

print('"{}" computation time = {} ms'.format('General dot product', (toc - tic) * 1e3))


# In[27]:


def L1(yhat, y):
    
    loss = numpy.sum(numpy.abs(y - yhat))
    
    return loss


# In[28]:


def loss_fun0(yhat, y):
    
    loss = numpy.sum(numpy.abs(y - yhat))
    
    return loss


# In[29]:


yhat = np.array([.9, 0.2, 0.1, 0.4, 0.9])

y = np.array([1, 0, 0, 1, 1])

print(loss_fun0(yhat, y))
# Expected output: 1.1


# In[30]:


yhat = array([.9, 0.2, 0.1, 0.4, 0.9])

y = array([1, 0, 0, 1, 1])

print(loss_fun0(yhat, y))
# Expected output: 1.1


# In[31]:


def loss_fun1(yhat, y):
    
    loss = numpy.sum(numpy.abs(y - yhat))
    
    return loss


# In[32]:


yhat = array([.9, 0.2, 0.1, 0.4, 0.9])

y = array([1, 0, 0, 1, 1])

print(loss_fun1(yhat, y))
# Expected output: 1.1


# In[33]:


def loss_fun2(yhat, y):
    
    loss = numpy.dot(y - yhat)
    
    return loss


# In[34]:


yhat = array([.9, 0.2, 0.1, 0.4, 0.9])

y = array([1, 0, 0, 1, 1])

print(loss_fun2(yhat,y))

# Expected output: 0.43


# In[35]:


def loss_fun2(yhat, y):
    
    loss = numpy.dot(y - yhat, y - yhat)
    
    return loss


# In[36]:


yhat = array([.9, 0.2, 0.1, 0.4, 0.9])

y = array([1, 0, 0, 1, 1])

print(loss_fun2(yhat,y))

# Expected output: 0.43

