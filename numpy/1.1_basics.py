import numpy as np 
import time
import os

os.system("clear")
# we use numpy because list in python are very slow and can take multiple datatypes
# whereas numpy is highly optimized c++ library which performs fast operations over array() and takes 
# similar datas as input

# Still we can keep different datatypes in numpy array but we generally dont do

print("-----------------------------------------------------------------------------------------------")

lists = [1,2,3,4,5]
print("tupil multiplication: ",lists*2) # duplicate created

array = np.array(lists)
print("array multiplication: ",array*array) # element wise multiplication


print("-----------------------------------------------------------------------------------------------")

arr_id = np.array([1,2,3,4,5,6]) # array in numpy takes list from python as input []
print("\n\n1D array: ",arr_id)

arr_2d = np.array([[1,2,3],[4,5,6]])
print("2D array:\n",arr_2d)

print("-----------------------------------------------------------------------------------------------")

start1 = time.time()

new_list = [i*2 for i in range(1000220)]
print("Time taken for list calc: ",time.time()-start1)

start2 = time.time()
numpy_array = np.arange(1000220)*2
print("Time taken for array calc: ",time.time()-start2)

print("-----------------------------------------------------------------------------------------------")

zeros1 = np.zeros(10,int)
zeros2 = np.zeros((3,4),int)
print("zero array:\n ",zeros1)
print("zero array:\n ",zeros2)


ones1 = np.ones(10,int)
ones2 = np.ones((3,4),int)
print("ones array:\n ",ones1)
print("ones array:\n ",ones2)

full = np.full((3,4),7,int)
print("full array:\n ",full)

print("-----------------------------------------------------------------------------------------------")

random = np.random.random((4,5))
print("Radom array:\n ",random)

sequence = np.arange(1,20,2)
print("sequence array:\n ",sequence)

print("-----------------------------------------------------------------------------------------------")

# Vector Matrix and Tensor

vector = np.array([1,2,3,4,5]) # 1D == vector
print("vector: \n",vector)

matrix = np.array([[1,2,3,4],
                   [5,6,7,8],
                  [33,5,7,10]])   # 2D == matrix
print("matrix: \n",matrix)



Tensor = np.array([[[1,2,3],[4,5,6]], # >=3D == tensor
                    [[4,3,7],[1,7,8]],
                    [[3,5,4],[8,7,6]]])
print("Tensor: \n",Tensor)


print("-----------------------------------------------------------------------------------------------")

#Array Properties

arr = np.array([[1,2,3],
                [4,5,6],
                [44,6,5]]) # they dont allow inhomogeneous shape, i.e allways has to be square or rectangle matrix

print("shape: ",arr.shape)
print("dimention: ",arr.ndim)
print("size: ",arr.size) # no of elements present
print("data type: ",arr.dtype)


print("-----------------------------------------------------------------------------------------------")

#Array reshaping

arr = np.arange(16)
print("original array: \n",arr)

reshaped = arr.reshape((4,4)) # need to be exactly no of elements = rows*col otherwise throws error
print("reshaped array: \n",reshaped)

flattened = reshaped.flatten()
print("flattened array: \n",flattened)

raveled = reshaped.ravel()
print("raveled array: \n",raveled)

print("-----------------------------------------------------------------------------------------------")

#Transpose of a matrix     rows <==> cols

Transpose = reshaped.T
print("Transposed matrix: \n",Transpose)




