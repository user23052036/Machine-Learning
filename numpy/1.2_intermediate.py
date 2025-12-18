import numpy as numpy

arr_1d = numpy.arange(11)
print("Original array:\n",arr_1d)

# slicing in 1D array
print("sliced array:\n",arr_1d[3:8])
# with step
print("sliced array:\n",arr_1d[3:8:2]) # start < stop ==> move forward
# backwards
print("sliced array:\n",arr_1d[8:2:-1]) # start > stop  <== move backward


# slicing in 2D array
arr_2d = numpy.array([
                    [1,2,3],
                    [4,5,6],
                    [7,8,9]]) 
print("original matrix:\n",arr_2d)
print("specific element:\n",arr_2d[1,1]) # [row,col]
print("Entire row:\n",arr_2d[2]) # row_indx
print("column:\n",arr_2d[:, 1:3]) # all rows of of 1st to 2nd column
# [:, start:stop]   start,stop for columns (stop is excluded)


print("-----------------------------------------------------------------------------------------------")

# Sorting

unsorted = numpy.array([2,66,8,3,0,12])
print("original array: ",unsorted)
print("Sorted array: ",numpy.sort(unsorted))

array_2d = numpy.array([[13,12,3],[8,2,3],[5,5,98],[2,7,7]])
print("original 2D array\n: ",array_2d)
print("Sorted 2D array by column:\n",numpy.sort(array_2d,axis=0)) # column-wise(top-bottom)
print("Sorted 2D array by row:\n",numpy.sort(array_2d,axis=1)) # row-wise(left-right)

print("-----------------------------------------------------------------------------------------------")

#Filter with mask
# numbers = numpy.array([i for i in range(1,11)])   <--- slow
numbers = numpy.arange(1,11)
mask = numbers%2 == 0
print("type of mask: ",type(mask))
print("mask = ",mask)

even_numbers = numbers[mask]
print("Even numbers using mask: ",even_numbers,end="\n\n")

# where()
where_indx = numpy.where(numbers > 7)
print("type of where: ",type(where_indx))
print("where = ",where_indx)
print("numbers greater than 7 using where: ",numbers[where_indx],end="\n\n")

# transforms the whole array and doesnot return indices in the form of tupil
new_condition = numpy.where(numbers > 5, numbers*2, numbers)
print("new where condition array: ",new_condition)

print("-----------------------------------------------------------------------------------------------")


#Adding and removing data

array1 = numpy.array([1,2,3,8])
array2 = numpy.array([4,5,6,7])

# it will not concatenate it will be just added and shape of array needs to be same
combined = array1 + array2
print("added array: ",combined)

concat = numpy.concatenate((array1,array2)) # takes tupil as input
print("concatenated array: ",concat)

# array compatibility

a = numpy.array([1,2,3])
b = numpy.array([5,6,7])
print("compatibility shape",a.shape == b.shape)


print("-----------------------------------------------------------------------------------------------")


original = numpy.array([[1,2,3],[4,5,6],[7,8,9]])
print("original:\n", original)

# Add row
with_new_row = numpy.vstack((original,[55,44,33]))
print("after adding new row:\n", with_new_row)

# Correct 3×1 column
new_col = numpy.array([[55],
                    [44],
                    [33]])

# Add column
with_new_col = numpy.hstack((original, new_col))
print("after adding new col:\n", with_new_col)


#deletion using delete() returns array after delete
deleted_array_row = numpy.delete(original,1,axis=0) #removes first indexed row
deleted_array_col = numpy.delete(original,1,axis=1) #removes first indexed col