
import numpy as numpy
import matplotlib.pyplot as plt

# sales data of past 10 years for 50 restaurant in swiggi app
sales_data = numpy.array([
#    2050   2049   2048   2047   2046   2045   2044   2043   2042   2041    <--- in years

 [1,329114,485992,196337,175400,267810,345228,412990,298774,187665,463120], # abc biriyani point
 [2,154992,387550,451221,221879,306745,498112,132904,467390,256781,392145], # kaji barfi stall
 [3,487223,216890,334556,157994,492210,312874,229765,401288,178399,265478], # chilli restaurant
 [4,276533,368120,442199,124567,237889,415332,395876,281091,147882,354901], # manji momo
 [5,198744,122399,341550,466782,293144,412788,172665,304522,478901,221456], # wow momo
 [6,431220,211944,498120,178660,367450,244918,455872,186704,253980,392611], # biriyani house
 [7,322781,174299,308811,243500,488099,332410,196755,412908,468721,185679],
 [8,257990,382744,495611,312880,138522,289766,467300,177899,235440,354110],
 [9,468800,299532,149880,224991,398122,274550,496088,183499,362711,140922],
 [10,145644,487110,259088,221340,355900,204411,419722,134899,294755,481220],
 [11,274500,311998,487720,246551,122477,381220,433900,195222,344781,409995],
 [12,392244,178640,266799,411200,399544,299977,137880,452910,170511,354200],
 [13,132499,407322,388700,152880,467991,356400,214599,241411,412900,324500],
 [14,498770,221340,154799,300522,386910,273411,452099,314599,122488,280990],
 [15,233400,172811,481940,256710,412099,499200,147732,388511,244710,287990],
 [16,354710,261499,429800,482210,231440,157399,497700,281091,354222,234890],
 [17,214766,392311,398201,199022,344099,450700, 271922,353199,175811,457200],
 [18,399522,218410,433712,342900,197660,321499,167990,298811,487233,389140],
 [19,476831,295700,388299,188990,256411,412220,332911,419988,147722,231400],
 [20,287611,398411,265522,344820,147912,227455,411990,178899,352610,488720],
 [21,389220,214522,444388,212090,387811,199540,364722,299411,478211,166299],
 [22,455110,177822,333998,462210,200488,311899,268400,387221,191140,354911],
 [23,219710,322700,422988,331144,498722,155420,433710,271144,361188,289944],
 [24,196411,344522,498722,402310,283799,136540,487911,197100,302200,419211],
 [25,402899,211144,289800,357211,169990,421533,318411,462122,254720,189533],
 [26,146533,276411,397800,199533,268410,350722,299411,487211,164822,300911],
 [27,286599,321488,225199,431877,395211,187411,176944,364199,422311,172890],
 [28,366410,251122,351799,212800,443210,300188,254990,416799,172844,379122],
 [29,278411,477200,168899,347988,391100,188722,237744,398900,468322,299811],
 [30,324988,219588,466299,171411,422140,391200,196410,312799,385221,261700]
])


# swiggi sales analysis

print("shape of sales data: ",sales_data.shape)
print("sales of first 5 restaurant:\n",sales_data[0:5])
print("sales of first 5 restaurant:\n",sales_data[:, 1:5]) # select all rowa and then given range [1:5] col 1 to 5

print("year wise add: ",numpy.sum(sales_data,axis=0))
# another way to do it
yearly_data = numpy.sum(sales_data[:,1:],axis=0)
print("yearly data: ",yearly_data)

min_sales = numpy.min(sales_data[:, 1:],axis=1)
print("Minimum sales per restaurant: \n",min_sales)

max_sales = numpy.min(sales_data[:, 1:],axis=0)
print("maximum sales per year: \n",max_sales)

avg_sales = numpy.mean(sales_data[:, 1:],axis=1)
print("average sales per restaurant: \n",avg_sales)

cumulative_sales = numpy.cumsum(sales_data[:, 1:],axis=1)
print("cumulative sales per restaurant: \n",cumulative_sales)


plt.figure(figsize=(10,6))
plt.plot(numpy.mean(cumulative_sales,axis=0))
plt.title("Average cumulative sales across restaurants")
# so for first year it will see all 30 restaurats cumulative sale and divide it with 30 to get mean cumulative sale
# and so on for the 3nd and 3rd year
plt.xlabel("years")
plt.ylabel("sales")
plt.grid(True)
plt.show()

print("-----------------------------------------------------------------------------------------------")

# vector operations
vector1 = numpy.array([1,2,3,4])
vector2 = numpy.array([5,6,7,8])

print("vector addition: ",vector1+vector2)

# This is called Hadamard (element-wise) multiplication, not matrix multiplication.
print("vector hadamard multiplication: ",vector1*vector2)

v1 = vector1.reshape(4,1)   # column
v2 = vector2.reshape(1,4)   # row
print("matrix multiplication: ",numpy.matmul(v1, v2))

dot_product = numpy.dot(vector1,vector2)
angle = numpy.arccos(dot_product / (numpy.linalg.norm(vector1)*numpy.linalg.norm(vector2)))
print("Dot product: ",dot_product)
print("angle: ",angle)


print("-----------------------------------------------------------------------------------------------")

# vectorization and broadcasting

arr = numpy.array(["a", "b", "c"])
# arr.upper()   # ❌ doesn’t work
vectorized_upper = numpy.vectorize(str.upper)
print("vectorizaton of upper(): ",vectorized_upper(arr))
