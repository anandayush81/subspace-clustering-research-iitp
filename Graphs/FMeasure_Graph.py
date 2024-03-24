import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "FIRES", "P3C", "CLIQUE" , "SUBCLU", "FOSubClust", "CHAMELEOCLUST", "STATPC", "SCHISM","INSCY", "PROCLUS",  "KYMEROCLUST",  "DOC", "MINECLUS" ]
x=[9.64, 8.64, 7.28, 7.07, 6.93, 6.57, 6.5, 6.14, 5.36, 5.21, 4.14, 3.57, 2.71]
fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Average F-measure Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot