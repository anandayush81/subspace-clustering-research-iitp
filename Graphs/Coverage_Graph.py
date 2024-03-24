import matplotlib.pyplot as plt
import numpy

xValues=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "FIRES","PROCLUS","INSCY","STATPC","P3C","DOC","SCHISM", "SUBCLU", "MINECLUS" , "CLIQUE", "CHAMELEOCLUST", "KYMEROCLUST", "MOOSubClust"]
x=[5.78, 5.71, 4.92, 4.28, 3.71, 2.86, 2.28, 2.14, 1.86, 1, 1, 1, 1 ]

fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Average Inverse Entropy Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot