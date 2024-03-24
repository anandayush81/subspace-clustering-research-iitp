import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "SUBCLU", "CLIQUE","SCHISM","FIRES","INSCY","STATPC","P3C", "DOC", "PROCLUS", "MINECLUS", "KYMEROCLUST", "CHAMELEOCLUST" , "MOOSubClust"]
x=[9.86, 9.42, 9.29, 7.86, 7.14, 6.78, 5.14, 4.86, 4.57, 3.79, 3.357, 3.28, 3.07]

fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Average CE Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot