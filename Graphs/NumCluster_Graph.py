import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "SUBCLU", "SCHISM", "CLIQUE", "INSCY",  "MINECLUS", "STATPC", "DOC",  "KYMEROCLUST", "PROCLUS", "CHAMELEOCLUST" , "MOOSubClust" , "P3C", "FIRES"]
x=[10.57, 9.64, 9.64, 8.71, 6.57, 5.85, 5.42, 5.14, 4.21, 3.5, 3.0, 2.85, 2.78]

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