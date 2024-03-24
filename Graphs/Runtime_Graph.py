import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "DOC", "INSCY","SUBCLU", "STATPC", "MINECLUS",  "SCHISM", "CLIQUE", "CHAMELEOCLUST", "P3C", "PROCLUS","FIRES", "MOOSubClust", "KYMEROCLUST"]
x=[11.20, 10.35, 9.78, 8.78, 8.28, 7.57, 7.21, 5.78, 5.71, 5.42, 4.57, 3.30, 1.50]

fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Average Accuracy Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot