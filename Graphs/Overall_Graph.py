import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]
y = [ "FIRES","SUBCLU","CLIQUE","SCHISM", "P3C","STATPC","INSCY","CHAMELEOCLUST","PROCLUS","DOC","KYMEROCLUST", "FOSubClust","MINECLUS"]
x = [ 8.51, 8.51, 8.23, 7.33, 7.04, 6.31, 5.91, 5.285, 4.91, 4.42, 3.93, 3.91, 3.66]

fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Overall Average Quality Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot