import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "CLIQUE","SUBCLU","P3C","FIRES","CHAMELEOCLUST","SCHISM","STATPC","KYMEROCLUST","PROCLUS","DOC","MINECLUS" ,"INSCY", "MOOSubClust"]
x=[10.5, 9.21, 8.71, 8.14, 8.07, 7.64, 6.5, 6.28, 5.50, 5.14, 4.64, 4.42, 1.25 ]

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