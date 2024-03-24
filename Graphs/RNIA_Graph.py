import matplotlib.pyplot as plt
import numpy

xValues=[1,2,3,4,5,6,7,8,9,10,11,12,13]

y = [ "CLIQUE", "SUBCLU", "FIRES","SCHISM", "INSCY","STATPC","P3C", "PROCLUS","MOOSubClust","CHAMELEOCLUST", "DOC",  "KYMEROCLUST", "MINECLUS"  ]
x=[ 10.35, 10.35, 9.71, 9.21, 7.57, 7.07, 6.36, 5.36, 5.21, 4.36, 4.00, 3.43, 2.5 ]

fig = plt.figure()
ax = fig.gca()

#ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 13, 1))
 
plt.scatter(x, y, color= "green", marker= "*", s=30) # plotting points as a scatter plot
#plt.scatter(x, y, label= "stars", color= "green", marker= "o", s=30)
 
#plt.xlabel('Average Inverse RNIA Rank',  fontweight='bold' ) # x-axis label
plt.xticks(xValues)
#plt.ylabel('y - axis') # y-axis label

#plt.title('My scatter plot!') # plot title
plt.legend() # showing legend
plt.grid(axis='x')
plt.show() # function to show the plot