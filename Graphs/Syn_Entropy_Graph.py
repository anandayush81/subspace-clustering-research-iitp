import matplotlib.pyplot as plt
plt.plot([10, 8, 9, 8 , 9, 10, 9, 8, 10, 11, 8, 7, 13, 9, 10, 10], [.97, .98, .96, .93, .962, .94, .91, .92, .96, .942, .94, .93, .96, .95, .972, .962], '*') # ro replaced by *
#plt.plot([7, 7, 8, 8 , 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13], [.97, .98, .96, .93, .96, .94, .91, .92, .96, .94, .94, .93, .96, .95, .972, .962], 'ro')
#plt.plot([7, 7, 8, 9, 8 , 9, 10], [.92, .98, .93, .94, .95, .96, 97], 'ro')
plt.axis([6, 14, .90, .99])
plt.xlabel('Number of Clusters',  fontweight='bold' )
plt.ylabel('Entropy', fontweight='bold') # y-axis label
plt.show()
