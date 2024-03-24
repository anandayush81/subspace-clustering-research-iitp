import sys
import pandas as pd
import random
import math
from datetime import datetime
import time
import numpy as np
from collections import OrderedDict
#import pygmo as pg
import pandas as pd
from sklearn.metrics import accuracy_score

start_time = time.clock()
NOT_ZERO_DIVISION_SECURITY = 1e-10

arr=[]  
population=[]
population_new=[]
population_final=[]
sol=[]
Model_dict={}
genotype_dict={}
Model_final_dict={}
genotype_final_dict={}

F1_len=0
run=2 # number of times One_Child will run
NoOfClasses =7
dimension =10
ExpCluster= 3 * NoOfClasses
SDmax=70 #10*dimension
NewSDmax=20 #makeNewClusters SDmax

sample_size=200 #window size
dataCount=sample_size
init_sample_size=2000 #initial sample size
population_size=10 #population count of solutions

Iter=4000 #initial and makeNewClusters number of iterations
IterNew=Iter/2
NoOfIteration=2885 #window number of iterations
totalLambda=5   #total No of iterations before which only OneChild executes
Evaluation_interval=10 #Evaluation is done after every interval
#membershipList=[] # Sample data belongs to which cluster
membershipListAllData=[]
mean=[] # mean array of data ;each element corresponds to the mean of data of each dimension
deviation=[] #deviation array of data ;each element corresponds to the deviation of data of each dimension
PredLevel=[] #Predicted class labels







for i in range(sample_size):
    PredLevel.append(0)

init_dataCount=0
readfile=open('covtype_new.data','r')
while init_dataCount<init_sample_size:
    line=readfile.readline()
    line = line.strip()
    my_list=[]
    for word in line.split(',')[0:len(line.split(','))-1]:
        my_list.append(word)
    arr.append(my_list)
    init_dataCount=init_dataCount+1
init_mydata_list_of_list = arr

pd.DataFrame(arr, columns=["F1", "F2","F3", "F4","F5", "F6","F7", "F8","F9", "F10"])




def normalizedData(M,S,size):
	sample_size=size
	#mydata_list_of_list=M
	mydata_list_of_list=[]
	sampleData1=[]

	for i in range(sample_size):
		sampleData1.append([])
		mydata_list_of_list.append([])
		for j in range(dimension):
			mydata_list_of_list[i].append(M[i][j])
			sampleData1[i].append(S[i][j])
			
	deviatedData_list_of_list=[] # data - mean
	for i in range(sample_size): # Initializing
		deviatedData_list_of_list.append([])
		for j in range(dimension):
			deviatedData_list_of_list[i].append(0)

	for d in range(dimension): # Mean data calculation for each dimension
		add=0.0
		for data in range(sample_size):
			add=add+float(mydata_list_of_list[data][d])
		mean[d]=(float)(add/sample_size)

	for data in range(sample_size): # deviation calculation of each data for each dimension
		for d in range(dimension):
			deviatedData_list_of_list[data][d]=np.square(float(mydata_list_of_list[data][d])-float(mean[d]))

	for d in range(dimension): # Mean deviation
		addition=0.0
		for data in range(sample_size):
			addition+=float(deviatedData_list_of_list[data][d])
		deviation[d]=np.sqrt(addition/ sample_size)

	for i in range(sample_size): # Normalized data
		for j in range(dimension):
			if float(deviation[j])==0.0:
				sampleData1[i][j]=0.0
			else:
				sampleData1[i][j]=(float(mydata_list_of_list[i][j])-float(mean[j]))/float(deviation[j])

	return sampleData1
init_sampleData=init_mydata_list_of_list

for i in range(dimension):
    mean.append(0)
    deviation.append(0)

init_sampleData=normalizedData(init_mydata_list_of_list,init_sampleData,init_sample_size)

pd.DataFrame(init_sampleData, columns=["F1", "F2","F3", "F4","F5", "F6","F7", "F8","F9", "F10"])







Iteration=0
AvgMaxAccuracy=0.0
AvgAvgDim=0.0
AvgMaxCE=0.0
AvgTotalClusters=0

#initializing population_size empty models
for j in range(population_size):
    Model=[]
    for i in range(SDmax):
	       Model.append(np.zeros(dimension,dtype=float))

    for k in range (len(Model)):
    	Model[k]=Model[k].tolist()
    Model_dict[j]=Model

#print len(Model_dict), Model_dict

 #defines the no of data in each dimension
for j in range(population_size):
    genotype=[]
    for i in range(SDmax):
	       genotype.append(np.zeros(dimension,dtype=float))
    #genotype_dict[j]=genotype
    for k in range (len(genotype)):
    	genotype[k]=genotype[k].tolist()
    genotype_dict[j]=genotype

#print len(Model_dict), genotype_dict

for i in range(population_size):
	sol.append((0,0))

for i in range(population_size):
	population.append((0,0))
	population_new.append((0,0))

for i in range(2*population_size):
	population_final.append((0,0))
    
    
    
    
    
    

    
    
def OneChild(M,G,S, SDmax,size):
	weight=0

	Model_child=[]
	genotype_child=[]

	for i in range(SDmax):
		Model_child.append([])
		genotype_child.append([])
		for j in range(dimension):
			Model_child[i].append(M[i][j])
			genotype_child[i].append(G[i][j])
	sampleData=S
	sample_size=size
	#genotype_child[1]=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
	#genotype_child[2]=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
	clusterRow=[]
	nonClusterRow=[]
	weight=np.sum(genotype_child,dtype=float)
	if weight==SDmax:
		r1=(random.randint(0,SDmax-1))
		r2=(random.randint(0,dimension-1))
		while genotype_child[r1][r2] == 0:
			r1=(random.randint(0,SDmax-1))
			r2=(random.randint(0,dimension-1))
		genotype_child[r1][r2]=genotype_child[r1][r2]-1
		if genotype_child[r1][r2]==0:
			Model_child[r1][r2]=0.0
	rSample=(random.randint(0,sample_size-1))
	rdim=(random.randint(0,dimension-1))
	for m in range(SDmax):
		geneCount=0
		for d in range(dimension):
			geneCount=geneCount+genotype_child[m][d]
		if geneCount>0:
			clusterRow.append(m)
		else:
			nonClusterRow.append(m)
	#print 'Noncluster=',nonClusterRow
	#print 'Cluster',clusterRow
	prob=random.random()
    #print len(nonClusterRow)
    #print "weight=", (1/(weight+0.0)) 
    #print 'prob=',prob
	if weight==0:
		weight=1

	if (prob <=1/(weight+0.0) and len(nonClusterRow)!=0) or (0.0<prob<=.50): #changes made to this line--verify
		point=(random.randint(0,len(nonClusterRow)-1))
		c=nonClusterRow[point]
	else:
		point=(random.randint(0,len(clusterRow)-1))
		c=clusterRow[point]

	genotype_child[c][rdim]=genotype_child[c][rdim]+1
	Model_child[c][rdim]=sampleData[rSample][rdim]

	#print genotype_child[c][rdim]
	#print Model_child[c][rdim] # End opf child function

	return Model_child, genotype_child




#print()
i=0
while i<population_size:
    #print('Iteration:',Iteration)
    Modelnew, genotypenew=OneChild(list(Model_dict.values())[i],list(genotype_dict.values())[i],init_sampleData,SDmax,init_sample_size)
    for j in range(run):
    	Modelnew, genotypenew=OneChild(Modelnew,genotypenew,init_sampleData,SDmax,init_sample_size)
    	j=j+1
print(Modelnew)
print(genotypenew)