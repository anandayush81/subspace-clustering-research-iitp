import sys
import pandas as pd
import random
import math
from datetime import datetime
import time
import numpy as np
from collections import OrderedDict
import pygmo as pg
import pandas as pd
from sklearn.metrics import accuracy_score

start_time = time.clock()
NOT_ZERO_DIVISION_SECURITY = 1e-10

population={}
population_new={}
population_final={}
import pygmo as pg

arr=[]  
#population=[]
#population_new=[]
#population_final=[]
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




#pd.DataFrame(arr, columns=["F1", "F2","F3", "F4","F5", "F6","F7", "F8","F9", "F10"])


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



#pd.DataFrame(init_sampleData, columns=["F1", "F2","F3", "F4","F5", "F6","F7", "F8","F9", "F10"])


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




def Check_Empty_Model(M,row_num):
	Model=M
	count=0
	for d in range(dimension):
		if Model[row_num][d]!=0:
			break
		else:
			count=count+1
	if count==dimension:
		return True
	else:
		return False



def membershipDegree(M,S,SD):
	Model=M
	sampleData=S
	sample_size=len(sampleData)
	SDmax=SD
	membershipList=[]

	for i in range(SDmax): #List to put Crisp membership degree #commented now
		membershipList.append(np.zeros(sample_size,dtype=float))
	#print membershipList
	#for i in range(SDmax):
		#for j in range(sample_size):
			#membershipList[i][j]=0
	for s in range(sample_size):
		currentCluster=0
		dis=9999999.0
		for m in range (SDmax):
			newDis=0.0
			if not Check_Empty_Model(Model,m):
				newDis=0.0
				for d in range (dimension):
					if float(Model[m][d])==0.0:
						newDis=newDis + abs(float(sampleData[s][d])-0.0)
					else:
						newDis=newDis+abs(float(sampleData[s][d])-float(Model[m][d]))
				if newDis<dis:
					dis=newDis
					currentCluster=m
		for n in range(SDmax):
			if n==currentCluster and dis != 9999999.0:
				membershipList[n][s]=1

	for i in range (len(membershipList)):
		membershipList[i]= membershipList[i].tolist()

	return membershipList


#membershipList= membershipDegree(Modelnew,init_sampleData,SDmax)


def FeatureSet(Model,m):
	featureSet=[]
	for i in range (dimension):
		if Model[m][i]!=0:
			featureSet.append(i)
	return set(featureSet)


def No_of_Clusters(M,member):
	Model=Modelnew
	membershipList=member
	clusters=0
	for m in range(len(Model)):
		if np.count_nonzero(Model[m]!=0) and np.count_nonzero(membershipList[m]!=0):
			clusters=clusters+1
	return clusters

def dimension_non_redundancy(M,member): 
	Model=M
	membershipList=member
	featureSet=0
	Total_Cluster= No_of_Clusters(Model,membershipList)
	for i in range(len(Model)-1):
		if np.count_nonzero(Model[i]!=0) and np.count_nonzero(membershipList[i]!=0):
			featureSet_i=FeatureSet(Model,i)
			for j in range(i+1, len(Model)):
				if np.count_nonzero(Model[j]!=0) and np.count_nonzero(membershipList[j]!=0):
					featureSet_j=FeatureSet(Model,j)
					#featureSet+=(len(featureSet_i.intersection(featureSet_j)))/(dimension+0.0)  # Modified # confuse/ doubt
					featureSet+=len(featureSet_i.intersection(featureSet_j))
	return ((featureSet * 2)/((0.0+NOT_ZERO_DIVISION_SECURITY)+Total_Cluster*(Total_Cluster-1)))






def Feature_Per_Cluster(Model, membershipList):
	elements=0
	Total_Cluster=No_of_Clusters(Model,membershipList)
	for i in range(len(Model)):
		elements=elements+np.count_nonzero(Model[i])
	FPC=(elements+0.0)/Total_Cluster
	FPC=abs((dimension/2)-FPC)
	return FPC  #/(dimension+0.0)


def Calculate_PSM(M,member):
	Model=M
	membershipList=member
	Model=np.asarray(Model)
	membershipList=np.asarray(membershipList)

	DNR=dimension_non_redundancy(Model,membershipList)
	#DC=dimension_coverage(Model,membershipList)
	FPC=Feature_Per_Cluster(Model, membershipList)
	PSM=DNR+FPC
	return PSM

#PSM=Calculate_PSM(Modelnew,membershipList)
#print(PSM)

def calculate_ICC(M,S,SD, member):
	membershipList=member
	Model=M
	Model=np.asarray(Model)
	membershipList=np.asarray(membershipList)
	sampleData=S
	sample_size=len(sampleData)
	SDmax=SD
	compact=0.0
	flag=0
	NoCluster=0
	for m in range(SDmax):
		E_c_dis=0.0
		if np.count_nonzero(Model[m]!=0) and np.count_nonzero(membershipList[m]!=0):
			points=np.count_nonzero(membershipList[m])
			NoCluster=NoCluster+1
			flag=1
			for s in range(len(sampleData)):
				if (membershipList[m][s]!=0):
					DisCal=0
					for d in range(dimension):
						if float(Model[m][d])==0.0:
							DisCal=DisCal+ abs(float(sampleData[s][d])-0.0)
						else:
							DisCal=DisCal+ abs(float(sampleData[s][d])-float(Model[m][d]))
					E_c_dis=E_c_dis+(membershipList[m][s]*DisCal)
		if (flag==1):
			compact=compact+(E_c_dis/(points+0.0))
			flag=0 

	return compact/(NoCluster+0.0)







def initialize(S,size):

    sampleData=S
    sample_size=size
    Iteration=0
    while Iteration<Iter:
        print ("Iteration",Iteration)
        Model_new=[]
        genotype_new=[]
        Model_dict_new={}
        genotype_dict_new={}
        for i in range(population_size):
            Prob=random.random()
            if Prob >= 0.0 and Prob<=0.15: #and Iteration>totalLambda:
                Prob_new=random.random()
                Model_new,genotype_new=subspaceChange(list(Model_dict.values())[i],list(genotype_dict.values())[i],sampleData,Prob_new,sample_size,SDmax)
                membershipList= membershipDegree(Model_new,sampleData,SDmax)
                PSM= Calculate_PSM(Model_new,membershipList)
                PBM= Calculate_ICC(Model_new,sampleData,SDmax,membershipList)
                population_new[i%population_size]=(PSM,ICC)
                Model_dict_new[i%population_size]=Model_new
                genotype_dict_new[i%population_size]=genotype_new

            elif Prob > 0.15 and Prob <=0.3: #and Iteration>totalLambda:
                Model_new,genotype_new=clusterCenterChange(list(Model_dict.values())[i],list(genotype_dict.values())[i],sampleData,SDmax)
                membershipList= membershipDegree(Model_new,sampleData,SDmax)
                PSM= Calculate_PSM(Model_new,membershipList)
                ICC= Calculate_ICC(Model_new,sampleData,SDmax,membershipList)
                population_new[i%population_size]=(PSM,ICC)
                Model_dict_new[i%population_size]=Model_new
                genotype_dict_new[i%population_size]=genotype_new
            else:
                Model_new, genotype_new=OneChild(list(Model_dict.values())[i],list(genotype_dict.values())[i],sampleData,SDmax,sample_size)
                membershipList= membershipDegree(Model_new,sampleData,SDmax)
                PSM= Calculate_PSM(Model_new,membershipList)
                ICC= Calculate_ICC(Model_new,sampleData,SDmax,membershipList)
                population_new[i%population_size]=(PSM,ICC)
                Model_dict_new[i%population_size]=Model_new
                genotype_dict_new[i%population_size]=genotype_new

        for k in range(population_size):
            population_final[k]=population[k]
            Model_final_dict[k]=Model_dict[k]
            genotype_final_dict[k]=genotype_dict[k]

        for k in range(population_size):
            population_final[k+population_size]=population_new[k]
            Model_final_dict[k+population_size]=Model_dict_new[k]
            genotype_final_dict[k+population_size]=genotype_dict_new[k]
            #print ("Population final:",population_final)
        F=non_dominating(population_final)
        GenerateSolution(F)
        for k in range(population_size):
            index=population_final.index(sol[k])
            Model_dict[k]=Model_final_dict[index]
            population[k]=population_final[index]
            genotype_dict[k]=genotype_final_dict[index]
        Iteration=Iteration+1





def subspaceChange(M,G,s,p,size,SDmax):

    Model_sub=[]
    genotype_sub=[]
    for i in range (SDmax):
    	Model_sub.append([])
    	genotype_sub.append([])
    	for j in range(dimension):
    		Model_sub[i].append(M[i][j])
    		genotype_sub[i].append(G[i][j])

    sampleData=s
    sample_size=size
    #SDmax=SD
    Prob=p
    count=0
    if Prob >=0.0 and Prob <=0.15: #replace
        r1=(random.randint(0,SDmax-1))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]==0.0 and count<10:
            r1=(random.randint(0,SDmax-1))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=10:
            Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
            return Model_new,genotype_new
        rSample=(random.randint(0,sample_size-1))
        rdim=(random.randint(0,dimension-1))
        Model_sub[r1][r2]=sampleData[rSample][rdim]
        genotype_sub[r1][r2]=genotype_sub[r1][r2]+1
    elif Prob>0.15 and Prob <=0.7: #add
        r1=(random.randint(0,SDmax-1))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]!=0.0 and count<10:
            r1=(random.randint(0,SDmax-1))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=10:
            Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
            return Model_new,genotype_new
        rSample=(random.randint(0,sample_size-1))
        rdim=(random.randint(0,dimension-1))
        Model_sub[r1][r2]=sampleData[rSample][rdim]
        genotype_sub[r1][r2]=genotype_sub[r1][r2]+1
    else: #delete
        r1=(random.randint(0,SDmax-1))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]==0.0 and count<10:
            r1=(random.randint(0,SDmax-1))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=10:
            Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
            return Model_new,genotype_new
        Model_sub[r1][r2]=0.0
        genotype_sub[r1][r2]=0 #genotype_sub[r1][r2]-1
    return Model_sub,genotype_sub





def clusterCenterChange(M,G,S,SDmax):

	#count=0

	Model_change=[]
	genotype_change=[]
	for i in range (SDmax):
		Model_change.append([])
		genotype_change.append([])
		for j in range (dimension):
			Model_change[i].append(M[i][j])
			genotype_change[i].append(G[i][j])

	count=0
	sampleData=S
	sample_size=len(sampleData)
	max=np.max(np.max(Model_change))
	min=np.min(np.min(Model_change))
	r1=(random.randint(0,SDmax-1))
	r2=(random.randint(0,dimension-1))
	while Model_change[r1][r2]==0.0 and count<10:
		r1=(random.randint(0,SDmax-1))
		r2=(random.randint(0,dimension-1))
		count=count+1

	if count>=10:
		Model_new,genotype_new=OneChild(Model_change,genotype_change,sampleData,SDmax,sample_size)
		return Model_new,genotype_new

	tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
	while tempval<min or tempval>max:
		tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
		Model_change[r1][r2]=tempval
	return Model_change,genotype_change



def non_dominating(P):
	population=P
	F_one_set=[]
	for i in range(2*population_size):
		F_one_set.append([])
	ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=population)
	#ndr.split(",")
	print(ndf)
	print(dl)
	print(dc)
	print(ndr)
	for i in range(2*population_size):
		F_one_set[ndr[i]].append(population[i])
	F_one_set2 = [x for x in F_one_set if x != []]
	return F_one_set2


def Sort(F,i):
	return (sorted(F,key=lambda x:x[i]))
def Max(F,i):
	return (max(F,key=lambda x:x[i])[i])
def Min(F,i):
	return (min(F,key=lambda x:x[i])[i])

#returns crowding distance array of elements in F in descending order
def crowding_distance(F):
	distance=[]
	for i in range(len(F)):
		distance.append(0)
	noOfObjectives=2
	for i in range (noOfObjectives):
		F_new=Sort(F,i)
		#print (F_new)
		if Max(F_new,i)==Min(F_new,i):
			continue
		distance[0]=99999.0
		distance[len(F)-1]=99999.0
		for j in range (1,len(F)-1):
			distance[j]+=(float)(F_new[j+1][i]-F_new[j-1][i])/(Max(F_new,i)-Min(F_new,i))
	#distance.sort(reverse=True)
	return distance




def GenerateSolution(F):
	global F1_len
	k=population_size
	distance=[]
	i=0
	x=0
	while True:
		if i>=len(F):
			return
		else:
			if len(F[i])>k:
				distance=crowding_distance(F[i])
				#print(distance)
				e=dict()
				for j in range(len(F[i])):
					e[distance[j]]=j
				distance.sort(reverse=True)
				for j in range (k):
					sol[x]=F[i][e.get(distance[j])]
					x=x+1
				break
			else:
				for j in range(len(F[i])):
					sol[x]=F[i][j]
					x=x+1
				k=k-len(F[i])
		if i==0:
			F1_len=len(F[0])
		    #print (F1_len)
		    #print ("F in generate solution",F)
		i=i+1




















#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@








#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



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
init_sampleData=init_mydata_list_of_list

for i in range(dimension):
    mean.append(0)
    deviation.append(0)

init_sampleData=normalizedData(init_mydata_list_of_list,init_sampleData,init_sample_size)



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



for i in range(SDmax):
	membershipListAllData.append([])
	for j in range(dataCount):
		membershipListAllData[i].append(0)
#print membershipListAllData



i=0
while i<population_size:
    #print('Iteration:',Iteration)
    Modelnew, genotypenew=OneChild(list(Model_dict.values())[i],list(genotype_dict.values())[i],init_sampleData,SDmax,init_sample_size)
    for j in range(run):
    	Modelnew, genotypenew=OneChild(Modelnew,genotypenew,init_sampleData,SDmax,init_sample_size)
    	j=j+1

    membershipList= membershipDegree(Modelnew,init_sampleData,SDmax)
    #XB= Calculate_XB(Modelnew,init_sampleData,SDmax, membershipList)
    #PBM= Calculate_PBM(Modelnew,init_sampleData,SDmax,membershipList)
    #FNR=dimension_non_redundancy(Modelnew,membershipList)
    #FPC=Feature_Per_Cluster(Modelnew, membershipList)
    PSM=Calculate_PSM(Modelnew,membershipList)
    ICC=data_clusterDistance(Modelnew,init_sampleData,SDmax, membershipList)

    population[i%population_size]=(PSM,ICC) #Putting solution for crowding distance
    Model_dict[i]=Modelnew		#Model corresponding to iteration/(XB,PBM) pair
    genotype_dict[i]=genotypenew #genotype corresponding to iteration/(XB,PBM) pair
    i=i+1   
    
#print(population)


initialize(init_sampleData,init_sample_size)

print(population_final)


    
    
    
    

    
    




