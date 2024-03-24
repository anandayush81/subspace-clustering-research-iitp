import sys
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

returnCount=10


arr=[]
population=[]
population_new=[]
population_final=[]
sol=[]
Model_dict={}
genotype_dict={}
Model_final_dict={}
genotype_final_dict={}
Genotype_dict={}

F1_len=0
run=2 # number of times One_Child will run
NoOfClasses =7
dimension =10
ExpCluster= 3 * NoOfClasses
SDmax=35 #10*dimension
#NewSDmax=20 #makeNewClusters SDmax
NewSDmax=10;

sample_size=200 #window size
dataCount=sample_size
init_sample_size=2000 #initial sample size
population_size=10 #population count of solutions

Iter=4000 #initial and makeNewClusters number of iterations
#Iter=50
IterNew=Iter/2
NoOfIteration=2885 #window number of iterations
#NoOfIteration=500
totalLambda=5   #total No of iterations before which only OneChild executes
Evaluation_interval=10 #Evaluation is done after every interval
#membershipList=[] # Sample data belongs to which cluster

membershipListAllData=[]
mean=[] # mean array of data ;each element corresponds to the mean of data of each dimension
deviation=[] #deviation array of data ;each element corresponds to the deviation of data of each dimension
PredLevel=[] #Predicted class labels



trueLevelStaticSeries=[]
trueLevelStatic=[]
PredLevelStatic=[]
PredLevelStaticSeries=[]


#Function to perform normalization on data
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

# Function to Check Empty Modelnew










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
	#Model=Modelnew
	Model=M
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

def Calculate_ICC(M,S,SD, member):
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

#function to calculate Number of cluster
def Cal_Noof_Cluster(SD,size, member):
	membershipList=member
	SDmax=SD
	countCluster=0
	sample_size=size
	for m in range(SDmax):
		for s in range(sample_size):
			if membershipList[m][s]!=0:
				countCluster=countCluster+1
				break
	return countCluster

#Function to calculate Manhattan distance between a cluster centre and a data
def data_minclusterDistance(M,D):
    Model=M
    data=D
    m_final=-1
    minD=(float)(sys.maxint)
    for m in range(SDmax):
        if not Check_Empty_Model(Model,m):
            DisCal=0.0
            for d in range(dimension):
                DisCal+=(float)(abs(Model[m][d]-data[d]))
            if DisCal<minD:
                m_final=m
                minD=DisCal

    return minD,m_final

# Function to calculate Maximum distance between cluster
def MaxClusterDistance(M,S,SD):
    Model=M
    sampleData=S
    SDmax=SD
    MaxCD=0.0
    for i in range(SDmax-1):
		if not Check_Empty_Model(Model,i):
			for j in range(i+1, SDmax):
				if not Check_Empty_Model(Model,j):
					tempDis=0
					for d in range(dimension):
						if (float(Model[i][d])!=0.0 and float(Model[j][d]!=0.0)):
							tempDis=tempDis+ abs(float(Model[i][d])-float(Model[j][d]))
						else:
							tempDis=tempDis+ abs(float(Model[i][d])-0.0)
					if MaxCD<tempDis:
						MaxCD=tempDis
    return MaxCD

# Function to calculate Minimum distance between cluster
def MinClusterDistance(M,S,SD):
    Model=M
    sampleData=S
    SDmax=SD
    MinCD=9999999.0
    for i in range(SDmax-1):
		if not Check_Empty_Model(Model,i):
			for j in range(i+1, SDmax):
				if not Check_Empty_Model(Model,j):
					tempDis=0
					for d in range(dimension):
						if (float(Model[i][d])!=0 and float(Model[j][d])!=0):
							tempDis=tempDis+ abs(float(Model[i][d])-float(Model[j][d]))
						else:
							tempDis=tempDis+ abs(float(Model[i][d])-0.0)
					if tempDis<MinCD:
						MinCD=tempDis
    if MinCD==9999999.0:
		return 0.0
    else:
		return MinCD

#Function to calculate threshold values for all cluster centers

def threshold(M,S,size, member): ## Distance calculation is wrong.. use manhattan distance
    Model=M
    sampleData=S
    membershipList=member
    sample_size=size
    t=[]
    for m in range(len(Model)):
        if not Check_Empty_Model(Model,m) and np.count_nonzero(membershipList[m]!=0):
            Dis=0.0
            for s in range(sample_size):
                if membershipList[m][s]==1:
                    DisCal=0.0
                    for d in range(dimension):
                        DisCal=DisCal+ abs(float(sampleData[s][d])-float(Model[m][d]))
                    if DisCal>Dis:
                    	Dis=DisCal

            t.append((float)(Dis))
        else:
            t.append(0.0)
    threshold_value= max(t)-4.0
    #print threshold_value
    return threshold_value
'''

#Function to calculate threshold values for all cluster centers

def threshold(M,S,size, member): ## Distance calculation is wrong.. use manhattan distance
    Model=M
    sampleData=S
    membershipList=member
    sample_size=size
    t=[]
    Dis=0.0
    count=0
    for m in range(len(Model)):
        if not Check_Empty_Model(Model,m) and np.count_nonzero(membershipList[m]!=0):      
            for s in range(sample_size):
                if membershipList[m][s]==1:
                	
                    DisCal=0.0
                    count=count+1
                    for d in range(dimension):
                        DisCal=DisCal+ abs(float(sampleData[s][d])-float(Model[m][d]))
            		Dis=Dis+DisCal

    threshold_value= (Dis+0.0)/count
    #threshold_value2=threshold1(Model,sampleData,sample_size, membershipList)
    #threshold_value=(threshold_value1+threshold_value2)/(2+0.0)
    print threshold_value
    return threshold_value
'''

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




def membershipDegreeSample(M,S,SD,epoc,memberall):
	Model=M
	sampleData=S
	sample_size=len(sampleData)
	SDmax=SD
	membershipListAll=[]

	
	for i in range(SDmax):
		membershipListAll.append([])
		for j in range(init_sample_size):
			membershipListAll[i].append(memberall[i][j])
	
	for i in range(SDmax):
		for j in range (sample_size):
			membershipListAll[i][((epoc*sample_size)+j)%init_sample_size]= 0

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
				membershipListAll[n][((epoc*sample_size)+s)%init_sample_size]=1
				

	#for i in range (len(membershipListAll)):
		#membershipList[i]= membershipListAll[i].tolist()

	return membershipListAll


#function to calculate distance between data and standard mean
def data_meanDistance(M,S):
    Model=M
    sampleData=S
    sample_size=len(sampleData)
    E1_dis=0.0
    for s in range(sample_size):
		for d in range(dimension):
			E1_dis=E1_dis+ abs(float(sampleData[s][d])-0.0)
    return E1_dis

#function to calculate distance between a data and its corrosponding cluster center
def data_clusterDistance(M,S,SD, member):
	membershipList=member
	Model=M
	sampleData=S
	sample_size=len(sampleData)
	E_c_dis=0.0
	SDmax=SD
	for m in range(SDmax):
		if not Check_Empty_Model(Model,m):
			for s in range(sample_size):
				DisCal=0
				for d in range(dimension):
					if float(Model[m][d])==0.0:
						#print float(sampleData[s][d])
						DisCal=DisCal+ abs(float(sampleData[s][d])-0.0)
					else:
						DisCal=DisCal+ abs(float(sampleData[s][d])-float(Model[m][d]))
				E_c_dis=E_c_dis+membershipList[m][s]*DisCal
	return E_c_dis

#function to calculate PBM index
# def Calculate_PBM(M,S,SD, member):
# 	membershipList=member
# 	Model=M
# 	sampleData=S
# 	sample_size=len(sampleData)
# 	Num_cluster=Cal_Noof_Cluster(SD,sample_size, membershipList)
# 	E1_distance=data_meanDistance(Model,sampleData)
# 	E_c_distance=data_clusterDistance(Model,sampleData,SD, membershipList)
# 	MCD=MaxClusterDistance(Model,sampleData,SD)
# 	if Num_cluster==0 or E_c_distance==0.0 or MCD==0.0:
# 		PBM_value=0.0
# 	else:
# 		PBM_value=(E1_distance*E1_distance*MCD*MCD)/(Num_cluster*Num_cluster*E_c_distance*E_c_distance)

# 	if PBM_value==0.0:
# 		return 0.0
# 	else:
# 		return 1/PBM_value

# #function to calculate XB index
# def Calculate_XB(M,S,SD, member):
# 	membershipList=member
# 	Model=M
# 	sampleData=S
# 	compactDistance=data_clusterDistance(Model,sampleData,SD, membershipList)
# 	mCD=MinClusterDistance(Model,sampleData,SD)
# 	if mCD==0.0:
# 		XB_value=0.0
# 	else:
# 		XB_value=(compactDistance*compactDistance)/sample_size*mCD*mCD
# 	return XB_value

# #Code for crowding distance

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


def non_dominating(P):
	population=P
	F_one_set=[]
	for i in range(2*population_size):
		F_one_set.append([])
	ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=population)
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

# def OneChild(M,G,S, SDmax,size):
# 	weight=0

# 	Model_child=[]
# 	genotype_child=[]

# 	for i in range(SDmax):
# 		Model_child.append([])
# 		genotype_child.append([])
# 		for j in range(dimension):
# 			Model_child[i].append(M[i][j])
# 			genotype_child[i].append(G[i][j])
# 	sampleData=S
# 	sample_size=size
# 	#genotype_child[1]=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
# 	#genotype_child[2]=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
# 	clusterRow=[]
# 	nonClusterRow=[]
# 	weight=np.sum(genotype_child,dtype=float)
# 	if weight==SDmax:
# 		r1=(random.randint(0,SDmax-1))
# 		r2=(random.randint(0,dimension-1))
# 		while genotype_child[r1][r2] == 0:
# 			r1=(random.randint(0,SDmax-1))
# 			r2=(random.randint(0,dimension-1))
# 		genotype_child[r1][r2]=genotype_child[r1][r2]-1
# 		if genotype_child[r1][r2]==0:
# 			Model_child[r1][r2]=0.0
# 	rSample=(random.randint(0,sample_size-1))
# 	rdim=(random.randint(0,dimension-1))
# 	for m in range(SDmax):
# 		geneCount=0
# 		for d in range(dimension):
# 			geneCount=geneCount+genotype_child[m][d]
# 		if geneCount>0:
# 			clusterRow.append(m)
# 		else:
# 			nonClusterRow.append(m)
# 	#print 'Noncluster=',nonClusterRow
# 	#print 'Cluster',clusterRow
# 	prob=random.random()
#     #print len(nonClusterRow)
#     #print "weight=", (1/(weight+0.0)) 
#     #print 'prob=',prob
# 	if weight==0:
# 		weight=1

# 	if (prob <=1/(weight+0.0)):# and len(nonClusterRow)!=0) or (0.0<prob<=.50): #changes made to this line--verify
# 		point=(random.randint(0,len(nonClusterRow)-1))
# 		c=nonClusterRow[point]
# 	else:
# 		point=(random.randint(0,len(clusterRow)-1))
# 		c=clusterRow[point]

# 	genotype_child[c][rdim]=genotype_child[c][rdim]+1
# 	Model_child[c][rdim]=sampleData[rSample][rdim]

# 	#print genotype_child[c][rdim]
# 	#print Model_child[c][rdim] # End opf child function

# 	return Model_child, genotype_child





def OneChild(M ,G,S, SDmax, sample_size):
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
	clusterRow=[]
	nonClusterRow=[]
	#print 'child',M
	for i in range(SDmax):
		for j in range(dimension):
			weight=weight + genotype_child[i][j]
	#print 'weight Before', weight

	if weight>=2 * SDmax:
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

	prob=random.random()

	if weight==0:
		weight=1

	if prob <=(1/(weight+0.0)):
		point=(random.randint(0,len(nonClusterRow)-1))
		c=nonClusterRow[point]
		genotype_child[c][rdim]=genotype_child[c][rdim]+1
		Model_child[c][rdim]=sampleData[rSample][rdim]
	else:
		point=(random.randint(0,len(clusterRow)-1))
		c=clusterRow[point]
		genotype_child[c][rdim]=genotype_child[c][rdim]+1
		Model_child[c][rdim]=sampleData[rSample][rdim]

	global weight_Genome
	weight_Genome =weight

	return Model_child, genotype_child











#change center of chosen cluster according to guassian distribution
# def clusterCenterChange(M,G,S,SDmax):

# 	#count=0

# 	Model_change=[]
# 	genotype_change=[]
# 	for i in range (SDmax):
# 		Model_change.append([])
# 		genotype_change.append([])
# 		for j in range (dimension):
# 			Model_change[i].append(M[i][j])
# 			genotype_change[i].append(G[i][j])

# 	count=0
# 	sampleData=S
# 	sample_size=len(sampleData)
# 	max=np.max(np.max(Model_change))
# 	min=np.min(np.min(Model_change))
# 	r1=(random.randint(0,SDmax-1))
# 	r2=(random.randint(0,dimension-1))
# 	while Model_change[r1][r2]==0.0 and count<10:
# 		r1=(random.randint(0,SDmax-1))
# 		r2=(random.randint(0,dimension-1))
# 		count=count+1

# 	if count>=10:
# 		Model_new,genotype_new=OneChild(Model_change,genotype_change,sampleData,SDmax,sample_size)
# 		return Model_new,genotype_new

# 	tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
# 	while tempval<min or tempval>max:
# 		tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
# 		Model_change[r1][r2]=tempval
# 	return Model_change,genotype_change


def clusterCenterChange(M,G,S,SDmax):
	Model_change=[]
	Genotype_change=[]
	clusterRowSub=[]
	for i in range (SDmax):
		Model_change.append([])
		Genotype_change.append([])
		for j in range (dimension):
			Model_change[i].append(M[i][j])
			Genotype_change[i].append(G[i][j])
	count=0

	Model_change=np.asarray(Model_change)
	for m in range(len(Model_change)):
		if np.count_nonzero(Model_change[m]!=0):
			clusterRowSub.append(m)
	Model_change=Model_change.tolist()

	sampleData=S
	sample_size=len(sampleData)
	max=np.max(np.max(Model_change))
	min=np.min(np.min(Model_change))
	#r1=(random.randint(0,SDmax-1))
	r1=(random.choice(clusterRowSub))
	r2=(random.randint(0,dimension-1))
	while Model_change[r1][r2]==0.0 and count<returnCount:
		#r1=(random.randint(0,SDmax-1))
		r1=(random.choice(clusterRowSub))
		r2=(random.randint(0,dimension-1))
		count=count+1

	if count>=returnCount:
		#Model_new,genotype_new=OneChild(Model_change,Genotype_change,sampleData,SDmax,sample_size)
		rSample=(random.randint(0,sample_size-1))
		rdim=(random.randint(0,dimension-1))
		Model_change[r1][r2]=sampleData[rSample][rdim]
		Genotype_change[r1][r2]=Genotype_change[r1][r2]+1
		return Model_change,Genotype_change

	tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
	while tempval<min or tempval>max:
		tempval=np.random.normal(Model_change[r1][r2],0.1935,1)
	Model_change[r1][r2]=tempval[0]
	return Model_change,Genotype_change




#change subspace according to value of p
# def subspaceChange(M,G,s,p,size,SDmax):

#     Model_sub=[]
#     genotype_sub=[]
#     for i in range (SDmax):
#     	Model_sub.append([])
#     	genotype_sub.append([])
#     	for j in range(dimension):
#     		Model_sub[i].append(M[i][j])
#     		genotype_sub[i].append(G[i][j])

#     sampleData=s
#     sample_size=size
#     #SDmax=SD
#     Prob=p
#     count=0
#     if Prob >=0.0 and Prob <=0.33: #replace
#         r1=(random.randint(0,SDmax-1))
#         r2=(random.randint(0,dimension-1))
#         while Model_sub[r1][r2]==0.0 and count<10:
#             r1=(random.randint(0,SDmax-1))
#             r2=(random.randint(0,dimension-1))
#             count=count+1
#         if count>=10:
#             Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
#             return Model_new,genotype_new
#         rSample=(random.randint(0,sample_size-1))
#         rdim=(random.randint(0,dimension-1))
#         Model_sub[r1][r2]=sampleData[rSample][rdim]
#         genotype_sub[r1][r2]=genotype_sub[r1][r2]+1
#     elif Prob>0.33 and Prob <=0.66: #add
#         r1=(random.randint(0,SDmax-1))
#         r2=(random.randint(0,dimension-1))
#         while Model_sub[r1][r2]!=0.0 and count<10:
#             r1=(random.randint(0,SDmax-1))
#             r2=(random.randint(0,dimension-1))
#             count=count+1
#         if count>=10:
#             Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
#             return Model_new,genotype_new
#         rSample=(random.randint(0,sample_size-1))
#         rdim=(random.randint(0,dimension-1))
#         Model_sub[r1][r2]=sampleData[rSample][rdim]
#         genotype_sub[r1][r2]=genotype_sub[r1][r2]+1
#     else: #delete
#         r1=(random.randint(0,SDmax-1))
#         r2=(random.randint(0,dimension-1))
#         while Model_sub[r1][r2]==0.0 and count<10:
#             r1=(random.randint(0,SDmax-1))
#             r2=(random.randint(0,dimension-1))
#             count=count+1
#         if count>=10:
#             Model_new,genotype_new=OneChild(Model_sub,genotype_sub,sampleData,SDmax,sample_size)
#             return Model_new,genotype_new
#         Model_sub[r1][r2]=0.0
#         genotype_sub[r1][r2]=0 #genotype_sub[r1][r2]-1
#     return Model_sub,genotype_sub




def subspaceChange(M,G,s,p,size,SDmax):
    Model_sub=[]
    Genotype_sub=[]
    clusterRowSub=[]
    for i in range (SDmax):
        Model_sub.append([])
        Genotype_sub.append([])
        for j in range(dimension):
            Model_sub[i].append(M[i][j])
            Genotype_sub[i].append(G[i][j])

    sampleData=s
    sample_size=size
    Model_sub=np.asarray(Model_sub)
    for m in range(len(Model_sub)):
    	if np.count_nonzero(Model_sub[m]!=0):
    		clusterRowSub.append(m)
    Model_sub=Model_sub.tolist()

    Prob=p
    count=0
    if Prob >=0.0 and Prob <=0.33: #replace
        r1=(random.choice(clusterRowSub))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]==0.0 and count<returnCount:
            r1=(random.choice(clusterRowSub))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=returnCount:
            #Model_new,genotype_new=OneChild(Model_sub,Genotype_sub,sampleData,SDmax,sample_size)
            rSample=(random.randint(0,sample_size-1))
            rdim=(random.randint(0,dimension-1))
            Model_sub[r1][r2]=sampleData[rSample][rdim]
            Genotype_sub[r1][r2]=Genotype_sub[r1][r2]+1
            return Model_sub,Genotype_sub
        rSample=(random.randint(0,sample_size-1))
        rdim=(random.randint(0,dimension-1))
        Model_sub[r1][r2]=sampleData[rSample][rdim]
        Genotype_sub[r1][r2]=Genotype_sub[r1][r2]+1
    elif Prob>0.33 and Prob <=0.68: #add
        r1=(random.choice(clusterRowSub))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]!=0.0 and count<returnCount:
            r1=(random.choice(clusterRowSub))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=returnCount:
            #Model_new,genotype_new=OneChild(Model_sub,Genotype_sub,sampleData,SDmax,sample_size)
            Model_sub[r1][r2]=0.0
            Genotype_sub[r1][r2]=0
            # rSample=(random.randint(0,sample_size-1))
            # rdim=(random.randint(0,dimension-1))
            # Model_sub[r1][r2]=sampleData[rSample][rdim]
            # Genotype_sub[r1][r2]=Genotype_sub[r1][r2]+1
            return Model_sub,Genotype_sub
        rSample=(random.randint(0,sample_size-1))
        rdim=(random.randint(0,dimension-1))
        Model_sub[r1][r2]=sampleData[rSample][rdim]
        Genotype_sub[r1][r2]=Genotype_sub[r1][r2]+1
    else: #delete
        r1=(random.choice(clusterRowSub))
        r2=(random.randint(0,dimension-1))
        while Model_sub[r1][r2]==0.0 and count<returnCount:
            r1=(random.choice(clusterRowSub))
            r2=(random.randint(0,dimension-1))
            count=count+1
        if count>=returnCount:
            #Model_new,genotype_new=OneChild(Model_sub,Genotype_sub,sampleData,SDmax,sample_size)
            rSample=(random.randint(0,sample_size-1))
            rdim=(random.randint(0,dimension-1))
            Model_sub[r1][r2]=sampleData[rSample][rdim]
            Genotype_sub[r1][r2]=Genotype_sub[r1][r2]+1
            return Model_sub,Genotype_sub
        Model_sub[r1][r2]=0.0
        Genotype_sub[r1][r2]=0 #genotype_sub[r1][r2]-1
    return Model_sub,Genotype_sub






def deleteEmptyCluster(M, membershipListStream, SDmax, membershipAll): #problem in data, need modification
    Model=[]
    membershipListAll=[]
    for i in range(SDmax):
    	Model.append([])
    	for j in range(dimension):
    		Model[i].append(M[i][j])

    for i in range (SDmax):
    	membershipListAll.append([])
    	for j in range (init_sample_size):
    		membershipListAll[i].append(membershipAll[i][j])



    for m in range(SDmax):
        if np.count_nonzero(membershipListStream[m])==0:
            for j in range(dimension):
                Model[m][j]=0.0
            for k in range(init_sample_size):
            	membershipListAll[m][k]=0
    return Model, membershipListAll

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
            if Prob >= 0.0 and Prob<=0.33: #and Iteration>totalLambda:
                Prob_new=random.random()
                Model_new,genotype_new=subspaceChange(list(Model_dict.values())[i],list(genotype_dict.values())[i],sampleData,Prob_new,sample_size,SDmax)
                membershipList= membershipDegree(Model_new,sampleData,SDmax)
                PSM= Calculate_PSM(Model_new,membershipList)
                ICC= Calculate_ICC(Model_new,sampleData,SDmax,membershipList)
                population_new[i%population_size]=(PSM,ICC)
                Model_dict_new[i%population_size]=Model_new
                genotype_dict_new[i%population_size]=genotype_new

            elif Prob > 0.33 and Prob <=0.66: #and Iteration>totalLambda:
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

def makeNewClusters(M,SD,NoCluster,n):

	#Model=[]
	#for i in range (SDmax):
		#Model.append([])
		#for j in range (dimension):
			#Model[i].append(M[i][j])
    #Model=M
    M_final_dict={}
    G_final_dict={}
    M_dict={}
    G_dict={}
    SDmax=SD
    sampleData=NoCluster
    sample_size=len(sampleData)
    #print "SampleSize=", sample_size

    Model=[]
    for i in range (n):
        Model.append([])
        for j in range(dimension):
            Model[i].append(M[i][j])


    for j in range(population_size):
        Model1=[]
        for i in range(SDmax):
            Model1.append(np.zeros(dimension,dtype=float))
        M_dict[j]=Model1
    for j in range(population_size):
        genotype1=[]
        for i in range(SDmax):
            genotype1.append(np.zeros(dimension,dtype=float))
        G_dict[j]=genotype1

    i=0
    while i<population_size:
        #print('Iteration:',Iteration)
        Mnew, gnew=OneChild(list(M_dict.values())[i],list(G_dict.values())[i],sampleData,SDmax,sample_size)
        for j in range (run/2):
            Mnew, gnew=OneChild(Mnew,gnew,sampleData,SDmax,sample_size)
            j=j+1

        membershipList= membershipDegree(Mnew,sampleData,SDmax) ##check membershipdegree here
        #XB= Calculate_XB(Mnew,sampleData,SDmax, membershipList)
        #PBM= Calculate_PBM(Mnew,sampleData,SDmax, membershipList)
        PSM=Calculate_PSM(Mnew,membershipList)
        ICC=Calculate_ICC(Mnew,sampleData,SDmax, membershipList)
        population[i%population_size]=(PSM,ICC) #Putting solution for crowding distance
        M_dict[i]=Mnew		#Model corresponding to iteration/(XB,PBM) pair
        G_dict[i]=gnew #genotype corresponding to iteration/(XB,PBM) pair
        i=i+1

    Iteration=0
    while Iteration<IterNew:
        #print ("Iteration",Iteration)
        M_new=[]
        g_new=[]
        M_dict_new={}
        G_dict_new={}
        for i in range(population_size):
            Prob=random.random()
            if Prob >= 0.0 and Prob<=0.33 and Iteration>totalLambda:
                Prob_new=random.random()
                M_new,g_new=subspaceChange(list(M_dict.values())[i],list(G_dict.values())[i],sampleData,Prob_new,sample_size,SDmax)
                membershipList= membershipDegree(M_new,sampleData,SDmax)
                #XB= Calculate_XB(M_new,sampleData,SDmax,membershipList)
                #PBM= Calculate_PBM(M_new,sampleData,SDmax,membershipList)
                PSM=Calculate_PSM(M_new,membershipList)
                ICC=Calculate_ICC(M_new,sampleData,SDmax, membershipList)
                population_new[i%population_size]=(PSM,ICC)
                M_dict_new[i%population_size]=M_new
                G_dict_new[i%population_size]=g_new

            elif Prob > 0.33 and Prob <=0.66 and Iteration>totalLambda:
                M_new,g_new=clusterCenterChange(list(M_dict.values())[i],list(G_dict.values())[i],sampleData,SDmax)
                membershipList= membershipDegree(M_new,sampleData,SDmax)
                #XB= Calculate_XB(M_new,sampleData,SDmax,membershipList)
                #PBM= Calculate_PBM(M_new,sampleData,SDmax,membershipList)
                PSM=Calculate_PSM(M_new,membershipList)
                ICC=Calculate_ICC(M_new,sampleData,SDmax, membershipList)
                population_new[i%population_size]=(PSM,ICC)
                M_dict_new[i%population_size]=M_new
                G_dict_new[i%population_size]=g_new
            else:
                M_new, g_new=OneChild(list(M_dict.values())[i],list(G_dict.values())[i],sampleData,SDmax,sample_size)
                membershipList= membershipDegree(M_new,sampleData,SDmax)
                #XB= Calculate_XB(M_new,sampleData,SDmax,membershipList)
                #PBM= Calculate_PBM(M_new,sampleData,SDmax,membershipList)
                PSM=Calculate_PSM(M_new,membershipList)
                ICC=Calculate_ICC(M_new,sampleData,SDmax, membershipList)
                
                population_new[i%population_size]=(PSM,ICC)
                M_dict_new[i%population_size]=M_new
                G_dict_new[i%population_size]=g_new

        for k in range(population_size):
            population_final[k]=population[k]
            M_final_dict[k]=M_dict[k]
            G_final_dict[k]=G_dict[k]

        for k in range(population_size):
            population_final[k+population_size]=population_new[k]
            M_final_dict[k+population_size]=M_dict_new[k]
            G_final_dict[k+population_size]=G_dict_new[k]
            #print ("Population final:",population_final)
        F=non_dominating(population_final)
        GenerateSolution(F) ## check here sol
        for k in range(population_size):
            index=population_final.index(sol[k])
            M_dict[k]=M_final_dict[index]
            population[k]=population_final[index]
            G_dict[k]=G_final_dict[index]
        Iteration=Iteration+1

    M_Selected=[]
    minD=(float)(sys.maxint)
    for i in range(F1_len):
        membershipList= membershipDegree(list(Model_dict.values())[i],sampleData,SDmax) #### Newly Added  init_sampleData is replaced by sampleData
        dis=data_clusterDistance(list(M_dict.values())[i],sampleData,SDmax,membershipList)
        if dis<minD:
            minD=dis
            M_Selected=list(M_dict.values())[i]

    #print M_Selected
    for i in range(SDmax):
        if np.count_nonzero(M_Selected[i])!=0: #to ensure cluster centers as origin are not added
            k=0
            while not Check_Empty_Model(Model,k):
                k=k+1
                if k==n:
                    return Model
            Model[k]=M_Selected[i]
    #print Model
    return Model

def BUILDSUBSPACECLUSTERS(data, M):

	Model=M
	allData=data

	for i in range(SDmax):
		for j in range(dataCount):
			membershipListAllData[i][j]=0

	#print membershipListAllData

	for s in range(dataCount):
		currentCluster=0
		dis=9999999.0
		for m in range(SDmax):
			if not Check_Empty_Model(Model,m):
				newDis=0.0
				for d in range(dimension):
					if float(Model[m][d])==0.0:
						newDis=newDis+ abs(float(allData[s][d])-0.0)
					else:
						newDis=newDis+ abs(float(allData[s][d])-float(Model[m][d]))
				if newDis < dis:
					dis = newDis
					currentCluster=m
		for n in range(SDmax):
			if n==currentCluster and dis != 9999999.0:
				membershipListAllData[n][s]=1

	return membershipListAllData

def Check_Empty_ModelAllData(member):
    membershipListAllDataCopy=[]
    membershipListAllData=member
    for m in range (SDmax):
        for s in range(dataCount):
            if membershipListAllData[m][s]!=0:
                membershipListAllDataCopy.append(membershipListAllData[m])
                break
    #print len(membershipListAllDataCopy)
    return membershipListAllDataCopy
	#print  (membershipListAllDataCopy)






def Check_Empty_ModelAllDataStatic(member):
    membershipListAllDataCopy=[]
    membershipListAllData=member
    for m in range (SDmax):
        for s in range(2000):
            if membershipListAllData[m][s]!=0:
                membershipListAllDataCopy.append(membershipListAllData[m])
                break
    #print len(membershipListAllDataCopy)
    return membershipListAllDataCopy
	#print  (membershipListAllDataCopy)





def FinalcountCluster(member):
	membershipListAllData=member
	cluster=0
	for m in range (SDmax):
		for s in range(dataCount):
			if membershipListAllData[m][s]!=0:
				cluster=cluster+1
				break
	return cluster




# Start of Functions Computes a max weight perfect matching in a bipartite graph
def improveLabels(val):
    """ change the labels, and maintain minSlack.
    """
    for u in S:
        lu[u] -= val
    for v in V:
        if v in T:
            lv[v] += val
        else:
            minSlack[v][0] -= val

def improveMatching(v):
    """ apply the alternating path from v to the root in the tree.
    """
    u = T[v]
    if u in Mu:
        improveMatching(Mu[u])
    Mu[u] = v
    Mv[v] = u

def slack(u,v): return lu[u]+lv[v]-w[u][v]

def augment():
    """ augment the matching, possibly improving the labels on the way.
    """
    while True:
        # select edge (u,v) with u in S, v not in T and min slack
        ((val, u), v) = min([(minSlack[v], v) for v in V if v not in T])
        assert u in S
        if val>0:
            improveLabels(val)
        # now we are sure that (u,v) is saturated
        assert slack(u,v)==0
        T[v] = u                            # add (u,v) to the tree
        if v in Mv:
            u1 = Mv[v]                      # matched edge,
            assert not u1 in S
            S[u1] = True                    # ... add endpoint to tree
            for v in V:                     # maintain minSlack
                if not v in T and minSlack[v][0] > slack(u1,v):
                    minSlack[v] = [slack(u1,v), u1]
        else:
            improveMatching(v)              # v is a free vertex
            return

def maxWeightMatching(weights):
    """ given w, the weight matrix of a complete bipartite graph,
        returns the mappings Mu : U->V ,Mv : V->U encoding the matching
        as well as the value of it.
    """
    global U,V,S,T,Mu,Mv,lu,lv, minSlack, w
    w  = weights
    n  = len(w)
    U  = V = range(n)
    lu = [ max([w[u][v] for v in V]) for u in U]  # start with trivial labels
    lv = [ 0                         for v in V]
    Mu = {}                                       # start with empty matching
    Mv = {}
    while len(Mu)<n:
        free = [u for u in V if u not in Mu]      # choose free vertex u0
        u0 = free[0]
        S = {u0: True}                            # grow tree from u0 on
        T = {}
        minSlack = [[slack(u0,v), u0] for v in V]
        augment()
    #                                    val. of matching is total edge weight
    val = sum(lu)+sum(lv)
    return (Mu, Mv, val)

# End of Function Computes a max weight perfect matching in a bipartite graph

def sameClusterPair(data1,data2,Level):
	bit=0
	for i in range (len(Level)):
		if data1 in Level[i] and data2 in Level[i]:
			bit=1
			break
	if bit==1:
		return 1
	else:
		return 0


def _validity_cluster_checking(found_clusters_effective,
                               threshold_cluster_validity=0.0):
    return found_clusters_effective >= threshold_cluster_validity



def compute_only_accuracy(contingency_table,
                          valid_clusters,
                          found_clusters_effective):
    best_matching_hidden_cluster = contingency_table == contingency_table.max(0)
    #print 'vv',best_matching_hidden_cluster.sum(0)
    best_matching_hidden_cluster_weight = 1. / best_matching_hidden_cluster.sum(0)
    #print best_matching_hidden_cluster_weight
    correctly_predicted_objects = contingency_table * best_matching_hidden_cluster * best_matching_hidden_cluster_weight
    #print 'xxxxx',correctly_predicted_objects
    #print valid_clusters
    correctly_predicted_objects *= valid_clusters
    #print 'ccccc',(correctly_predicted_objects.sum(0))
    return sum(correctly_predicted_objects.sum(0)) * 1. / (sum(found_clusters_effective)+NOT_ZERO_DIVISION_SECURITY)


def accuracy(cluster_hidden=[],
             cluster_found=[],
             threshold_cluster_validity=0.0):
    
    contingency_table = pd.crosstab(cluster_hidden, cluster_found)
    #print contingency_table
    found_clusters_effective = contingency_table.sum(0)
    #print found_clusters_effective
    valid_clusters = _validity_cluster_checking(found_clusters_effective, threshold_cluster_validity)
    #print 'vc',valid_clusters
    return compute_only_accuracy(contingency_table, valid_clusters, found_clusters_effective)

def Evaluation(Model,member):
 
	membershipListAllData=member
	membershipListAllDataCopy=Check_Empty_ModelAllData(membershipListAllData)
	for s in range (dataCount):
		for m in range (SDmax): # SDmax can be replace by len(membershipListAllDataCopy)
			if membershipListAllDataCopy[m][s]==1:
				PredLevel[s]=m
				break

	Total_Cluster=len(membershipListAllDataCopy)
    #print 'Total_cluster2',Total_Cluster

	ClusterPred=[]
	ClusterTrue=[]
	dimensionSet=[]
	PredLevelModel=[]
	dimensionSetIndex=[]

	for i in range(Total_Cluster):
		ClusterPred.append([])

	for i in range(NoOfClasses):# check number of classes
		ClusterTrue.append([])

	for i in range(dataCount):
		ClusterPred[PredLevel[i]].append(i)
		ClusterTrue[trueLevel[i]].append(i)
    #print ClusterTrue
	ClusterTrue1 = filter(None, ClusterTrue)
	ClusterTrue=ClusterTrue1

	for i in range (Total_Cluster):
		dimensionSetIndex.append([])
		print "Points in Cluster", i, "is=", len(ClusterPred[i])
		if (len(ClusterPred[i])<2):
			print "exception"

    #for s in range (dataCount):
	    #for m in range (SDmax):
		    #if membershipListAllData[m][s]==1:
			    #PredLevelModel.append(m)
			    #break

	for i in range (Total_Cluster):
		X=ClusterPred[i][0]
		M=PredLevel[X]
		dimensionSet.append(Model[M])

	for i in range (Total_Cluster):
		for d in range (dimension):
			if dimensionSet[i][d]!=0:
				dimensionSetIndex[i].append(d)

	#### average Number of dimension per subspace cluster####
	Avgdim=0
	for i in range (Total_Cluster):
		Avgdim=Avgdim+len(dimensionSetIndex[i])
	Avgdim=(float)((Avgdim+0.0)/Total_Cluster)

	Total_Time=(time.clock() - start_time)

	return PredLevel, Avgdim, Total_Cluster








# def EvaluationStatic(Model,member):
# 	#PredLevelStatic=[]
# 	membershipListAllDataStatic=[]
# 	membershipListAllDataStatic=member
# 	#membershipListAllDataCopy=Check_Empty_ModelAllData(membershipListAllData)
# 	for s in range (2000):
# 		for m in range (SDmax): # SDmax can be replace by len(membershipListAllDataCopy)
# 			if membershipListAllDataStatic[m][s]==1:
# 				PredLevelStatic[s]=m
# 				break

# 	Total_Cluster=len(membershipListAllDataCopy)
#     #print 'Total_cluster2',Total_Cluster

# 	ClusterPred=[]
# 	ClusterTrue=[]
# 	dimensionSet=[]
# 	PredLevelModel=[]
# 	dimensionSetIndex=[]

# 	for i in range(Total_Cluster):
# 		ClusterPred.append([])

# 	for i in range(NoOfClasses):# check number of classes
# 		ClusterTrue.append([])

# 	for i in range(2000):
# 		ClusterPred[PredLevelStatic[i]].append(i)
# 		ClusterTrue[trueLevelStaticSeries[i]].append(i)
#     #print ClusterTrue
# 	ClusterTrue1 = filter(None, ClusterTrue)
# 	ClusterTrue=ClusterTrue1

# 	# for i in range (Total_Cluster):
# 	# 	dimensionSetIndex.append([])
# 	# 	print "Points in Cluster", i, "is=", len(ClusterPred[i])
# 	# 	if (len(ClusterPred[i])<2):
# 	# 		print "exception"

#     #for s in range (dataCount):
# 	    #for m in range (SDmax):
# 		    #if membershipListAllData[m][s]==1:
# 			    #PredLevelModel.append(m)
# 			    #break

# 	# for i in range (Total_Cluster):
# 	# 	X=ClusterPred[i][0]
# 	# 	M=PredLevel[X]
# 	# 	dimensionSet.append(Model[M])

# 	# for i in range (Total_Cluster):
# 	# 	for d in range (dimension):
# 	# 		if dimensionSet[i][d]!=0:
# 	# 			dimensionSetIndex[i].append(d)

# 	# #### average Number of dimension per subspace cluster####
# 	# Avgdim=0
# 	# for i in range (Total_Cluster):
# 	# 	Avgdim=Avgdim+len(dimensionSetIndex[i])
# 	# Avgdim=(float)((Avgdim+0.0)/Total_Cluster)

# 	# Total_Time=(time.clock() - start_time)

# 	return PredLevelStatic, Total_Cluster








def Initial_Model(Model, Genotype, sampleData, SDmax, sample_size):

	Model_Init=[]
	Genotype_Init=[]

	for i in range(SDmax):
		Model_Init.append([])
		Genotype_Init.append([])
		for j in range(dimension):
			Model_Init[i].append(Model[i][j])
			Genotype_Init[i].append(Genotype[i][j])

	random_list=random.sample(range(0,SDmax),SDmax)
	count=0
	flag=0
	while count<SDmax:
		for i in random_list:
			for j in range(0,dimension):
				prob=random.random()
				if prob>0.5:
					x1=(random.randint(0,sample_size-1))
					x2=(random.randint(0,dimension-1))
					Model_Init[i][j]=sampleData[x1][x2]
					Genotype_Init[i][j]=Genotype_Init[i][j]+1
					count=count+1
					if count>=SDmax:
						flag=1
						break
			if flag==1:
				break
		if flag==1:
			break
	return Model_Init,Genotype_Init


#   *************************           Start of the program           ************************

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


    for word in line.split(',')[(len(line.split(','))-1):len(line.split(','))]:
        trueLevelStatic.append(word)


    arr.append(my_list)
    init_dataCount=init_dataCount+1


trueLevelStaticSeries = [int(x) for x in trueLevelStatic]
# #NoOfClasses=len(set(trueLevel))
trueLevelStaticSeries=pd.Series(trueLevelStaticSeries)
# #print 'true',trueLevel




init_mydata_list_of_list = arr
#init_mydata_list_of_list = np.array(arr, dtype=np.float32)
init_sampleData=init_mydata_list_of_list

#print ('init_sample_size=',init_sampleData)

for i in range(dimension):
    mean.append(0)
    deviation.append(0)

init_sampleData=normalizedData(init_mydata_list_of_list,init_sampleData,init_sample_size)

#print init_sampleData
current_data=init_sampleData

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


#for i in range(SDmax): #List to put Crisp membership degree
    #membershipListAllData.append(np.zeros(dataCount,dtype=float))

for i in range(SDmax):
	membershipListAllData.append([])
	for j in range(dataCount):
		membershipListAllData[i].append(0)
#print membershipListAllData


# i=0
# while i<population_size:
#     #print('Iteration:',Iteration)
#     Modelnew, genotypenew=OneChild(list(Model_dict.values())[i],list(genotype_dict.values())[i],init_sampleData,SDmax,init_sample_size)
#     #for j in range(run):
#     for j in range(SDmax):
#     	Modelnew, genotypenew=OneChild(Modelnew,genotypenew,init_sampleData,SDmax,init_sample_size)
#     	j=j+1

#     membershipList= membershipDegree(Modelnew,init_sampleData,SDmax)
#     #XB= Calculate_XB(Modelnew,init_sampleData,SDmax, membershipList)
#     #PBM= Calculate_PBM(Modelnew,init_sampleData,SDmax,membershipList)
#     #FNR=dimension_non_redundancy(Modelnew,membershipList)
#     #FPC=Feature_Per_Cluster(Modelnew, membershipList)
#     PSM=Calculate_PSM(Modelnew,membershipList)
#     ICC=Calculate_ICC(Modelnew,init_sampleData,SDmax, membershipList)

#     population[i%population_size]=(PSM,ICC) #Putting solution for crowding distance
#     Model_dict[i]=Modelnew		#Model corresponding to iteration/(XB,PBM) pair
#     genotype_dict[i]=genotypenew #genotype corresponding to iteration/(XB,PBM) pair
#     i=i+1
#print '1st=',Model_dict


for i in range(0,len(Genotype_dict)):
	Model_dict[i], genotype_dict[i]=Initial_Model(list(Model_dict.values()[i]),list(genotype_dict.values()[i]),sampleData,SDmax,sample_size)





i=0
while i<population_size:
    Modelnew, genotypenew=OneChild(list(Model_dict.values()[i]),list(genotype_dict.values()[i]),init_sampleData,SDmax,sample_size)
    membershipList= membershipDegree(Modelnew,init_sampleData,SDmax) 
    PSM= Calculate_PSM(Modelnew, membershipList)
    ICC= Calculate_ICC(Modelnew,init_sampleData,SDmax, membershipList) #Intra Cluster Distance
    population[i]=(PSM,ICC) #Putting solution for crowding distance
    Model_dict[i]=Modelnew		#Model corresponding to iteration/(XB,PBM) pair
    Genotype_dict[i]=genotypenew #genotype corresponding to iteration/(XB,PBM) pair
    i=i+1


initialize(init_sampleData,init_sample_size)






# PredLevelStatic, Total_Cluster =EvaluationStatic(Model_dict.values()[0],membershipList)

# Accuracy=accuracy(trueLevelStaticSeries, PredLevelStatic, threshold_cluster_validity=0.0)
# print("Static Accuracy = ",Accuracy)







#print '2nd=',Model_dict

Model_Selected=[]
minD=(float)(sys.maxint)

for i in range(F1_len):
	membershipList= membershipDegree(list(Model_dict.values())[i],init_sampleData,SDmax) #### Newly Added 
	dis=data_clusterDistance(list(Model_dict.values())[i],init_sampleData,SDmax,membershipList)
	if dis<minD:
		minD=dis
		Model_Selected=list(Model_dict.values())[i]

membershipList= membershipDegree(Model_Selected,init_sampleData,SDmax)
#print 'Model_Selected=', Model_Selected
Iteration=0
Threshold=threshold(Model_Selected,init_sampleData,init_sample_size, membershipList)
Evaluation_count=0
count=0
AvgAccuracy=0.0
AvgPurity=0.0
AvgCE=0.0
AvgAvgDim=0.0
AvgTotalClusters=0.0
while Iteration<NoOfIteration:
    #print ("Windows:",Iteration)
    print
    start_time=time.clock()
    dataStreamCount=0
    trueLevel1=[]
    trueLevel=[]
    mydata_list_of_list=[]
    sampleData=[]
    NoCluster=[]
    membershipListAll=[]
    membershipListAll= membershipList
    membershipListStream=[]

    for i in range(SDmax): #List to put Crisp membership degree
        membershipListStream.append(np.zeros(sample_size,dtype=float))
    for i in range (len(membershipListStream)):
        membershipListStream[i]= membershipListStream[i].tolist()


    for i in range(SDmax):
        for j in range (sample_size):
            membershipListAll[i][((Iteration*sample_size)+j)%init_sample_size]= 0

    while dataStreamCount<sample_size:
        line=readfile.readline()
        if not line:
            Iteration=NoOfIteration
            break
        line = line.strip()
        my_list=[]
        for word in line.split(',')[0:len(line.split(','))-1]:
            my_list.append(word)
        for word in line.split(',')[(len(line.split(','))-1):len(line.split(','))]:
            trueLevel1.append(word)
        mydata_list_of_list.append(my_list)
        dataStreamCount=dataStreamCount+1

    trueLevel = [int(x) for x in trueLevel1]
    #NoOfClasses=len(set(trueLevel))
    trueLevel=pd.Series(trueLevel)
    #print 'true',trueLevel
    sampleData=mydata_list_of_list



    sampleData=normalizedData(mydata_list_of_list,sampleData,sample_size)



    remaining=[]
    # for i in range(sample_size):
    # 	remaining.append([])



    for i in range(sample_size):
        current_data[((Iteration*sample_size)+i)%init_sample_size]=sampleData[i]
        MinimumDistance,ClusterCenter=data_minclusterDistance(Model_Selected,sampleData[i])
        #print Threshold[ClusterCenter]
        if MinimumDistance<Threshold:
            membershipListStream[ClusterCenter][i]=1
            membershipListAll[ClusterCenter][((Iteration*sample_size)+i)%init_sample_size]=1
            remaining.append(sampleData[i]) 
        else:
            NoCluster.append(sampleData[i])

    Model_Selected, membershipListAll=deleteEmptyCluster(Model_Selected, membershipListStream, SDmax, membershipListAll)
    
    #print Model_Selected
    if len(NoCluster)!=0:
        Model_Selected=makeNewClusters(Model_Selected,NewSDmax,NoCluster,SDmax)

    membershipremaining=[]
    for i in range(SDmax): 
        membershipremaining.append(np.zeros(sample_size,dtype=float))

    membershipremaining=membershipDegree(Model_Selected,remaining,SDmax)
    Model_Selected=makeNewClusters(Model_Selected,NewSDmax,remaining,SDmax)

    
    membershipListStream=membershipDegree(Model_Selected,sampleData,SDmax)
    membershipListAll=membershipDegreeSample(Model_Selected,sampleData,SDmax,Iteration,membershipListAll)
    Threshold=threshold(Model_Selected,current_data,init_sample_size,membershipListAll)
    #Threshold=threshold(Model_Selected,sampleData,sample_size,membershipListStream)
    print 'No_Cluster', len(NoCluster)
    #print 'threshold=',Threshold
   

    if Evaluation_count==Evaluation_interval-1:
        Evaluation_count=0
        count=count+1
        membershipListStream=BUILDSUBSPACECLUSTERS(sampleData,Model_Selected)
        #Accuracy,Purity,CE,Total_Cluster,AvgDim,Total_Time=Evaluation(Model_Selected)
        PredLevel,AvgDim, Total_Cluster =Evaluation(Model_Selected,membershipListStream)
        PredLevel=pd.Series(PredLevel)
        #print Model_Selected
        Accuracy=accuracy(trueLevel, PredLevel, threshold_cluster_validity=0.0)

        AvgAccuracy=AvgAccuracy+Accuracy
        AvgAvgDim=AvgAvgDim+AvgDim
        AvgTotalClusters=AvgTotalClusters+Total_Cluster
        print
        #print ('Unique=',len(set(trueLevelStatic)))
        print ('Predicted=',len(set(PredLevel)))
        print ('Actual=',len(set(trueLevel)))
        print ('Difference=',len(set(PredLevel))-len(set(trueLevel)))
        print ('Accuracy=',Accuracy)
        print ('NumClusters=',Total_Cluster)
        print ('AvgDim=',AvgDim)
        print ("Windows:",Iteration)
        print "--------------------*********************------------------------"
        
    else:
        Evaluation_count=Evaluation_count+1
    Iteration=Iteration+1

readfile.close()
AvgAccuracy=(float)(AvgAccuracy/count)
AvgAvgDim=(float)(AvgAvgDim/count)
AvgTotalClusters=(float)(AvgTotalClusters/count)
print ('Average Accuracy=',AvgAccuracy)
print ('Avg NumClusters=',AvgTotalClusters)
print ('Average of AvgDim=',AvgAvgDim)