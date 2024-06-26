
def No_of_Clusters(M,member):
	Model=M
	membershipList=member
	clusters=0
	for m in range(len(Model)):
		if np.count_nonzero(Model[m]!=0) and np.count_nonzero(membershipList[m]!=0):
			clusters=clusters+1
	return clusters

def FeatureSet(Model,m):
	featureSet=[]
	for i in range (dimension):
		if Model[m][i]!=0:
			featureSet.append(i)
	return set(featureSet)

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

# def dimension_coverage(M,member):
# 	Model=M
# 	membershipList=member
# 	featureSet=[]
# 	featureSet=set(featureSet)
# 	for i in range(len(Model)):
# 		if np.count_nonzero(Model[i]!=0) and np.count_nonzero(membershipList[i]!=0):
# 			featureSet_i=FeatureSet(Model,i)
# 			featureSet=featureSet.union(featureSet_i)
# 	return (len(featureSet)+0.0)/dimension

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

#function to calculate distance between a data and its corrosponding cluster center
def data_clusterDistance(M,S,SD, member):
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




	