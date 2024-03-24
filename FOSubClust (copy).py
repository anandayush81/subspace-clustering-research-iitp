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
	return (Model_Init,Genotype_Init)

for i in range(0,len(Model_dict)):
	Model_dict[i], Genotype_dict[i]=Initial_Model(list(Model_dict.values()[i]),list(Genotype_dict.values()[i]),sampleData,SDmax,sample_size)

i=0
while i<population_size:
    Modelnew, Genotypenew=OneChild(list(Model_dict.values())[i],list(Genotype_dict.values())[i],sampleData,SDmax,sample_size)
    membershipList= membershipDegree(Modelnew,sampleData) 
    PSM= Calculate_PSM(Modelnew, membershipList)
    ICD= data_clusterDistance(Modelnew,sampleData,SDmax, membershipList) #Intra Cluster Distance
    population[i]=(PSM,ICD) #Putting solution for crowding distance
    Model_dict[i]=Modelnew		#Model corresponding to iteration/(XB,PBM) pair
    Genotype_dict[i]=Genotypenew #genotype corresponding to iteration/(XB,PBM) pair
    i=i+1

 #change subspace according to value of p
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

#change center of chosen cluster according to guassian distribution
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
