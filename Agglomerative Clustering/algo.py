import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def createDataset(k, means, std, size=100):
    data = np.empty( (0,3) )
    for cluster in range(k):
        pts = np.random.normal(means[cluster], std[cluster], size=(size//k,2))
        pts = np.c_[pts, np.ones(size//k)*cluster ]
        data = np.append( data, pts, axis=0)
    np.random.shuffle(data)
    return data
data = createDataset( 4, [ (3,2), (8,8), (2,9), (9,4) ], [ 1.5, 1.1, 1, 1.4] )
for k in range(4):
    temp = data[:,0:2][data[:,2]==k]
    #plt.scatter( temp[:,0], temp[:,1],  )


X_train, X_test, Y_train, Y_test = train_test_split( data[:,0:2], data[:,2], test_size = 0.2, stratify=data[:,2] )

def distance(X1,X2):
	return np.sqrt( np.sum( (X1-X2)**2, axis=1 ) )

def mergeNearest(clusters):
	S_i, S_j = 0, 1
	dis = float('inf')
	for i in range(0,len(clusters)-1):
		for j in range(i+1, len(clusters)):
			m1 = np.expand_dims(np.mean(clusters[i],axis=0),axis=0)
			m2 = np.expand_dims(np.mean(clusters[j],axis=0),axis=0)
			cdis = distance(m1,m2)[0]
			if cdis<dis:
				dis,S_i,S_j = cdis,i,j

	a = clusters.pop(S_j)
	b = clusters.pop(S_i)
	clusters.append( np.append( a, b, axis=0 ) )

initial_custers =  [ np.expand_dims(p1,axis=0) for  p1 in X_train]

while len(initial_custers)!=4:
	mergeNearest(initial_custers)
print([ np.mean(x, axis=0) for x in initial_custers ])