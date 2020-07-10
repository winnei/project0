##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################
import os
import numpy as np
import sklearn
from sklearn.cluster import KMeans

from load_data import load_data

#comment!

#create Class: 
class GraspClustering:
    
    def train(self):
        #load training_data and train - try diff k, get sum of sq errors, store fitted model 
        print('train!') 
        
        training_data_path=os.environ['TRAIN_DATA_PATH'] #an environmental variable
        training_data=load_data(training_data_path) #daya in NUMPY ARRAY format
        
        #GOAL: use kmeans and cluster the points.
        #hyperparameter: no of clusters, k
        k_range=range(1,10) #initialise possible values of k
        sse=[]      #sum of squared error
        diffls=[0]  #list of differences in squared error
        best_k=1    #best no of clusters
        
        for k in k_range:
            cluster=KMeans(n_clusters=k,random_state=5)  #repeat fitting for k range
            cluster.fit(training_data)
            sse.append(cluster.inertia_)  #inertia gives within cluster sum of squares: measure of error
        #print(sse)
        
        for i in range(len(sse)-1):
            diff=sse[i]-sse[i+1] #sse should decrease. diff should be positive
            diffls.append(diff) #but diff should suddenly decrease
            
        for i in range(len(sse)-1):
            if diffls[i+1]<0.1*(diffls[1]):
                break
            else:
                best_k+=1
            
        #print(diffls)
        print(best_k)
        
        fittedmodel=KMeans(n_clusters=best_k,random_state=5)
        fittedmodel.fit(training_data)
        self.fittedmodel=fittedmodel
        
        
    def predict(self,test_data):
        #Calls fitted model
        #loads new data: test_data and predicts cluster labels
        #return 1 column vector with N rows (for N points in test_data)
        print('test!')
        
        fittedmodel=self.fittedmodel
        
        labels=fittedmodel.predict(test_data)
     
        return(labels)