
# coding: utf-8

# In[3]:

import numpy as np
import load_dataset
import timeit
import collections
import timeit
from bisect import bisect
import scipy.spatial.distance as sd


# In[4]:

y_train , x_train= load_dataset.read("training","MNIST")
y_test, x_test= load_dataset.read("testing","MNIST")
x_train = x_train.reshape([60000,28*28])
x_test = x_test.reshape([10000,28*28])


# In[ ]:

distanceMatrix = sd.cdist(x_test,x_train)


# In[ ]:

k = [1,3,5,10,30,50,70,80,90,100]


# In[ ]:

def efficiency(k):
    
    ct =0.00
    cf =0.00
    for i in range(len(x_test)):
        """
        labels = []
        tempDist = list(distanceMatrix[i])
        
        for j in range(k):
            index = np.argmin(tempDist)
            labels.append(y_train[index])
            del tempDist[index]
            
        #print labels
        """
        labels = y_train[np.argpartition(distanceMatrix[i], k)[:k]]
        preictedLabel = collections.Counter(labels).most_common(1)[0][0]
        #print preictedLabel
        #print y_test[i]
        if preictedLabel == y_test[i]:
            ct+= 1
        else:
            cf += 1
    return ct/(ct+cf)
            
                


        
        
        
        


# In[ ]:

y=[]
for nbr in k:
    eff = efficiency(nbr)
    y.append(eff)
    print str(nbr)+" - "+str(eff)
    


# In[ ]:

import matplotlib.pyplot as plt
plt.ylabel("accuracy")
plt.xlabel("number of neighbours")
plt.title("knn Learning Curve")
plt.plot(k,y)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:


    
    
    


# In[ ]:




# In[ ]:


                                                               


# In[ ]:


    


# In[ ]:




# In[ ]:



