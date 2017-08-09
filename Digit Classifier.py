
# coding: utf-8

# In[40]:

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as pmimg
from sklearn.cross_validation import train_test_split
from sklearn import svm
get_ipython().magic('matplotlib inline')



# In[42]:

images1=pd.read_csv('Desktop/train (1).csv')


# In[6]:

images1.head(5)


# In[7]:

images=images1.iloc[0:5000,1:]
labels=images1.iloc[0:5000,:1]


# In[9]:

labels.head(5)


# In[48]:

train_images,test_images,train_labels,test_labels=train_test_split(images,labels,random_state=10)


# In[49]:

img=train_images.iloc[1].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')


# In[50]:

train_images[train_images>0]=1
test_images[test_images>0]=1


# In[51]:

img=train_images.iloc[1].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='binary')


# In[52]:

test=pd.read_csv('Desktop/test (1).csv')


# In[53]:

test[test>0]=1


# In[54]:

clf=svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
clf.score(test_images,test_labels)


# In[55]:

clf.predict(test[0:5000])


# In[56]:

from sklearn.linear_model import LogisticRegression


# In[59]:

logreg=LogisticRegression()
logreg.fit(train_images,train_labels.values.ravel())
predict=logreg.predict(test_images)


# In[60]:

from sklearn import metrics
metrics.accuracy_score(test_labels,predict)


# In[68]:

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(train_images,train_labels.values.ravel())
pred=knn.predict(test_images)


# In[69]:

from sklearn import metrics
metrics.accuracy_score(test_labels,pred)


# In[71]:

k_range=range(1,30)
accuracy=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_images,train_labels.values.ravel())
    pred=knn.predict(test_images)
    accuracy.append(metrics.accuracy_score(test_labels,pred))


# In[73]:

plt.plot(k_range,accuracy)
plt.xlabel('value of k')
plt.ylabel('accuracy')


# In[88]:

image=images1.iloc[0:5000,1:]
label=images1.iloc[0:5000,0]
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=10)
scores=cross_val_score(knn,image,label,cv=10,scoring='accuracy')
scores.mean()


# In[ ]:

k_range=range(1,30)
accuracy_1=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores_1=cross_val_score(knn,image,label,cv=10,scoring='accuracy')
    accuracy_1.append(scores_1.mean())

    


# In[ ]:

accuracy_1


# In[ ]:

plt.plot(k_range,accuracy_1)
plt.xlabel('k_range')
plt.ylabel('accuracy_1')


# In[ ]:



