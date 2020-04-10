import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading the csv file
data=pd.read_csv('cpdata.csv')
print(data.head(1))

#creating dummy variable for target i.e label
label= pd.get_dummies(data.label).iloc[: , 1:]
data=pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('lets predict the crop')
print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values

#dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing Desision tree classifier
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()

#fitting the classifier into training test
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
#finding accuracy model
a=accuracy_score(y_test,pred)
print("The accuracy of this model is: ",a*100)

#using firebase to import data to be tested
from firebase import firebase
firebase=firebase.FirebaseApplication('https://cropit-eb156.firebaseio.com/')
tp=firebase.get('/Realtime' ,None)

ah=tp['Air Humidity']
atemp=tp['Air Temp']
shum=tp['Soil Humidity']
pH=tp['Soil pH']
rain=tp['Rainfall']

l=[]
l.append(ah)
l.append(atemp)
l.append(pH)
l.append(rain)
predictcrop=[l]

#putting the names of crop in a single list
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugercane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','graps','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
cr='rice'

#predicting the crop
predictions = clf.predict(predictcrop)
count=0
for i in range(0,30):
	if(predictions[0][i]==1):
		c=crops[i]
		count=count+1
		break;
	i=i+1
if(count==0):
	print('The Predict Crop is %s' %cr)
else:
	print('The Predicted Crop is %s' %c)

#sending the predict crop to dat base
cp=firebase.put('/croppredicted','crop',c)

