import numpy as np
import pandas as pd 
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc



def datareading():
	'''
	Read data as csv
	'''
	hazel_df = pd.read_csv("hazelnuts.txt",sep="\t",header=None)
	
	hazel_df = hazel_df.transpose()
	hazel_df.columns = ["sample_id","length","width","thickness","surface_area","mass","compactness",
	                    "hardness","shell_top_radius","water_content","carbohydrate_content","variety"]


	'''
	saperate features from lables 
	'''

	allfeatures = hazel_df.drop(["variety","sample_id"], axis =1) 

	target_feature = hazel_df["variety"]

	return target_feature,allfeatures

'''
this function returns ndarray with 
normlized with zero mean unit variance features
'''

target_feature,allfeatures= datareading()

#Model training
score = list()
missclass =0 
def normalize_input_data(allfeatures):
	
	featuremap = allfeatures.values.astype(float)
	for col in range (featuremap.shape[1]) :
		column = featuremap[:,col]
		minimumvalue = min(column[:5])
		maxvalue = max(column[:5])
		meanvalue = np.mean(column[:5])
		featuremap[:,col]=(column-meanvalue)/(maxvalue-minimumvalue)
	return featuremap

X_norm= normalize_input_data(allfeatures)

Y=list()

y=np.array(target_feature)

label_dict = {'c_avellana':0, 'c_americana':1, 'c_cornuta':2}


for i in y:
    Y.append(label_dict[i])
y=np.array(Y,dtype=int)
y_unique = np.unique(y) 

def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

_label_dict = {0 :'c_avellana' , 1 :'c_americana' , 2 :'c_cornuta'}



def logisticRegression(X, y,theta,num_iter):
    
    #Sigmoid function
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
    
    #Cost function
    def costFunc(theta, X, y, lr = 0.001):
        h = sigmoid(X.dot(theta))
        r = (lr/(2 * len(y))) * np.sum(theta**2)
        return (1 / len(y)) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + r
    
    #Gradient descent function
    def gradientFunc(theta, X, y, lr = 0.001):
        m, n = X.shape
        theta = theta.reshape((n, 1))
        y = y.reshape((m, 1))
        h = sigmoid(X.dot(theta))
        r = lr * theta /m
        return ((1 / m) * X.T.dot(h - y)) + r
    
    #Finding best theta
    for i in range(num_iter):
        lineq = np.dot(X, theta)
        h = sigmoid(lineq)
        #Calculating cost function of each class
        cost = costFunc(theta, X,y) 
        cost = cost.sum(axis = 0)
        #Applying gradient descent to find new theta
        delta = gradientFunc(theta,X,y) 
        theta = theta - delta    
    return theta 






for fold in range(10):
    X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.33)
    #OneVsRest
    i,k,n= 0,3,10 #No of classes and features
    all_theta = np.zeros((k, n))
    for hazelnut in y_unique:
        np_y_train = np.array(y_train == hazelnut, dtype = int)
        best_theta = logisticRegression(X_train, np_y_train, np.zeros((n,1)),10000)
        all_theta[i] = best_theta.T
        i += 1   
    #Predictions
    prediction = sigmoid(X_test.dot(all_theta.T))
    prediction = prediction.tolist()
    pred = list()
    act = list()
    for _i,i in enumerate(prediction):
        pred.append(_label_dict[ i.index(max(i)) ])
        if _label_dict[ i.index(max(i)) ] !=  _label_dict[y_test[_i]] :
            missclass += 1
        act.append(_label_dict[y_test[_i]]) 
    score.append(round(accuracy_score(pred, act)*100,2))
    print("The score for Logistic Regression for fold",fold+1,"is: ",score[fold] ,'%', " No of misclassfied",missclass)
print("The overall score for Logistic Regression is: ", round(sum(score)/len(score),2),'%')


#Writing actual labels and predicted labels to csv file
output=list()
for i in range(len(pred)):
    output.append([pred[i],act[i], 'Matched' if pred[i] == act[i] else 'Unmatched'])
    Result = pd.DataFrame(output, columns=["Predicted Values", "Actual Value", "Matched/Unmatched"])
Result.to_csv('output.csv', header=True, index=False)