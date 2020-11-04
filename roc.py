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




X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2)
k=3
n=10
all_theta = np.zeros((k, n))

missclass=0
i=0 

for hazelnut in y_unique:
    np_y_train = np.array(y_train == hazelnut, dtype = int)
    best_theta = logisticRegression(X_train, np_y_train, np.zeros((n,1)),10000)
    all_theta[i] = best_theta.T
    i += 1   



prediction = sigmoid(X_test.dot(all_theta.T))


print (prediction.shape)
prediction = prediction.tolist()
pred = list()
act = list()
lablelist=[0,1,2]
for _i,i in enumerate(prediction):
    pred.append(_label_dict[ i.index(max(i)) ])
    if _label_dict[ i.index(max(i)) ] !=  _label_dict[y_test[_i]] :
        missclass += 1
    act.append(_label_dict[y_test[_i]])
score.append(round(accuracy_score(pred, act)*100,2))



print("The overall score for Logistic Regression is: ", round(sum(score)/len(score),2),'%')

def turning_into_softmax(pred):
	pred = np.asarray(pred).reshape((41,3))
	softmax_pred=list()
	for i in range(len(pred)):
		softmax_pred.append(np.exp(pred[i,:])/sum(np.exp(pred[i,:])))
	return np.asarray(softmax_pred)

   
def roc_estimatio(y_test , y_pred_proba, average="macro",num_class=3):  
    y_test = list(y_test)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    

    for i in range(num_class):
        
        y__ = [y[i] for y in y_pred_proba ]
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y__)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return (fpr,tpr,roc_auc)
	
pred_soft =	turning_into_softmax(prediction) 
for i in range (len(pred_soft)):
	print("sum of row {}".format(sum(pred_soft[i,:])))

fpr , tpr, roc_auc = roc_estimation(y_test,pred_soft)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= 3

plt.figure()

roc_auc = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr,
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc),
         color='navy', linestyle=':', linewidth=4)

plt.show()