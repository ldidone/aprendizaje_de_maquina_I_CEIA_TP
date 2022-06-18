# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:03:52 2022

@author: 10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import skew



def metrics_ (y_hat, num_vp, num_fp ,y_test_1):

    cont_pos = 0
    cont_neg = 0
    cont = 0
    cont_ = 0
    
    for k in range(len(y_hat)):
        
        if y_hat[k] == list(y_test_1)[k]:
            
            cont = cont + 1
            
            if  list(y_test_1)[k] == num_vp: #SI Y[K] SÍ ES FRAUDE
                
                cont_pos = cont_pos + 1 #VP
            
        else:
            
            if y_hat[k] == num_vp and  list(y_test_1)[k] == num_fp: #SI Y[K] NO ES FRAUDE
                
                cont_neg = cont_neg + 1 #FP
            
            if y_hat[k] == num_fp and  list(y_test_1)[k] == num_vp: 
            
                cont_ = cont_ + 1 #FN
    
    print("Precisión: {}".format(cont_pos/(cont_neg + cont_pos)))
    print("Exactitud: {}".format(cont/len(y_hat)))
    print("Recall: {}".format(cont_pos/(cont_ + cont_pos)))
    print("F1 SCORE : {}".format((2 * (cont_pos/(cont_neg + cont_pos)) * (cont_pos/(cont_ + cont_pos))) / ((cont_pos/(cont_neg + cont_pos)) + (cont_pos/(cont_ + cont_pos))) ))


df = pd.read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')
df["FUEL"].replace({"gasoline": 70, "kerosene": 30, "lpg" : 90, "thinner":  50}, inplace=True)
df.pop("STATUS")

df_v = pd.DataFrame()

for k in range(1,len(df.columns)):
        
        pca = PCA(n_components=k)
        pca.fit_transform(df);
        
        v=pca.singular_values_
        l=[]
        sw=True
        
        for i in range(0,k):
            
          value=sum(v[0:i])/sum(v)
          
          l.append(value)
          
          if value > .9 and sw:
            sw=False
            #print("Son suficiente {} componentes para explicar los 784 atributos con un {}%".format(i,value*100))
            break
        
        index_= [i for i in range(0,k)]
        df_componentes_pca=pd.DataFrame(pca.components_,columns=df.columns,index = index_)
        
        columnas = list(df.columns)
        datos = list(sum(np.abs(pca.components_)))
        str_="Correlación por atributo para {} componentes".format(k)
        df_corr=pd.DataFrame(data = datos,index=columnas,columns=[str_])
        df_corr.style.set_properties(**{'text-align': 'center'})
        df_v["Componentes PCA:"+str(k)] = df_corr[str_]
        #print(df_v)
        

#DISTANCE
#FREQUENCY
#FUEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import time

for i in range(3):
    
    df = pd.read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')
    df["FUEL"].replace({"gasoline": .70, "kerosene": .30, "lpg" : .90, "thinner":  .50}, inplace=True)
    
    df = df.sample(frac=1,random_state=0).reset_index(drop=True)
    
    x = df.tail(50000)
    x = x.drop(columns  = "STATUS")
    print("")
    print("")

    if i == 0:
        t = "FUEL","DISTANCE"
        print(t)
        x = x[["FUEL","DISTANCE"]]
        
        
    else:
        
        if i == 1 :
            t = "DESIBEL","AIRFLOW","FUEL"
            print(t)
            x = x[["DESIBEL","AIRFLOW","FUEL"]]
            
        else:
            
            #x = x.drop(columns =  "DISTANCE")
            print("Todas las columnas")
        
    y = df["STATUS"].tail(50000)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
    model = DecisionTreeClassifier()
    start_time = time.time()
    model.fit(xtrain, ytrain)
    print("--- %s seconds ---" % (time.time() - start_time))
    y_hat = model.predict(xtest)
    metrics_(y_hat, 1,0,ytest)






#fig, axes = plt.subplots(nrows=2, ncols=2)


plt.figure()
df_extinto_0 = df[df['STATUS'] == 0]
skew(df_extinto_0["DISTANCE"])
df_extinto_0["DISTANCE"].plot.density(title = "Distribución de distancia para incendios no apagados")

plt.figure()
df_extinto_1 = df[df['STATUS'] == 1]
skew(df_extinto_1["DISTANCE"])
df_extinto_1["DISTANCE"].plot.density( title = "Distribución de distancia para incendios apagados")


plt.figure()
df_extinto_1 = df[df['STATUS'] == 1]
skew(df_extinto_1["DISTANCE"])
df_extinto_1["DISTANCE"].plot.density()
df_extinto_1 = df[df['STATUS'] == 0]
skew(df_extinto_1["DISTANCE"])
df_extinto_1["DISTANCE"].plot.density()

plt.figure()
df_distance_100 = df[df["DISTANCE"]<=100]
plt.title("Distribución de SATATUS para incendios a menos de 100 unidades de distancia", loc='center', wrap=True)
df_distance_100["STATUS"].plot.density()


list_=[]

cont = 0
cont_neg = 0
for k in range(len(df)):
    
    if df["DISTANCE"][k] < 100 and df["STATUS"][k] == 1 : 
        
        cont = cont +1           
        
    if df["DISTANCE"][k] >= 100 and df["STATUS"][k] == 1 :
        cont_neg = cont_neg + 1
        
print("Porcentaje de registros a menos de 100 unidades de distancia que fueron apagados: {}".format(100*(cont/(cont_neg + cont))))



cont = 0
cont_neg = 0
for k in range(len(df)):
    
    if df["DISTANCE"][k] < 100 and df["STATUS"][k] == 0 : 
        
        cont = cont +1           
        
    if df["DISTANCE"][k] >= 100 and df["STATUS"][k] == 0 :
        cont_neg = cont_neg + 1
        
print("Porcentaje de registros a mas de 100 unidades de distancia que no fueron apagados: {}".format(100*(cont_neg/(cont_neg + cont))))












from sklearn.feature_selection import mutual_info_regression

def custom_mi_reg(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    print(a)
    return  mutual_info_regression(a, b)[0] # should return a float value


df = pd.read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')
df_svc = df.sample(frac=1).copy()
df_svc = df_svc.tail(1000)

del df['STATUS']

df_ = df.corr(method=custom_mi_reg)


del df['SIZE']
del df['FUEL']
del df['FREQUENCY']






#https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python




from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = (np.array([df_svc.FREQUENCY,df_svc.AIRFLOW])).transpose() #ENTRENO SOLO CON 1000
y = np.array(df_svc.STATUS) #ENTRENO SOLO COM 1000

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()


df_test = df_svc.head(5000)
X_test = (np.array([df_test.FREQUENCY,df_test.AIRFLOW])).transpose() 
y_test = np.array(df_test.STATUS)
y_hat = model.predict(X_test)

metrics_(list(y_hat), 1,0,list(y_test))
