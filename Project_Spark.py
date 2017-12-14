from pyspark.sql import Row
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
import numpy as np

def sigma(z):
    if np.any(1/(1+np.exp(-z))>=0.5):
        return 1.
    else:
        return 0. 

def map_function(x,W,b):
    list_summ = []
    for i in range(0,len(x[0])):
        list_summ.append((sigma(np.dot(np.transpose(W),np.array(x[0]))+b)-float(x[1]))*x[0][i])
    list_summ.append(sigma(np.dot(np.transpose(W),np.array(x[0]))+b)-float(x[1]))
    return list(list_summ)

def map_function_cost(x,W,b):
    res = 0.0
    y_estim = sigma(np.dot(np.transpose(W),np.array(x[0]))+b)
    y = float(x[1])
    if y_estim == 0:
        res = 1-y
    else:
        res = y*np.log(y_estim)+(1-y)*(1-np.log(y_estim))
    return res
    
def reduce_function_cost(x,y):
    res2 = float(x)+float(y)
    return res2
    
def reduce_function(x,y):
    list_summ2 = []
    for i in range(0,len(x)):
        list_summ2.append(float(y[i])+float(x[i]))
    return list(list_summ2)

def train(df,iterations, learning_rate, lambda_reg):
    #Initialisation
    
    W = np.random.rand(56,1)    
    b = np.random.random()
    #Process

    for it in range(iterations):      
        rdd_predict = df.rdd.map(lambda x: map_function(x,W,b))
        m = rdd_predict.count()
        dW = [(1./m)*h for h in rdd_predict.reduce(reduce_function)]
        for i in range(0,len(dW)-1):
            dW[i] = dW[i] + (lambda_reg/m)*W[i]
            W[i] = W[i] - learning_rate * dW[i]
        broadcastW = sc.broadcast(W)
        db = dW[-1]
        b= b-learning_rate*db
        broadcastb = sc.broadcast(b)
        #tmp1 contain the sum
        cost = -(1/m)*df.rdd.map(lambda x: map_function_cost(x,broadcastW.value,broadcastb.value)).reduce(reduce_function_cost)+lambda_reg/((2.*m)*np.sum(np.power(W,2)))
        print str(cost)
    return (W,b)

def cost_function(y_estim,y,W,lambda_reg):
        summ=0.
        for m in range(0,len(y)):
            if y_estim[m][0]==0:
                summ = summ +(1.-float(y[m][0]))
            else:
                summ = summ + float(y[m][0])*np.log(y_estim[m][0]) + (1.-float(y[m][0]))*(1.-np.log(y_estim[m][0]))
        
        res = (1./len(y))*summ + (lambda_reg/(2.*len(y)))*np.sum(np.power(W,2))
        return res

def predict(W,b,X):
    y_predict= np.zeros((len(X),1))
    for m in range(0,len(X)):
        y_inter = np.dot(np.transpose(W),X[m])
        y_predict[m]=sigma(y_inter[0]+b)
    return y_predict


def accuracy_metric(actual, predicted):
    correct = 0;
    for i in range(len(actual)):
        if float(actual[i]) == predicted[i]:
            correct += 1;
    return correct / float(len(actual)) * 100.0;

inputDF1 = spark.read.format("csv").option("delimiter"," ").load("C:\\Users\\tjass\\Documents\\Parallelize\\spam.txt").drop("_c56").withColumnRenamed("_c57","_c56")
#DF = inputDF1.select([inputDF1[c].cast("Double") for c in inputDF1.columns])

#Normalization
RDD_feature = inputDF1.rdd.map(lambda x: Vectors.dense(x[:len(x)-1]))
scaler1 = StandardScaler(withMean=True, withStd=True).fit(RDD_feature)

features = scaler1.transform(RDD_feature).flatMap(lambda x: Row(x.values.tolist())).zipWithIndex()
label = inputDF1.rdd.flatMap(lambda x : Row(str(x[len(x)-1]))).zipWithIndex()
df_label = label.toDF().withColumnRenamed("_1","label").withColumnRenamed("_2","index")
df_features = features.toDF().withColumnRenamed("_1","features").withColumnRenamed("_2","index")
df = df_features.join(df_label,df_features["index"]==df_label["index"]).drop("index")

(trainingdf, testdf) = df.randomSplit([0.7, 0.3])


(W,b) = train(trainingdf,5,0.1,0.1)

X = testdf.rdd.map(lambda x: x[0]).collect()
Y = testdf.rdd.map(lambda x: float(x[1])).collect()

Y_predict = predict(W,b,X)
print("Accuracy Metric")
print(accuracy_metric(Y,Y_predict))