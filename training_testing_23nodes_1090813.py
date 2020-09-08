"""
Last review on Thu Aug 13 11:58:13 2020

@author: Ronald Chou
"""
import numpy as np
import os
from PIL import Image  #from套件import函式
import matplotlib.pyplot as plt
import math
import random
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

class NN:
  def __init__(self, NI, NH, NO):
    # number of nodes in layers 數量  
    self.ni = NI + 1 #+1 for bias
    self.nh = NH
    self.no = NO
    
    # initialize node-activations值
    self.ai, self.ah, self.ao = [],[],[]
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    # create node weight matrices wi=weight input ,wo =weight output
    #self.wi = makeMatrix (self.ni, self.nh)
    #self.bi = biasMatrix (self.nh)
    #self.wo = makeMatrix (self.nh, self.no)
    #self.bo = biasMatrix (self.no)
    self.wi = np.loadtxt(open("23wi.csv","rb"),delimiter=",",skiprows=0)
    self.wo = np.loadtxt(open("23wo.csv","rb"),delimiter=",",skiprows=0)
    
    #randomizeMatrix ( self.wi, -0.2, 0.2 )
    #randomizeMatrix ( self.wo, -0.2, 0.2 )
    # create last change in weights matrices for momentum
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)
    
  def runNN (self, inputs):
    if len(inputs) != self.ni-1:
      print('incorrect number of inputs')
    
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = tanh (sum)
    
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = tanh (sum)
      
    return self.ao

  def backPropagate (self, targets, N, M):
    #param targets：實例的類別
    #param N：本次學習率
    #param M：上次學習率
    #return：MSE
    output_deltas = [0.0] * self.no 
    for k in range(self.no):
      error = targets[k] - self.ao[k] #dE/da
      output_deltas[k] =  error * dtanh(self.ao[k])   #dE/da*da/dz=dE/dz
      #print("output_deltas[k]:",output_deltas[k])  
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]  #dE/dz*dz/dw
        #print("change:", change)
        self.wo[j][k] += N*change + M*self.co[j][k]
        #print("self.wo[%s][%s]:"%(j,k),self.wo[j][k])
        self.co[j][k] = change   #保留當作下次學習
    # update output bias
        #self.bo[k] = N*output_deltas[k]
    # 計算隱藏層deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error1 = 0.0
      for k in range(self.no):
        error1 += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error1 * dtanh(self.ah[j])    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change

    error2 = 0.0
    for k in range(2):
      error2 += round(0.5 * (targets[k]-self.ao[k])**2, 6)
    return error2

  def test(self, test,testspeed,name):
    #param patterns：測試數據
    right = 0
    predict_y = []
    y_true = []
    y_pred = []
    for p in test:
      predict = onehot(self.runNN(p[0]))
      if predict == p[1]:
        right += 1
        if predict[0] == 1:
            predict[0] = 80
            y_true.append("smooth")
            y_pred.append("smooth")
        else : 
            predict[0] = 40
            y_true.append("jam")
            y_pred.append("jam")
      elif predict[0] == 1:
        predict[0] = 80
        y_true.append("jam")
        y_pred.append("smooth")
      else : 
        predict[0] = 40
        y_true.append("smooth")
        y_pred.append("jam")
      predict_y.append(predict[0])
    print('accuracy =', right/len(test))
    x = np.linspace(0,len(test),len(test))
    fig = plt.figure()
    now = datetime.now()
    NOW = now.strftime('%m%d_%H%M')
    plt.title('{}_{} accuracy = {}'.format(NOW,name,round(right/len(test),4))) 
    plt.xlabel('times [5mins per dot]')
    plt.ylabel('Speed [km/hr.]')
    plt.yticks(np.linspace(0, 120, 5))
    plt.plot(x, testspeed, '.-', color = 'SteelBlue', label="Predict", linewidth=1) 
    plt.show()
    fig.savefig('{}_{}accuracy={}.png'.format(NOW,name,right/len(test)))
    cm = confusion_matrix(y_true, y_pred, labels=["smooth", "jam"]) 
    ax = sns.heatmap(cm,annot=True,fmt ='g',cmap="YlGnBu",xticklabels =['smooth','jam'],yticklabels =['smooth','jam'])
    plt.ylabel('True')
    plt.xlabel("Predict")
    plt.title("confuse_matrix_{}".format(name))
    plt.savefig("confuse_matrix_{}_{}.png".format(name,NOW))
    plt.show()
    fig = plt.figure()
    plt.errorbar(x, predict_y, yerr=2, fmt='o', color='SteelBlue', ecolor='LightSteelBlue', elinewidth=2)
    plt.errorbar(x, predict_y, yerr=2, fmt='o', color='SteelBlue', ecolor='LightSteelBlue', elinewidth=2)   
    print(classification_report(y_true, y_pred))
  def train (self, patterns,test,testspeed, max_iterations = 100, N=0.2, M=0.04,name=''):
     #param patterns：訓練集,test:測試集
     #param max_iterations：最大迭代次數
     #param N：本次學習率
     #param M：上次學習率
    for i in range(max_iterations):
      error = 0  
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.runNN(inputs)
        error += self.backPropagate(targets, N, M)
      if i % 2 == 0:
        error = error/len(patterns)
        print ('The %s Combined error:'%i, error)
        self.weights()
    self.test(test,testspeed,name)
    
  def weights(self):
      # 建立 CSV 檔寫入器
      writer = csv.writer(open('23wi.csv', 'w', newline=''))
      # 寫入一列資料
      for i in range(self.ni):
          writer.writerow(self.wi[i])
      writer = csv.writer(open('23wo.csv', 'w', newline=''))
      for i in range(self.nh):
          writer.writerow(self.wo[i])
      
def mask(input_array,csv_name):
    writer = csv.writer(open(csv_name, 'a+', newline=''),delimiter=",")
    sobel_x = np.array(([-1,-2,-1],
                        [ 0, 0, 0],
                        [ 1, 2, 1]))
    sobel_y = np.array(([-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]))
    sobel_xy= np.array(([-1,-1, 0],
                        [-1, 0, 1],
                        [ 0, 1, 1]))
    prewitt_x = np.array(([-1,-1,-1],
                          [ 0, 0, 0],
                          [ 1, 1, 1]))
    prewitt_y = np.array(([-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]))
    prewitt_xy= np.array(([-2,-1, 0],
                          [-1, 0, 1],
                          [ 0, 1, 2]))
    laplacian_1 = np.array(([ 0,-1, 0],
                            [-1, 4,-1],
                            [ 0,-1, 0]))
    laplacian_2 = np.array(([-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1])) 
    kernel_list = ("sobel_x","sobel_y","sobel_xy","prewitt_x","prewitt_y","prewitt_xy","laplacian_1","laplacian_2")
        
    ap = []
    for w in kernel_list:
        cnn = conv(input_array,eval(w))
        cnn = pooling( cnn, 2, 2)
        cnn = cnn.reshape((1,-1))
        ap = np.append(ap,cnn)
    writer.writerow(ap)#ap必須是[]

def standard(cnn_csv,csv_standard):
    b = np.loadtxt(open(cnn_csv,"r+"),delimiter=",",dtype = 'float64')
    for i in range(len(b[1])):
        y = b[:,i]
        y = preprocessing.scale(y, axis=0, with_mean=True, with_std=True, copy=True)
        b[:,i] = y
    with open(csv_standard, 'w',newline='') as f:  #沒加入newline=''會多空一行
        csv_write = csv.writer(f)
        for line in b:
            csv_write.writerow(line)
def csv_cnn(csv_name):
    a = np.loadtxt(open(csv_name+'.csv',"r+"),delimiter=",",dtype = 'float64')
    b = []   
    for i in range(len(a)):
        b = a[i]
        c0 = []
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        c5 = []
        e = [] 
        for j in range(36):           
            if j % 6 == 0:
                c0.append(b[j])
            if j % 6 == 1:
                c1.append(b[j])
            if j % 6 == 2:
                c2.append(b[j])
            if j % 6 == 3:
                c3.append(b[j])
            if j % 6 == 4:
                c4.append(b[j])
            if j % 6 == 5:
                c5.append(b[j])
        e.append(c0)
        e.append(c1)
        e.append(c2)
        e.append(c3)
        e.append(c4)
        e.append(c5)
        e = np.array(e)  
        mask(e,csv_name+'output.csv')

    standard(csv_name+'output.csv',csv_name+'standard.csv')
        
def conv(image,kernel):
    i_h , i_w = image.shape #取原始影像長寬係數而已 i_h=image長 ,i_w=image寬
    h , w = kernel.shape    #kernel長h、寬w
    #卷積後的影像
    new_h = i_h - h + 1
    new_w = i_w - w + 1
    new_image = np.zeros((new_h,new_w),dtype=np.float)#卷積後影像初始
    # 進行卷積操作，矩陣對應元素值相乘
    for i in range(new_w):
        for j in range(new_h):
            new_image[i,j]=np.sum(image[i:i+h,j:j+w]*kernel)
    return new_image

def pooling(inputmap,poolsize,poolstride):
    #inputmap sizes
    inputmap = np.array(inputmap)
    in_row,in_col = np.shape(inputmap)
    #outputmap sizes
    out_row,out_col = int(np.ceil(in_row/poolstride)),int(np.ceil(in_col/poolstride))
    outputmap = np.zeros((out_row,out_col))
    #取餘數
    mod_row,mod_col = np.mod(in_row,poolstride),np.mod(in_col,poolstride)
    #padding(edge mode是把邊緣數值擴充)
    temp_map = np.lib.pad(inputmap,((0,poolsize-mod_row),(0,poolsize-mod_col)), 'edge')
    #max pooling
    for r in range(0,out_row):
        for c in range(0,out_col):
            x = r*poolstride
            y = c*poolstride
            poolfield = temp_map[x:x+poolsize,y:y+poolsize]
            poolout = np.max(poolfield)
            outputmap[r, c] = poolout
            #outputmap=np.rint(outputmap).astype('uint8')
    return outputmap

def input_target(x,y):
    nn_input = []
    for i in range(len(y)):
        target = [0,0]
        if int(y[i])>=60:#時速順暢
            target = [1,0]
        else:#塞車
            target = [0,1]
        pat = [[x[i],target]]
        nn_input.extend(pat)
    return nn_input
        
def onehot(z):
        if z[0]>=z[1]:
            return [1, 0]
        else:
            return [0, 1]
def tanh (x):
  return math.tanh(x)
  
def dtanh (y):
  return 1 - y**2 #1-Y平方

def relu(x):
    if(x<0):
        y=0
    else :y=x
    return y

def biasMatrix (J):
  m = []
  for i in range(J):
    m.append(round(random.uniform(-1,1),3))
  return m

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = round(random.uniform(a,b),3)

def split(name_csv):
    x=np.loadtxt(open(name_csv+'standard.csv',"r+"),delimiter=",",dtype = 'float64')
    y=np.loadtxt(open(name_csv+'speed.csv',"r+"),delimiter=",",dtype = 'float64')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.02)
    myNN.train (input_target(x_train,y_train),input_target(x,y),y,0,0.001,0,'11_hidden')
    
myNN = NN ( 32, 11, 2)

#CSV用一次就好 產出6X6矩陣，CNN完再正規化，只取STANDARD來用。
  
