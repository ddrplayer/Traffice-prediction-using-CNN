from tkinter import *
from tkinter.ttk import Notebook,Progressbar #用星號會有問題
from PIL import Image,ImageTk
import os
import tarfile
import time, datetime
import requests  #下載網頁資料模組
import csv
import training_testing_23nodes_1090813 as tt
import threading
import numpy as np
#print(tk.TkVersion) #顯示版本
#-----------------函式-----------------
now = 0
def online():
    #if time.minute % 5 > 0:
    #    time = time - datetime.timedelta(minutes=time.minute % 5)
    from_time = datetime.datetime.now().replace(second=0, microsecond=0) # 精確度設為分鐘，將捨棄分鐘以下的數值（歸零）
    down_count = 0
    global now
    for i in range(20):
        when = download_M05A_csv(from_time - datetime.timedelta(minutes=i * 5))
        print('檔案時間＝{}'.format(when))
        if not (when is None) and down_count == 0:
            date_folder = when.strftime('%Y%m%d')#202006251645
            hour_folder = when.strftime('%H')
            #print("good"+date_folder)
            file_name = when.strftime('%Y%m%d%H%M')
            now = when
            #print("god"+file_name)
            down_count = 1        
    print("hour_folder:"+hour_folder)
    mypath = 'D:/M05A/'
    date_List = os.listdir(mypath+date_folder)
    vs36 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    five_ago = [80,50,80,60,80,25]
    ten_ago = [80,50,80,60,80,25]
    fif_ago = [80,50,80,60,80,25]
    twenty_ago = [80,50,80,60,80,25]
    twefive_ago = [80,50,80,60,80,25]
    for hour_List in date_List:
        if int(hour_folder)>=int(hour_List)and int(hour_List)>=(int(hour_folder)-2) :
            print(hour_List)
            min_List = os.listdir(mypath+date_folder+'/'+hour_List+'/')
            for min_File in min_List:
                #count+=1 
                #if count ==18:
                #    break
                #f = open(path+"/"+file); #開啟檔案
                with open(mypath+date_folder+"/"+hour_List+"/"+min_File, newline='') as csvfile:
                        # 讀取 CSV 檔案內容    
                        rows = csv.reader(csvfile)
                        # 以迴圈輸出每一列
                        as0 = 90
                        av0 = 0
                        bs0 = 90
                        bv0 = 0
                        cs0 = 90
                        cv0 = 0
                        for row in rows:  #Direction：車行方向 (S：南；N：北)
                            if row[1]=='05F0309N': #頭城-宜蘭(北)
                                if int(row[4])!=0:
                                    as0 = min(int(row[4]),as0)
                                av0 = av0 + int(row[5])
                            if row[1]=='05F0287N':   #坪林行控專用道-頭城
                                if int(row[4])!=0:
                                    bs0 = min(int(row[4]),bs0)
                                bv0 = bv0 + int(row[5])
                            if row[1]=='05F0055N':  #石碇-坪林行控專用道
                                if int(row[4])!=0:
                                    cs0 = min(int(row[4]),cs0)
                                cv0 = cv0 + int(row[5])
                        current = [as0,av0,bs0,bv0,cs0,cv0]
                        
                        
                        for x in range(6):
                            vs36[x]= current[x]
                            vs36[x+6]=five_ago[x]
                            vs36[x+12]=ten_ago[x]
                            vs36[x+18]=fif_ago[x]
                            vs36[x+24]=twenty_ago[x]
                            vs36[x+30]=twefive_ago[x]
                        with open(file_name+'.csv', 'a', newline='') as csvfile:
                            # 建立 CSV 檔寫入器
                            writer = csv.writer(csvfile)
                            writer.writerow(vs36)
                        twefive_ago=twenty_ago
                        twenty_ago=fif_ago
                        fif_ago =ten_ago
                        ten_ago =five_ago
                        five_ago=current
    tt.csv_cnn(file_name)
    x=np.loadtxt(open(file_name+'standard.csv',"r+"),delimiter=",",dtype = 'float64')
    #st=repr(file_name)               #int型转换为string类型
    #numN2=st[9:13]      # 取4-5位
    #print(numN2)

    now = now + datetime.timedelta(minutes=25)
    #abc = ['a','b','c','d','e','f','g','h','i','j']
    global abc,ab
    abc = [label_a,label_b,label_c,label_d,label_e,label_f,label_g]
    ab = [lab_a,lab_b,lab_c,lab_d,lab_e,lab_f,lab_g]
    for i in range(len(x)):
        if i>=(len(x)-7):
            num = len(x)-i-1
            now = now + datetime.timedelta(minutes=5)            
            time_format = now.strftime('%H'+":"+'%M')
            abc[num].config(text = time_format)
            pred = tt.myNN.runNN(x[i])
            if pred[0]>=pred[1]:
                ab[num].config(text = 'smooth',fg='blue',bg='green',height=5,width=14)
            else:ab[num].config(text = 'jam',fg='red',bg='yellow',height=5,width=14)
def act():
    global now
    global ab
    #print(now)
    when = now - datetime.timedelta(minutes=30) #預測的第一個時間點
    print(when)
    date_folder = when.strftime('%Y%m%d')#202006251645
    my_hour = when.strftime('%H')
    #file_name = now_time.strftime('%Y%m%d%H%M')
    date_List = os.listdir('D:/M05A/'+date_folder)
    right = [right_lab0,right_lab1,right_lab2,right_lab3,right_lab4,right_lab5,right_lab6]
    count = 0
    for hour_List in date_List:
        if int(my_hour)==int(hour_List) :
            min_List = os.listdir('D:/M05A/'+date_folder+'/'+hour_List+'/')
            for min_File in min_List:
                hrmin_num = when.strftime('%H'+":"+'%M')
                st=repr(min_File)               #int型转换为string类型
                numN2=st[20:22]+":"+st[22:24] 
                if hrmin_num==numN2:
                    with open('D:/M05A/'+date_folder+"/"+hour_List+"/"+min_File, newline='') as csvfile:
                        #global bs0
                        bs0 = 90
                        rows = csv.reader(csvfile)
                        for row in rows:  #Direction：車行方向 (S：南；N：北)
                            if row[1]=='05F0287N':   #坪林行控專用道-頭城
                                if int(row[4])!=0:
                                    bs0 = min(int(row[4]),bs0)        
                        
                        #print(numN2+":"+bs0)    
                        #temp_lab = ab[count]
                        #get = temp_lab+".cget('text')
                        if bs0>=60 and ab[count].cget('text') =='smooth':
                            right[count].config(image =right_img,width=100)
                        if bs0<60 and ab[count].cget('text') =='jam':
                            right[count].config(image =right_img,width=100)
                        else:
                            right[count].config(image =wrong_img,width=100)
                        count = count+1  
                        when = when + datetime.timedelta(minutes=5)
                        
                else:
                    continue
        else:
            RuntimeError('驗證資料尚未下載完成')
    """
                if min_File.find(numN2) != -1:
                    with open('D:/M05A/'+date_folder+"/"+hour_folder+"/"+min_File, newline='') as csvfile:
                        for row in rows:  #Direction：車行方向 (S：南；N：北)
                            if row[1]=='05F0287N':   #坪林行控專用道-頭城
                                if int(row[4])!=0:
                                    bs0 = min(int(row[4]),bs0)
                                    print(numN2+":"+bs0)
              
             # if (min_File[15:19])==
    right_lab0.config(image =wrong_img,width=100)
    right_lab1.config(image =right_img,width=100)
    right_lab2.config(image =right_img,width=100)
    right_lab3.config(image =right_img,width=100)
    right_lab4.config(image =right_img,width=100)
    right_lab5.config(image =right_img,width=100)
    right_lab6.config(image =right_img,width=100)
    """    
def download_M05A_csv(time):
    # 將分鐘數字，整理成五分鐘間隔的整數
    if time.minute % 5 > 0:
        time = time - datetime.timedelta(minutes=time.minute % 5)
        
    time = time.replace(second=0, microsecond=0)
    date_folder_url = '{:d}{:0>2d}{:0>2d}/'.format(time.year, time.month, time.day)
    hour_folder_url = '{:0>2d}/'.format(time.hour)
#    0>2d是靠又對齊、留兩個位元
    file_name = 'TDCS_M05A_{:d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}.csv'.format(time.year, time.month, time.day, time.hour, time.minute, time.second)
#    print('file_name = {}'.format(file_name))
    base_folder = 'D:/M05A/'
    file = base_folder + date_folder_url + hour_folder_url + file_name
#    print('File path = {}'.format(os.path.abspath(file)))
    
    if os.path.isfile(file) and os.path.getsize(file) > 0:
        # 因為檔案存在，所以回傳true，也不需要重新下載了
        return time
    
    if not os.path.exists(base_folder + date_folder_url):
        os.mkdir(base_folder + date_folder_url)
        
    if not os.path.exists(base_folder + date_folder_url + hour_folder_url):
        os.mkdir(base_folder + date_folder_url + hour_folder_url)
        
    # download file:  #先看except錯誤有沒有，沒有就執行try.
    __tisv_etc_M05A_base_url__ = 'http://tisvcloud.freeway.gov.tw/history/TDCS/M05A/'
    req = requests.get(__tisv_etc_M05A_base_url__ + date_folder_url + hour_folder_url + file_name, timeout=20)
    try:
        if not req.status_code == requests.codes.ok:
            print('下載檔案[{}]失敗：{}'.format(req.url, req.status_code))
            return
        
        with open(file, 'wb') as download_csv_file:
            download_csv_file.write(req.content)
    
    except(OSError, Exception) as e:
        raise Exception('下載檔案失敗' + str(e))
    except(OSError, Exception) as e: #操作系统错误 or 常规错误的基类
        raise Exception('下載檔案失敗' + str(e))
        
    # 最後檢查一下檔案大小，若是檔案只有346k，就不是ok的資料，把它刪掉！
    if os.path.isfile(file) and os.path.getsize(file) < 1000:
        os.remove(file)
        
    return time

def download():
   # 以下下載csv檔案
    from_time = datetime.datetime.now().replace(second=0, microsecond=0) # 精確度設為分鐘，將捨棄分鐘以下的數值（歸零）
    text.insert(END,"\n" + "開始下載"+ number_entry.get()+"筆資料")
    down_count = 0
    for i in range(0, int(number_entry.get())):
        when = download_M05A_csv(from_time - datetime.timedelta(minutes=i * 5))
        #text.insert(END,"\n" + '檔案時間＝{}'.format(when))
        print('檔案時間＝{}'.format(when))
        down_count += 1
        pb['value'] = down_count 
    text.insert(END,"\n" + '完成下載{}筆檔案'.format(down_count))
def threading_down():
    td = threading.Thread(target = download)
    pb['maximum'] = number_entry.get()
    #pb['value'] = 0
    #td.setDaemon(True)
    td.start()
    #pb.start()
    #td.join()
    #pb.stop()
    #text.insert(END,"\n" + '完成下載{}筆檔案'.format(down_count))
def test():
    maxnum = 60
    pb2['maximum'] = maxnum
    for i in range(maxnum):
        pb2['value'] = i
        i = i+1
        time.sleep(1)
    #text3.insert(END,"\n" + 123_img)  
    text3.insert(END,"\n" + "CNN & Standard are finish. ") 
def ANN():
    tt.split(number_entry31.get()) 
    text3.insert(END,"\n" + "CNN & Standard are finish. ")    

def threading_ANN():
    #tc = threading.Thread(target = lambda: tt.csv_cnn(number_entry3.get()))
    text3.insert(END,"\n" + "Start training ")    
    TA = threading.Thread(target = ANN)
    #pb1['maximum'] = 9000
    TA.start()

def CNN_standerd():
    tt.csv_cnn(number_entry3.get())    

def threading_CNN():
    #tc = threading.Thread(target = lambda: tt.csv_cnn(number_entry3.get()))
    tc = threading.Thread(target = CNN_standerd)
    #pb1['maximum'] = 9000
    tc.start()
    text3.insert(END,"\n" + "Start (CNN & Standard) ")    
    for i in range(60):
        i = i+1
        time.sleep(1)
    text3.insert(END,"\n" + "CNN & Standard are finish. ")    
    
def threading_data():
    td = threading.Thread(target = data_arrange)
    pb1['maximum'] = 30
    #pb['value'] = 0
    #td.setDaemon(True)
    td.start()
    #pb.start()
    #td.join()
    #pb.stop()
    #text.insert(END,"\n" + '完成下載{}筆檔案'.format(down_count))
def data_arrange():
    mypath = 'D:/M05A/'
    date_List = os.listdir(mypath)
    # 開啟 CSV 檔案
    count=0
    vs36 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    five_ago = [80,50,80,60,80,25]
    ten_ago = [80,50,80,60,80,25]
    fif_ago = [80,50,80,60,80,25]
    twenty_ago = [80,50,80,60,80,25]
    twefive_ago = [80,50,80,60,80,25]
    text.insert(END,"\n" + "開始處理資料")
    for date_File in date_List:
        if int(date_File)>=int(date0_entry.get()) and int(date_File)<=int(date1_entry.get()):
            hour_list = os.listdir(mypath+date_File)
            count=count+1
            pb1['value'] = count
            print("processing "+str(count))
            for hour_File in hour_list:
                hour_csv = os.listdir(mypath+date_File+"/"+hour_File)
                for min_csv in hour_csv:
                    with open(mypath+date_File+"/"+hour_File+"/"+min_csv, newline='') as csvfile:
                        # 讀取 CSV 檔案內容    
                        rows = csv.reader(csvfile)
                        # 以迴圈輸出每一列
                        as0 = 90
                        av0 = 0
                        bs0 = 90
                        bv0 = 0
                        cs0 = 90
                        cv0 = 0
                        for row in rows:  #Direction：車行方向 (S：南；N：北)
                            if row[1]=='05F0309N': #頭城-宜蘭(北)
                                if int(row[4])!=0:
                                    as0 = min(int(row[4]),as0)
                                av0 = av0 + int(row[5])
                            if row[1]=='05F0287N':   #坪林行控專用道-頭城
                                if int(row[4])!=0:
                                    bs0 = min(int(row[4]),bs0)
                                bv0 = bv0 + int(row[5])
                            if row[1]=='05F0055N':  #石碇-坪林行控專用道
                                if int(row[4])!=0:
                                    cs0 = min(int(row[4]),cs0)
                                cv0 = cv0 + int(row[5])
                        current = [as0,av0,bs0,bv0,cs0,cv0]
                        
                        
                        for x in range(6):
                            vs36[x]= current[x]
                            vs36[x+6]=five_ago[x]
                            vs36[x+12]=ten_ago[x]
                            vs36[x+18]=fif_ago[x]
                            vs36[x+24]=twenty_ago[x]
                            vs36[x+30]=twefive_ago[x]
                        with open('{:s}{:0>8s}.csv'.format(date0_entry.get(),date1_entry.get()), 'a', newline='') as csvfile:
                            # 建立 CSV 檔寫入器
                            writer = csv.writer(csvfile)
                            writer.writerow(vs36)
        
                        twefive_ago=twenty_ago
                        twenty_ago=fif_ago
                        fif_ago =ten_ago
                        ten_ago =five_ago
                        five_ago=current
    text.insert(END,"\n" + "已完成處理"+str(count) +"天資料")
#def make_speed():
#    with open('iris.csv', newline='') as csvfile:


def speed_csv():
 #   if not os.path.exists(speed_entry.get() + "speed.csv"):
  #      os.mkdir(speed_entry.get() + "speed.csv")
  path = speed_entry.get()+'speed.csv'
  if os.path.exists(speed_entry.get() + "speed.csv"):
      text3.insert(END,"\n" +speed_entry.get()+'.csv has existed!')
  else:
      with open(path, 'a',newline='') as csvwrite:#沒加入newline=''會多空一行
          with open(speed_entry.get()+'.csv', newline='') as csvfile:
            writer = csv.writer(csvwrite)
            for i in range(12):
                writer.writerow([86])
            rows = csv.reader(csvfile)
            for row in rows:
                writer.writerow([row[2]])
      with open(speed_entry.get()+'.csv','a', newline='') as csvadd:
            add = csv.writer(csvadd)
            for i in range(12):
                add.writerow([20,86,26,86,50,86,20,86,26,86,50,86,20,86,26,86,50,86,20,86,26,86,50,86,20,86,26,86,50,86,20,86,26,86,50,86])
            text3.insert(END,"\n" +speed_entry.get()+'.csv was established!')
      
#-----------主視窗------------------  
root = Tk()
root.title("Tunnel traffic prediction system")
root.geometry("800x526+520+32") #大小寬乘高,距離螢幕左上角 720,30
#root.minsize(width=500,height=300) #最小視窗
#root.maxsize(width=760,height=460) #最小視窗
#root.resizable(False,False)#不能放大縮小
root.configure(bg="#f0f0f0")  #視窗背景顏色
#root.config(menu=menubar)

#-----------主視窗------------------  
notebook =  Notebook(root)
frame0 = Frame()
frame1 = Frame()
frame2 = Frame()
frame3 = Frame()

#---------frame1----------
notebook.add(frame1,text="CNN Structure")
labcome = Label(frame1,text = '歡迎使用雪山隧道路況預測系統!',fg="blue")
labcome.config(font="微軟正黑體 20")
labcome.pack(pady=15)

structure_img = PhotoImage(file = "Structure1.png")
lab1=Label(frame1,image = structure_img)
lab1.pack(pady=20)

#---------frame2----------
notebook.add(frame2,text="Preprocessing")
pw = PanedWindow(frame2)
labframe = LabelFrame(pw,text = 'Data Downlaod',width=140,height = 150)
lab2 = Label(labframe,text = '輸入要下載的筆數')
lab2.config(font="微軟正黑體 11",padx=16,pady=2)
#lab2.pack(ipady=10)
lab2.grid(row=0,column=0)
number_entry = Entry(labframe,font="微軟正黑體 12")   #get()
number_entry.grid(row=0,column=1)
btn = Button(labframe,text="Data Downlaod",command=threading_down)
btn.grid(row =1)
pw.add(labframe)
labframe.grid(row=0,column=0)

pbframe =LabelFrame(pw,text = '下載進度',width=340,height =150)
pb = Progressbar(pbframe,length=330,mode = 'determinate',orient=HORIZONTAL)
pb.pack(padx=5,pady=5)
amount = 0
pb['value']=0
pw.add(pbframe)
pbframe.grid(row=1,column=0)

labframe1 = LabelFrame(pw,text = 'Data Arrange',width=100,height = 150)
lab3 = Label(labframe1,text = '輸入要處裡的日期，西元年月日八碼',font="微軟正黑體 11")
#lab2.pack(ipady=10)
lab3.grid(row=0,column=0)
date0_entry = Entry(labframe1,font="微軟正黑體 11",width=10)
date0_entry.grid(row=1,column=0)
date1_entry = Entry(labframe1,font="微軟正黑體 11",width=10)
date1_entry.grid(row=1,column=1)
btn1 = Button(labframe1,text="Data Arrange",command=threading_data)
btn1.config(width=12)
#btn1.pack()
btn1.grid(row =2)
pb1frame =LabelFrame(pw,text = '處理進度',width=400,height =150)
pb1 = Progressbar(pb1frame,length=330,mode = 'determinate',orient=HORIZONTAL)
pb1.pack(padx=5,pady=5)
#amount = 0
pb1['value']=0

pw.add(labframe1)
labframe1.grid(row=2,column=0)
pw.add(pb1frame)
pb1frame.grid(row=3,column=0)
#pw.pack(fill=X,ipadx=10,ipady=10)
#pw.add(onemframe)
#mapframe.pack(padx=10,pady=5)
#sep = Separator(frame2)
#sep.pack(fill=Y)
pwin = PanedWindow(pw)
pw.add(pwin)
pwin.grid(row=4,column=0)
informationframe = LabelFrame(pwin,text="下載成功檔案明細",width=450,height=130)

pwin.add(informationframe)
informationframe.grid(row=2,column=0)

localtime = time.asctime( time.localtime(time.time()) )
yscrollbar = Scrollbar(informationframe)
text =  Text(informationframe,width=61,height=9)
yscrollbar.pack(side=RIGHT,fill=Y)
text.pack()
yscrollbar.config(command=text.yview)
text.config(yscrollcommand=yscrollbar.set)
text.insert(END,"登入系統時間:  "+localtime)

informationframe.grid(row=7,column=1,padx=15)


pw.pack(fill=Y,padx=10,pady=10)
#---------frame3----------
notebook.add(frame3,text="Training and Testing")
pw3 = PanedWindow(frame3)
labframe30 = LabelFrame(pw3,text = '生成速度csv檔',width=120,height = 150)
lab30 = Label(labframe30,text = '輸入檔名')
lab30.config(font="微軟正黑體 11")
lab30.grid(row=0,column=0)
speed_entry = Entry(labframe30,font="微軟正黑體 12")   #get()
speed_entry.grid(row=0,column=1)
btn30 = Button(labframe30,text="生成速度csv檔",command=speed_csv)
btn30.grid(row =1)
pw3.add(labframe30)
labframe30.grid(row=0,column=0)
labframe3 = LabelFrame(pw3,text = 'CNN & Standard',width=120,height = 150)
lab3 = Label(labframe3,text = '輸入要訓練的檔名')
lab3.config(font="微軟正黑體 11")
#lab2.pack(ipady=10)
lab3.grid(row=0,column=0)
number_entry3 = Entry(labframe3,font="微軟正黑體 12")   #get()
number_entry3.grid(row=0,column=1)
#para = number_entry3.get()
btn3 = Button(labframe3,text="Start",command=threading_CNN)
btn3.grid(row =1)
pw3.add(labframe3)
labframe3.grid(row=1,column=0)

labframe31 = LabelFrame(pw3,text = 'Training and Testing',width=120,height = 150)
lab31 = Label(labframe31,text = '輸入要訓練的檔名')
lab31.config(font="微軟正黑體 11")
#lab2.pack(ipady=10)
lab31.grid(row=0,column=0)
number_entry31 = Entry(labframe31,font="微軟正黑體 12")   #get()
number_entry31.grid(row=0,column=1)
#para = number_entry3.get()
btn31 = Button(labframe31,text="start",command=threading_ANN)
btn31.grid(row =1)
pw3.add(labframe31)
labframe31.grid(row=2,column=0)

#pb2frame =LabelFrame(pw3,text = 'ANN進度',width=340,height =150)
#pb2 = Progressbar(pb2frame,length=300,mode = 'determinate',orient=HORIZONTAL)
#pb2.pack(padx=5,pady=5)
#amount = 0
#pb2['value']=0
#pw3.add(pb2frame)
#pb2frame.grid(row=3,column=0)


informationframe3 = LabelFrame(pw3,text="console",width=450,height=130)
pw3.add(informationframe3)
informationframe3.grid(row=3,column=0)
#localtime = time.asctime( time.localtime(time.time()) )
yscrollbar3 = Scrollbar(informationframe3)
text3 =  Text(informationframe3,width=61,height=9)
yscrollbar3.pack(side=RIGHT,fill=Y)
text3.pack()
yscrollbar3.config(command=text.yview)
text3.config(yscrollcommand=yscrollbar.set)
text3.insert(END,"登入系統時間:  "+localtime)
informationframe3.grid(row=7,column=0,padx=15)
pw3.pack(fill=Y,padx=10,pady=10)

#---------frame0---------- 152command
notebook.add(frame0,text="Prediction System")
pw0 = PanedWindow(frame0)

#labframe0 = LabelFrame(pw0,text = '即時預測',width=120,height = 150)

btn0 = Button(pw0,text="開始即時預測",command=online)
pw0.add(btn0)
btn0.grid(row =0,column=0)
#labframe0.grid(row=0,column=0)

labframe01 = LabelFrame(pw0,text = 'prediction',fg = 'blue')

lab_g = Label(labframe01)
lab_g.grid(row=2,column=3)
label_g = Label(labframe01,text = '',fg='black',height=1,width=14)
label_g.grid(row=1,column=3)
lab_f = Label(labframe01)
lab_f.grid(row=2,column=4)
label_f = Label(labframe01,text = '',fg='black',height=1,width=14)
label_f.grid(row=1,column=4)
lab_e = Label(labframe01)
lab_e.grid(row=2,column=5)
label_e = Label(labframe01,text = '',fg='black',height=1,width=14)
label_e.grid(row=1,column=5)
lab_d = Label(labframe01)
lab_d.grid(row=2,column=6)
label_d = Label(labframe01,text = '',fg='black',height=1,width=14)
label_d.grid(row=1,column=6)
lab_c = Label(labframe01)
lab_c.grid(row=2,column=7)
label_c = Label(labframe01,text = '',fg='black',height=1,width=14)
label_c.grid(row=1,column=7)
lab_b = Label(labframe01)
lab_b.grid(row=2,column=8)
label_b = Label(labframe01,text = '',fg='black',height=1,width=14)
label_b.grid(row=1,column=8)
lab_a = Label(labframe01)
lab_a.grid(row=2,column=9)
label_a = Label(labframe01,text = '',fg='black',height=1,width=14)
label_a.grid(row=1,column=9)

pw0.add(labframe01)
labframe01.grid(row=4,column=0)
btn01 = Button(pw0,text="實際路況更新",command=act)
pw0.add(btn01)
btn01.grid(row =1,column=0)
labframe02 = LabelFrame(pw0,text = 'actually',fg = 'blue')
right_img = PhotoImage(file = "right.png")
wrong_img = PhotoImage(file = "wrong.png")
x_img = PhotoImage(file = "x.png")
right_lab0=Label(labframe02,image=x_img,width=100)
right_lab0.grid(row=10,column=0)
right_lab1=Label(labframe02,image=x_img,width=100)
right_lab1.grid(row=10,column=1)
right_lab2=Label(labframe02,image=x_img,width=100)
right_lab2.grid(row=10,column=2)
right_lab3=Label(labframe02,image=x_img,width=100)
right_lab3.grid(row=10,column=3)
right_lab4=Label(labframe02,image=x_img,width=100)
right_lab4.grid(row=10,column=4)
right_lab5=Label(labframe02,image=x_img,width=100)
right_lab5.grid(row=10,column=5)
right_lab6=Label(labframe02,image=x_img,width=100)
right_lab6.grid(row=10,column=6)
labframe02.grid(row=5,column=0)
pw0.pack(fill=Y,padx=10,pady=10)

#-------------------

notebook.pack(fill=BOTH,expand=TRUE)
LOGO = PhotoImage(file = "logo1.png")
lab=Label(root,image = LOGO)
lab.pack(anchor=S,side=RIGHT)
root.mainloop() #視窗要一直存在