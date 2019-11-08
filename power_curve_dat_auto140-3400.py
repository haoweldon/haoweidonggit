# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:15:33 2018

@author: 31376
@author:33202
"""

#coding=utf-8
import os
import glob
import pandas as pd
#import sys
import struct
import numpy as np
import math
import matplotlib.pyplot as plt
import xlrd
import boto
import boto3
import os
from boto.s3.connection import S3Connection,OrdinaryCallingFormat
import time
import urllib
import zipfile
from pyecharts import Line
from pyecharts import Page
from pyecharts import Scatter
from pyecharts import Overlap
import datetime
import time

def get_object_AWS(bucket_name, s3_path, local_file):
    u'''
    S3服务
    bucket_name :要下载的S3桶名
    s3_path :要下载至S3中的文件夹或文件名称
    local_file :要下载的本地文件夹或文件名称
    eg:
        下载文件
        get_object_AWS(bucket_name = u'test',s3_path = 'test1/1.txt',local_file = 'test/1.txt')
        下载文件夹
        get_object_AWS(bucket_name = u'test',s3_path = 'test1',local_file = 'test')
    '''
    try:
        conn = boto.s3.connect_to_region('cn-north-1',
           aws_access_key_id="AKIAO4LHUO6MQCXAI7FA",
           aws_secret_access_key="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",
           is_secure=True,
           calling_format = boto.s3.connection.OrdinaryCallingFormat()
            )
        b = conn.get_bucket(bucket_name)
        if os.path.splitext(s3_path)[1]!="" and os.path.splitext(local_file)[1]!="":
            key = b.lookup(s3_path)
            a,b = os.path.split(local_file)
            if not os.path.exists(a):
                os.makedirs(a)
            key.get_contents_to_filename(local_file)
            print local_file
        elif os.path.splitext(s3_path)[1]=="" and os.path.splitext(local_file)[1]=="":
            if s3_path[-1]!="/":
                s3_path = s3_path+"/"
            df = list_bucket(bucket_name,s3_path)
            if len(df)!=0 and not os.path.exists(local_file):
                os.makedirs(local_file)
            for filename in zip(*df)[0]:
                tmp  =  os.path.basename(filename)
                get_object_AWS(bucket_name, filename,os.path.join(local_file,tmp))
        else:
            print "Error"
    except Exception,e:
        print "Error",e
        
                
def list_bucket1(bucket,prefix):
    u'''
    S3服务
    bucket :要列表的S3桶名
    prefix :关键字
    返回 ：含有关键字的所有数据的S3key和文件大小组成的二维数组，第一列是Key，第二列是文件大小
    eg:
        list_bucket1(bucket = 'test',prefix = 'uncal/')
    '''
    try:
        c = boto.s3.connect_to_region('cn-north-1',
                    aws_access_key_id="AKIAO4LHUO6MQCXAI7FA",
                    aws_secret_access_key="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",
                    is_secure=True,
                    calling_format = boto.s3.connection.OrdinaryCallingFormat()                                
                     )
        b = c.get_bucket(bucket)
        x= b.list(prefix=prefix)
        df = [[i.key,float(i.size)/pow(1024,2)] for i in x if i.key.split("/")[-1]!='']
        return df
    except Exception,e:
        print "Error",e
        return []

def read_famos_dat(filename,sep = ""):
    indexs = []
    ins= {"CP":[4,5],"Cb":[7,8],"CN":[7],"CR":[4,5]}
    f =  open(filename, 'rb')
    tmp =  f.read()
    f.close()
    n=0
    i=0
    index= tmp.find("|CS")
    while 1:
        if tmp[index+i]==',':
            n+=1
        i+=1
        if n==4:
            break   

    data =  tmp[i+index:-1]   

    for row in tmp.split("\n"):
        if row[0:3]== '|CS':
            break
        else:
            tmp = row.split(",")
            header = tmp[0][1:]
            if header in ins:
                indexs.extend([tmp[i] for i in ins[header]])
    columns= []
    datas = []
    N=0
    ss = 7

    for i in range(len(indexs)/ss):
        tmp = indexs[ss*i:ss*(i+1)]
        columns.append(tmp[-1])
        tmp1 = data[int(tmp[2]):int(tmp[2])+int(tmp[3])]
        lengh = int(tmp[0])
        ff = tmp[1]
        if int(float(tmp[4]))==0:
            cr =1.0
            offset = 0.0
        else:
            cr = float(tmp[4])
            offset = float(tmp[5])
        
        if lengh == 8:
            format = "d"
        elif lengh == 4:
            if ff == "7":
                format = "f"
            if ff =="6":
                format = "l"
            if ff == "5":
                format = "L"
        elif lengh == 2:
            if ff == "3":
                format = "H"
            if ff == "4":
                format = "h"
                
        elif lengh == 1:
            if ff == "1":
                format = "b"
            if ff == "2":
                format = "B"
                
        tdata = struct.unpack("%d%s"%((len(tmp1)/lengh),format),tmp1)
        tdata = [tt*cr+offset for tt in tdata]
        datas.append(tdata)
        N+=len(tdata)
    return datas,columns

	
#以上都是从S3自动下载数据、解析程序，不需要关注
def temcaldat(datax,number): #计算十分钟数据
    group=(len(datax)-len(datax)%number)/number
    datax.insert(0,'label_rep',group+1)
    label=range(1,group+1)
    label_rep=[]
    for i in label:
        for j in range(number):
            label_rep.append(i)
    datax.loc[0:(len(datax)-len(datax)%number-1),'label_rep']=label_rep
    
    data_mean=datax.groupby('label_rep').mean()
    
    wind_std=datax[['windspeed1','label_rep']].groupby('label_rep').std()
    data_mean.insert(0,'tur',wind_std.windspeed1/data_mean.windspeed1)
    
    winddrc_std=datax[['winddirection1','label_rep']].groupby('label_rep').std()
    data_mean.insert(0,'wdrc_std',winddrc_std.winddirection1)
    
    data_min=datax.groupby('label_rep').min()
    data_min.columns=list(data_min.columns+'_min')
    data_max=datax.groupby('label_rep').max()
    data_max.columns=list(data_max.columns+'_max')
    
    data_mean.insert(0,'label_rep',range(1,group+1))
    data_min.insert(0,'label_rep',range(1,group+1))
    data_max.insert(0,'label_rep',range(1,group+1))
    y=pd.merge(data_min,data_mean, on='label_rep')
    y=pd.merge(y,data_max, on='label_rep')
    return y
def wind_standar(pressure,tem,wind):#风速标准化
    rho=pressure*100/(R*(tem+T0))
    wind_s=wind*((rho/report_rho)**(1/3.0))
    return wind_s
def cp(wind,power):#计算Cp
    cp=(power*1000)/(0.5*area*report_rho*(wind**3))
    return cp
def floatrange(start,stop,step):
    x=int((stop-start)/step)+1
    return([start+float(i)*step for i in range(x)]) 

def binpower(wind,power):#拟合功率曲线
    w_max=math.ceil(max(wind))
    if w_max-max(wind)>0.5:
         w_max=w_max-0.5
    w_min=1
    w_range=floatrange(w_min,w_max,0.5)
    w_l=len(w_range)
    ws=[]
    pwr=[]
    num=[]
    for i in range(w_l):
        xbin=wind[(wind>i*0.5+w_min-0.25)&(wind<i*0.5+w_min+0.25)]
        if len(xbin)>=1:
            ws.append(round(np.mean(wind[(wind>i*0.5+w_min-0.25)&(wind<i*0.5+w_min+0.25)]),2))
            pwr.append(round(np.mean(power[(wind>i*0.5+w_min-0.25)&(wind<i*0.5+w_min+0.25)]),2))
            num.append(len(xbin))
    wp=pd.DataFrame({'wind':ws,'power':pwr,'num':num})
    return(wp)

def continuousnum(data):
    x=data.diff()
    x=x.dropna(how='any')
    if len(np.unique(x))!=1:
        y=x[x!=0].index.tolist()
        y=pd.DataFrame({'y':y})
        yy=y.diff()
        return(max(yy.dropna(how='any').y))
    else:
        return(30000)
def GetNowTime():
    return time.strftime("%Y-%m-%d",time.localtime(time.time()))

def Ratecal(x,y):
    w_max=math.ceil(max([max(x.wind),max(y.wind)]))
    if w_max-max([max(x.wind),max(y.wind)])>0.5:
        w_max=w_max-0.5
    w_min=1
    w_range=floatrange(w_min,w_max,0.5)
    w_l=len(w_range)
    wind=[]
    rate=[]
    for i in range(w_l):
        xbin=x.wind[(x.wind>i*0.5+w_min-0.25)&(x.wind<i*0.5+w_min+0.25)].index.tolist()
        ybin=y.wind[(y.wind>i*0.5+w_min-0.25)&(y.wind<i*0.5+w_min+0.25)].index.tolist()
        if len(xbin)>=1&len(ybin)>=1:
            wind.append(round(x.wind[xbin],2))
            rate.append(round(100*float(x.power[xbin])/float(y.power[ybin]),2))
    power_rate=pd.DataFrame({'wind':wind,'rate':rate})
    return(power_rate)
def bindata(x,y,span):
    w_max=math.ceil(max(x))
    if w_max-max(x)>span:
        w_max=w_max-span
    w_min=math.floor(min(x))
    if min(x)-w_min>span:
        w_min=w_min+span    
    w_range=floatrange(w_min,w_max,span)
    w_l=len(w_range)
    ws=[]
    pwr=[]
    for i in range(w_l):
        xbin=x[(x>i*span+w_min-span/2)&(x<i*span+w_min+span/2)]
        if len(xbin)>=1:
            ws.append(round(np.mean(x[(x>i*span+w_min-span/2)&(x<i*span+w_min+span/2)]),2))
            pwr.append(round(np.mean(y[(x>i*span+w_min-span/2)&(x<i*span+w_min+span/2)]),2))
    wp=pd.DataFrame({'x':ws,'y':pwr})
    return(wp)
def getYesterday(): 
    today=datetime.date.today() 
    oneday=datetime.timedelta(days=1) 
    yesterday=today-oneday  
    return str(yesterday)
def dataload(bucket_name,bucket_name1,local_regeps,local_id): #自动增量下载数据
     #S3 bucket list
     bucketlist=list_bucket1(bucket_name,bucket_name1)
     bucketlist=pd.DataFrame(bucketlist)
     
     bucketlist=bucketlist.sort_values(by=0,ascending=True)
     bucketlist.columns=[u'file',u'size']
     bfn=[os.path.basename(bucketlist.file[i]) for i in range(len(bucketlist.file))]
     bfn=[bfn[i].split('.')[0] for i in range(len(bfn))]
     #local list
     localfile=glob.glob(local_regeps)
     localfile=sorted(localfile)
     lfn=[os.path.basename(localfile[i]) for i in range(len(localfile))]
     lfn=[lfn[i].split('.')[0] for i in range(len(lfn))]
     #remainder
     bfn.sort()
     ret_num=bfn.index(max(lfn))+1
     lbfn=len(bfn)
     l_num=range(ret_num,lbfn)
     if len(l_num):
          for i in l_num:
               local_file=local_id+'\\'+bfn[i]+'.'+os.path.basename(bucketlist.file[1]).split('.')[1]
               get_object_AWS(bucket_name,bucketlist.file[i],local_file)
 #主函数        
dataload(u'goldwind-wttest-typecertification','TypeTest/ForInnerShare/GW140P3400BSinoma686T100_Dabancheng/Calibrated/',\
         r'Z:\140-3400\2*.dat',r'Z:\140-3400')


design_power = xlrd.open_workbook(r'C:\Python27\design\140-3400.xlsx')
files=glob.glob(r'Z:\140-3400\2*.dat')
files=[os.path.basename(files[i]).split('_cal')[0] for i in range(len(files))]
files.sort()
#增量计算
try:
    existfile=pd.read_csv('Z:\\140-3400result\\'+'df10min'+getYesterday()+'.csv')
    files=files[files.index(max(existfile.file)):len(files)]
except Exception,e:
       print("Error",e)
       
dat_cmp=[os.path.basename(files[i]).split('_cal')[0] for i in range(len(files))]
dat_cmp.sort()
#dat_cmp=dat_cmp[1400:1500]
savepath='Z:\\140-3400result\\'
files=sorted(files)
f_flage=[]
for f in dat_cmp:
     filename='Z:\\140-3400\\'+f+'_cal.dat'
     s=os.path.basename(filename)
     savname=os.path.dirname(filename)+'\\'+s.split('.')[0]+'.csv'
     try:
          Datas,Columns=read_famos_dat(filename)
          df=pd.DataFrame(Datas).T
          Columns=[i.replace('_copy','') for i in Columns]
          df.columns=Columns
          
          df=df.loc[:,['P_net','P_con','Avail','windspeed1','windspeed2','windspeed3',u'WT_NacellePosition',\
                       'winddirection1','temperature','counter1','pitch1',u'pitch2', u'pitch3',\
                       u'gen_rpm',u'acc_x',u'acc_y','WT_Yaw_err','humidity','status_yaw','v_nac','az_nac']]
          
#          df[u'WT_NacellePosition']=df[u'WT_NacellePosition'].diff()
#          df[u'WT_NacellePosition'][0]=0
#          df.loc[(df[u'WT_NacellePosition']>=1),[u'WT_NacellePosition']]=1
#          df.loc[(df[u'WT_NacellePosition']<1),[u'WT_NacellePosition']]=0
          
          
          df[u'gen_rpm']=6+30*(df[u'gen_rpm']-6)/23
          print(f)
          if max(df.pitch1)>49:
               df.insert(0,'label_1',1)
          else:
               df.insert(0,'label_1',0)
          timespan=30000
          datay=temcaldat(df,timespan)
          datay['status_yaw']=sum(df[u'status_yaw'])
          file_ts=f
          datay.insert(0,'file',file_ts)
          del df['label_rep']
          if len(f_flage)==0:
               df10=datay
               f_flage=f
          else:
               df10=pd.concat([df10,datay])
     except Exception,e:
       print("Error",e)
#上面都是转十分钟的程序操作
try:
     df10=df10.reset_index(drop=True)
     data10=df10
     df10.dropna(how='any')
     df10=df10.reset_index(drop=True)
     #data clean
     del_row=df10[(df10.Avail!=1.0)].index.tolist()
     
     del_rowi=df10[(df10.winddirection1>126)&(df10.winddirection1<275)].index.tolist()
     del_row=del_row+del_rowi
     
     del_rowi=df10[(df10.P_net<=0)].index.tolist()
     del_row=del_row+del_rowi
     
     del_rowi=df10[(df10.temperature<2)&(df10.humidity>80)].index.tolist()
     del_row=del_row+del_rowi

     del_row=sorted(del_row)
     del_row=np.unique(del_row)
     if len(del_row)>0:
         df10=df10.drop(del_row)  
     df10=df10.reset_index(drop=True)
     s_rowi=df10[(df10.label_1!=0)].index.tolist()
     if len(s_rowi)>0:
         df10s=df10.iloc[s_rowi,]
  #上面都是剔除数据操作   
     #standar
     report_rho=1.225
     R=287.05
     T0=273.15
     h_hub_lidar=90
     
     df10['windspeedr']=df10.windspeed1
     
     df10.counter1=[df10.counter1[i]*math.exp((-9.80665*h_hub_lidar)/(R*(df10.temperature[i]+T0))) for i in range(len(df10.counter1))]
     df10.windspeed1=wind_standar(df10.counter1,df10.temperature,df10.windspeed1)
     df10.v_nac=wind_standar(df10.counter1,df10.temperature,df10.v_nac)
     #上面是风速标准化
	 
	 
	 #下面主要是第二页多维度分析用的变量加工
     #Cp
     area=15431.21
     df10.insert(0,'Cp',0)
     df10.Cp=cp(df10.windspeed1,df10.P_net)
     #lamada
     df10.insert(0,'lanmada',0)
     df10.lanmada=(2*math.pi*0.5*140*df10.gen_rpm/df10.windspeedr)/60
     #torque
     df10.insert(0,'torque',0)
     df10.torque=1000*df10.P_net/(df10.gen_rpm/9.55)
     #Kopt
     df10.insert(0,'Kopt',0)
     df10.Kopt=df10.torque/(np.square(df10.gen_rpm/9.55))
except Exception,e:
       print("Error",e)
if 'df10' in locals() or 'df10' in globals():
    if 'existfile' in locals() or 'existfile' in globals():
        df10=pd.concat([existfile,df10])
else:
    df10=existfile
#save
savname=savepath+'df10min'+GetNowTime()+'.csv'
df10.to_csv(savname,index=False)

wind_power=binpower(df10.windspeed1,df10.P_net)#拟合功率曲线
#wind_power=wind_power.rename(columns={'x':'wind','y':'power'});
wind_power.insert(0,'Cp',0)
wind_power.Cp=cp(wind_power.wind,wind_power.power)

wind_power.insert(0,'speed',0)
wind_speed=bindata(df10.windspeed1,df10.gen_rpm,0.5)
wind_power.speed=wind_speed.y
wind_power.insert(0,'lanmada',0)
wind_power.lanmada=(2*math.pi*0.5*140*wind_power.speed/wind_power.wind)/60

wind_pitch=bindata(df10.windspeed1,df10.pitch1,0.5)
wind_power.insert(0,'pitch',0)
wind_power.pitch=wind_pitch.y

table = design_power.sheet_by_name(u'Sheet1')
nrows=table.nrows
design=pd.DataFrame({'wind':table.col_values(0)[1:nrows],'power':table.col_values(1)[1:nrows]})
#plot
savname=savepath+'\\'+'powerline'+GetNowTime()+'.png'
plt.figure(1)
plt.subplot(111)
plt.plot(design.wind,design.power,'g-s',label='design_power')
plt.plot(wind_power.wind,wind_power.power,'b-s',label='power_curve')
plt.title('wind power')
plt.xlabel('wind(m/s)')
plt.ylabel('powe(kW)')
plt.grid(True)
plt.legend(loc='upper left',borderaxespad=0.,fontsize='x-small')
plt.savefig(savname,dpi=600)
#plt.show()
plt.pause(15)
plt.close()

savname=savepath+'Number_points'+GetNowTime()+'.png'
plt.figure(1)
plt.subplot(111)
plt.plot(wind_power.wind,wind_power.num,'go',label='design_power')
plt.title('wind bin points')
plt.xlabel('wind(m/s)')
plt.ylabel('points')
plt.grid(True)
plt.legend(loc='upper left',borderaxespad=0.,fontsize='x-small')
plt.savefig(savname,dpi=600)
#plt.show()
plt.pause(15)
plt.close()

if len(s_rowi)>0:
    savname=savepath+'\\'+'powerpot'+GetNowTime()+'.png'
    plt.figure(1)
    plt.subplot(111)
    plt.plot(df10.windspeed1,df10.P_net,'b+',label='power_curve_mean')
    plt.plot(df10.windspeed1,df10.P_net_max,'g+',label='power_curve_max')
    plt.plot(df10.windspeed1,df10.P_net_min,'r+',label='power_curve_min')
    plt.plot(df10s.windspeed1,df10s.P_net,'yo',label='power_curve_means')
    plt.plot(df10s.windspeed1,df10s.P_net_max,'y^',label='power_curve_maxs')
    plt.plot(df10s.windspeed1,df10s.P_net_min,'yv',label='power_curve_mins')
    plt.plot(design.wind,design.power,'k-',label='design_power')
    plt.title('wind power')
    plt.xlabel('wind(m/s)')
    plt.ylabel('powe(kW)')
    plt.grid(True)
    plt.legend(loc='upper left',borderaxespad=0.,fontsize='xx-small')
    plt.savefig(savname,dpi=600)
    #plt.show()
    plt.pause(15)
    plt.close()
else:
    savname=savepath+'\\'+'powerpot'+GetNowTime()+'.png'
    plt.figure(1)
    plt.subplot(111)
    plt.plot(df10.windspeed1,df10.P_net,'b+',label='power_curve_mean')
    plt.plot(df10.windspeed1,df10.P_net_max,'g+',label='power_curve_max')
    plt.plot(df10.windspeed1,df10.P_net_min,'r+',label='power_curve_min')
    plt.plot(design.wind,design.power,'k-',label='design_power')
    plt.title('wind power')
    plt.xlabel('wind(m/s)')
    plt.ylabel('powe(kW)')
    plt.grid(True)
    plt.legend(loc='upper left',borderaxespad=0.,fontsize='xx-small')
    plt.savefig(savname,dpi=600)
    #plt.show()
    plt.pause(15)
    plt.close()

savname=savepath+'\\'+'Cp'+GetNowTime()+'.png'
plt.figure(1)
plt.subplot(111)
plt.plot(wind_power.wind,wind_power.Cp,'b-s',label='Cp')
plt.title('wind Cp')
plt.xlabel('wind(m/s)')
plt.ylabel('Cp')
plt.grid(True)
plt.legend(loc='upper left',borderaxespad=0.,fontsize='x-small')
plt.savefig(savname,dpi=600)
#plt.show()
plt.pause(15)
plt.close()
