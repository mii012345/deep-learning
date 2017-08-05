import urllib.request
import sys
import numpy as np

url="http://weather.is.kochi-u.ac.jp/sat/gms.fareast/"
a=1
b=0
x = input("Please Enter Year You Want: ")
y = input("And Enter Folder You Save File: ") + "/"
c=[]
for ii in range(1,13):
    if ii < 10:
        str_a = "0"+str(a)
    else:
        str_a = str(a)
    for i in range(1,32):
        if i<10:
            url="http://weather.is.kochi-u.ac.jp/sat/gms.fareast/"+x+"/"+str_a+"/0"+str(i)+"/fe."+x[2:]+str_a+"0"+str(i)+"09.jpg"
            title=y+str(b)+".jpg"
        else:
            url="http://weather.is.kochi-u.ac.jp/sat/gms.fareast/"+x+"/"+str_a+"/"+str(i)+"/fe."+x[2:]+str_a+str(i)+"09.jpg"
            title=y+str(b)+".jpg"
        try:
            test = 0
            #urllib.request.urlretrieve(url,title)
        except:
            if ii == 2:
                if i == 29:
                    ii+=1
                    continue
            if ii == 4 or ii == 6 or ii == 9 or ii == 11:
                if i == 31:
                    ii+=1
                    continue
            d = str(ii)+"/"+str(i)
            c = np.append(c,[d],axis=0)
            continue
        print(b,url)
        b+=1
    a+=1
print("I Was Able To Download Files About "+str(x))
print("This Is Files I Could Not Download:")
for i in c:
    print(i)
