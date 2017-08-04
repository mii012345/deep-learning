import csv
import numpy as np
import pickle

with open('data (2).csv','r') as f:
    csv = csv.reader(f)
    csvlist = []
    for i in csv:
        csvlist.append(i)

#6行目から
mas = []
for i in range(364):
    i+=6

    a = 0
    b = 0
    c = 0
    date = csvlist[i][0]
    weather = csvlist[i][1]
    if date[0:10] == "2016/11/1 " or date[0:10] == "2016/11/2 " or date[0:10] == "2016/11/3 " or date[0:9] == "2016/11/4" or date[0:9] == "2016/11/5" or date[0:9] == "2016/11/6" or date[0:9] == "2016/11/7":
        continue

    if weather == "1" or weather == "2":
        a = 1
    elif weather == "3" or weather == "4" or weather == "5" or weather == "6":
        b = 1
    else:
        c = 1
    w = [a,b,c]
    print(date[0:10])
    mas.append(w)
mas = np.array(mas)
with open('tenki_num.pkl','wb') as f:
    pickle.dump(mas,f)
