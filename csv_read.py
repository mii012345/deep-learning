import csv
import cupy as np
import pickle

with open('data (4).csv','r') as f:
    csv = csv.reader(f)
    csvlist = []
    for i in csv:
        csvlist.append(i)

#6行目から
mas = []
for i in range(204):
    i+=6

    a = 0
    b = 0
    c = 0
    date = csvlist[i][0]
    weather = csvlist[i][1]
    if date[0:10] == "2017/1/1 9" or date[0:10] == "2017/1/2 9" or date[0:10] == "2017/1/3 9"  or date[0:10] == "2017/1/5 9":
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
with open('tenki_test.pkl','wb') as f:
    pickle.dump(mas,f)
