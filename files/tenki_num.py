from PIL import Image
import cupy as cp
import pickle
import os.path as p

def loadPic():
    if p.exists("tenki/tenki_pic.pkl"):
        with open('tenki/tenki_pic.pkl','rb') as f:
            mas = pickle.load(f)
    else:
        mas = cp.empty((0,3,360,360),int)
        for i in range(357):
            st = "tenki/"+str(i)+".jpg"
            pic = Image.open(st)
            im = cp.array(pic.crop((280,120,640,480)))
            im = im.T
            mas = cp.append(mas,[im],axis=0)
        with open('tenki/tenki_pic.pkl','wb') as f:
            pickle.dump(mas,f)
    with open('tenki/tenki.pkl','rb') as f:
        tenki = pickle.load(f)

    if p.exists("test/test_pic.pkl"):
        with open('test/test_pic.pkl','rb') as f:
            mastest = pickle.load(f)
    else:
        mastest = cp.empty((0,3,360,360),int)
        for i in range(205):
            st = "test/"+str(i)+".jpg"
            pic = Image.open(st)
            im = cp.array(pic.crop((280,120,640,480)))
            im = im.T
            mastest = cp.add(mastest,im)
        with open('test/test_pic.pkl','wb') as f:
            pickle.dump(mas,f)
    with open('test/tenki_test.pkl','rb') as f:
        tenkitest = pickle.load(f)
    return mas,tenki,mastest,tenkitest
