from tkinter import *
import os
from subprocess import call
from tkinter import messagebox
import os
#f=open("database_proj",'a+')
root = Tk()
#root.title("Simple Pharmacy Managment System")
root.configure(width=93,height=566,bg='BLACK')
var=-1

def rvb():
    call(["python", "rvb.py"])
   
def f():
    call(["python", "f.py"])
     

def bvf():
    call(["python", "bvf.py"])
 

def s():
    call(["python", "s.py"])    

def bvs():
    call(["python", "bvs.py"])


def dataset():
    call(["python", "dataset.py"])

def charts():
    call(["python", "charts.py"])    





#fn1353
label0= Label(root,text="GRAPHAS & CHARTS",bg="black",fg="white",font=("Times", 30))

button1= Button(root, text="RUNS VS BALL FACED", bg="white", fg="black", width=20, font=("Times", 12),command=rvb)
button3= Button(root, text="FOURS" , bg="white", fg="black", width =20, font=("Times", 12),command=f)
button4= Button(root, text="BALL FACED VS FOURS" , bg="white", fg="black", width =20, font=("Times", 12),command=bvf)
button5= Button(root, text="SIX", bg="white", fg="black", width =20, font=("Times", 12),command=s)
button6= Button(root, text="BALL FACED VS SIX", bg="white", fg="black", width =20, font=("Times", 12),command=bvs)
button7= Button(root, text="PIE CHARTS", bg="yellow", fg="black", width =20, font=("Times", 12),command=charts)
button8= Button(root, text="EXIT", bg="green", fg="black", width =20, font=("Times", 12),command=quit)

label0.grid(columnspan=6, padx=10, pady=10)


button1.grid(row=1,column=4, padx=40, pady=10)

button3.grid(row=2,column=4, padx=40, pady=10)
button4.grid(row=3,column=4, padx=40, pady=10)
button5.grid(row=4,column=4, padx=40, pady=10)
button6.grid(row=5,column=4, padx=40, pady=10)
button7.grid(row=3,column=5, padx=40, pady=10)
button8.grid(row=6,column=6, padx=40, pady=10)

root.mainloop()
