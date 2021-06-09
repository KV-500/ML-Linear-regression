from tkinter import *
import os
from subprocess import call
from tkinter import messagebox
import os
#f=open("database_proj",'a+')
root = Tk()
#root.title("Simple Pharmacy Managment System")
root.configure(width=566,height=566,bg='white')
var=-1

def ror():
    call(["python", "ror.py"])
   
def fs():
    call(["python", "fs.py"])
     

def rsf():
    call(["python", "rsf.py"])
 





#fn1353
label0= Label(root,text="PIE CHATRS",bg="black",fg="yellow",font=("Times", 30))

button1= Button(root, text="RANGE OF RUNS", bg="white", fg="black", width=30, font=("Times", 12),command=ror)
button2= Button(root, text="FOURS & SIX" , bg="white", fg="black", width =20, font=("Times", 12),command=fs)
button3= Button(root, text="RUNS VS SIX&FOURS" , bg="white", fg="black", width =20, font=("Times", 12),command=rsf)
button4= Button(root, text="EXIT", bg="green", fg="black", width =20, font=("Times", 12),command=quit)

label0.grid(columnspan=6, padx=10, pady=10)


button1.grid(row=1,column=4, padx=40, pady=10)
button2.grid(row=2,column=4, padx=40, pady=10)
button3.grid(row=3,column=4, padx=40, pady=10)
button4.grid(row=4,column=4, padx=40, pady=10)


root.mainloop()
