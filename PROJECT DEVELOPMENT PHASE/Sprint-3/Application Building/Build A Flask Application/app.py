import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model(r'C:/Users/Acer/IBM --TRAINING SESSION/PROJECT/signlanguage.h5')

@app.route('/loginpage',methods=['GET'])

def login():
    if (request.method == 'GET' != " "):
        uname=request.args.get('uname')
        password=request.args.get('pass')
        if uname=='uname' and password=='pass':
         return render_template("afterlogin1.html",name = uname)
    return render_template("login1.html")

@app.route('/homepage')
def home():
    return render_template("afterlogin1.html")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,r'C:/Users/Acer/IBM --TRAINING SESSION/PROJECT/flask/uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=["A","B","C","D","E","F","G","H","I"]
        text="The Predicted Alphabet is : " +str(index[pred[0]])
    return text
if __name__=='__main__':
    app.run(debug=True)