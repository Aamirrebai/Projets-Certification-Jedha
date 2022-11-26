from flask import Flask, request, render_template
from PIL import Image
import json
from nbformat import read
import numpy as np
from keras_preprocessing.image.utils import img_to_array
import pandas 
from keras.models import load_model
import os 
import datetime 

app = Flask(__name__)
model = load_model("model.h5")

SIZE = 256


def read_image(path_or_input):
     image = Image.open(path_or_input)
     return image, image.size

def to_vec(image):
     size = (SIZE, SIZE)
     image = image.resize(size)
     vec = img_to_array(image) / 256
     vec = np.expand_dims(vec, 0)
     return vec

def apply_pipeline(path_or_input):
     image, size_original = read_image(path_or_input)
     vec = to_vec(image)
     return model.predict(vec)[0][0], size_original


@app.route("/predict_refactored", methods=["GET"])
def predict_refactored():
     # 
     pathes = ["resize_normal/_7_1509590.png", "resize_pneumonia/_0_4089442.png"]
     predictions = {path: apply_pipeline(path) for path in pathes}
     return render_template("predict_refactored.html", predictions=predictions)



@app.route("/", methods = ['GET', 'POST'])
def index():
     return render_template('home.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
     # on extrait la valeur de target de l'url
     # http://127.0.0.1:5000/predict?target=normal
     target = request.args.get('target', type=str)
     print(target)
     
     # On définit le chemin vers l'image en fonction de la target
     if target == "pneumonia":
          path = "resize_pneumonia/_0_4089442.png"
          message = "Target is correct"
     # idem
     elif target =="normal":
          path = "resize_normal/_7_1509590.png"
          message = "Target is correct"
     # cas où l'url est incorrecte, et la  target pas dans [pneumonia, normal]
     else:
          path = None
          message = "Target isn't correct, its value is : " + target


     if path is not None:
         # on vérifi  si le path existe 
          proba = apply_pipeline(path)
     else:

          proba = None

     return render_template("predict.html", target=target, message=message, proba=proba)


@app.route('/predict_from_file', methods = ['GET','POST'])
def predict_from_file():
     if request.method == "POST":
          image_file = request.files['imagefile']
          proba = apply_pipeline(image_file)
     elif request.method == "GET":
          proba = None
          image_file = None
          print("This is a POST so not expecting any files")
     return render_template('predict_from_file.html', proba=proba, image_file=image_file)




@app.route('/dev', methods = ['GET','POST'])
def dev():


     proba = None


     if request.method == "POST":
          image_file = request.files['imagefile']
          image, _ = read_image(image_file)
          now = datetime.datetime.now().isoformat().replace(".", "_")
          filename = f"imge_uploaded_{now}.png"
          path = os.path.join("static", filename)
          image.save("static/image_uploaded.png", format="png")
          image_displayed = True




     elif request.method == "GET":
          proba = None
          image_file = None
          print("This is a POST so not expecting any files")
          image_displayed = False


     return render_template('dev.html', proba=proba, image_displayed=image_displayed,)


@app.route('/predict_pneumonia', methods = ['GET', 'POST'])
def predict_pneum():

     if request.method == "POST":
          image_file = request.files['imagefile']
          image, _ = read_image(image_file)
          image.save("static/image_uploaded.png", format="png")
          image_displayed = True
          proba = apply_pipeline(image_file)
     
     elif request.method == "GET":
          proba = None
          image_file = None
          print("This is a POST so not expecting any files")
          image_displayed = False
     
     
     return render_template('predict_pneumonia.html', proba=proba, image_displayed=image_displayed,)




@app.route('/predict_boostrap', methods = ['GET', 'POST'])
def predict_boos():

     if request.method == "POST":
          image_file = request.files['imagefile']
          image, original_size = read_image(image_file)
          image.save("static/image_uploaded.png", format="png")
          image_displayed = True
          proba, _ = apply_pipeline(image_file)
     
     elif request.method == "GET":
          proba = None
          image_file = None
          original_size = None
          print("This is a POST so not expecting any files")
          image_displayed = False


     if proba is not None:
          proba_display = str(round(proba * 100, 4)) + " %"
     else:
          proba_display = None
     
     return render_template('boostrap.html', 
     proba=proba, 
     proba_display=proba_display, 
     size=SIZE,
     
     original_size=original_size,
     image_displayed=image_displayed,)




 
if __name__ == "__main__":
    app.run(debug=True)