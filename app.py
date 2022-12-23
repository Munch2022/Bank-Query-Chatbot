from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import numpy as np

from response import BotReply
botreply= BotReply()

# Creating flask app object
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    if request.method == 'POST':
        message= request.form['message']
        my_pred= botreply.get_response(message)
    return render_template('index.html', prediction_text= my_pred)




if __name__ == '__main__':
    app.run(debug=True) 



# if __name__ == '__main__':
# 	app.run(host="0.0.0.0", port=8080)