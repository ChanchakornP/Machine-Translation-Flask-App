import json
from flask import Flask, jsonify, request, render_template
from inference import pred_th2en, pred_en2th
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/model/th2en', methods=['GET', 'POST']) 
def th2en():
    data = request.form['text']
    output = pred_th2en(str(data))
    return str(output)

@app.route('/model/en2th', methods=['GET', 'POST']) 
def en2th():
    data = request.form['text']
    output = pred_en2th(str(data))
    return str(output)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))