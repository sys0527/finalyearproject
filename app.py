# -Program Name : Intelligen Teaching Web Based Application - app.py
# -Description : Use for generate flask web application and handle request and respond of pages
# -First Written on: 12 Feb 2023
# -Editted on: 22 April 2023

from flask import Flask, render_template, Response
import virtualMouseControl as virtualMouse 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)    
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/virtual_mouse')
def virtual_mouse():
    return render_template('virtual_mouse.html')


@app.route('/video_feed')
def video_feed():
    return Response(virtualMouse.gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)