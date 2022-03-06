import os
from charset_normalizer import detect
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import shutil

import augmented

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    FILENAME = {}
    # get file 
    video = request.files['video']
    pattern = request.files['pattern']
    overlay = request.files['overlay']

    # save file
    video.save('static/video.mp4')
    pattern.save('static/pattern.png')
    overlay.save('static/overlay.png')

    if 'video' in request.files:
        detect = True

        # process file
        augmented.augmented(
            pattern_path='static/pattern.png',
            overlay_path='static/overlay.png',
            video_path='static/video.mp4',
            output_path='static/output.mp4',
            notebook_mode=True
            )

        # open file TODO:
        with open(f'static/output.mp4', 'rb') as videoFile:
            encode_video = base64.b64encode(videoFile.read())
            to_send = 'data:video/mp4;base64, ' + str(encode_video, 'utf-8')
    else:
        detect = False

    return render_template('index.html', init=True, detect=detect, image_to_show=to_send)

if __name__ == '__main__':
    try:
        os.makedirs('static')
    except:
        print('Directory already exist!')
    app.run()
    shutil.rmtree('static')