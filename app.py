import os
import json
import uuid
import time
import requests
import warnings
import numpy as np
from PIL import Image
from flask_cors import CORS 
import face_recognition as fr
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from spoof.src.utility import parse_model_name
from spoof.src.generate_patches import CropImage
from spoof.src.anti_spoof_predict import AntiSpoofPredict

warnings.filterwarnings('ignore')
load_dotenv()
app = Flask(__name__)
CORS(app)

# from env
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL")
confidence = 0
index = None
def real_time_face_recognition(known_name_encodings, known_names, face_encodings):
    global confidence
    results = None
    for face_encoding in face_encodings:
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]['Name']
        person = known_names[best_match_index]['filename']
        code = "DRAFT"
        tmp = 1 - face_distances[best_match_index]
        confidence = tmp
        if confidence > 0.55:
            name = known_names[best_match_index]['Name']
            person = known_names[best_match_index]['filename']
            code = "SUCCESS"
            response = requests.post(MAIN_SERVER_URL+"/recognize",json={"face_id":name, "code":code,"confidence_score":confidence},headers={"Content-Type": "application/json"})
            response = response.json()
            print(response)
            results = {"name": name, "confidence": float(confidence), "person": person, "code": code, "token": response["token"]}
        else:
            name = "UNKNOWN"
            person = "UNKNOWN"
            code = "NO_FACE"
            results = {"name": name, "confidence": float(confidence), "person": person, "code": code}
    return results

def add_encoding(face_id,filename, encoding):
    try:
        file = open('encoding_data.json', 'r')
        fileStr = file.read()
        encoding_data = {}
        if fileStr == '':
            encoding_data = {}
        else:
            encoding_data = json.loads(fileStr)
        encoding_data[face_id] = {"filename": filename, "encoding": encoding}
        file = open('encoding_data.json', 'w')
        file.write(json.dumps(encoding_data))
        print('DOne')
    except Exception as e:
        print("error in file:",e)

def get_encoding(face_id):
    encoding_data = json.loads(open('encoding_data.json').read())
    return encoding_data[face_id]

def get_all_encodings():
    # return face name and encodings
    encoding_data = json.loads(open('encoding_data.json').read())
    known_names = []
    known_name_encodings = []
    for face_id, data in encoding_data.items():
        known_names.append({"Name": face_id, "filename": data['filename']})
        known_name_encodings.append(data['encoding'])
    return known_names, known_name_encodings

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def crop_center(img, aspect_ratio):
    width, height = img.size
    new_height = width / aspect_ratio

    if new_height > height:
        new_width = height * aspect_ratio
        left = (width - new_width) / 2
        top = 0
        right = (width + new_width) / 2
        bottom = height
    else:
        new_width = width
        left = 0
        top = (height - new_height) / 2
        right = width
        bottom = (height + new_height) / 2

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped

def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = Image.open(image)
    # image = image.convert('RGB')
    image = crop_center(image, 3/4)
    image = np.array(image)
    # result = check_image(image)
    # if result is False:
    #     return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    print("label: ", label, "value: ", value)
    return label

@app.route('/recognize', methods=['POST'])
def recognize_face():
    global index
    try:
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        filename = request.form['filename']
        unique_id = str(uuid.uuid4())
        file.save(filename)
        label = test(file, model_dir='./spoof/resources/anti_spoof_models', device_id=0)
        if label != 1:
            print("Image is not Real Face")
            os.remove(filename)
            return jsonify({"results": "Image is not Real Face","code":"FAKE_FACE"})
        else:
            print('Image is live')    
            imgArry = fr.load_image_file(file)
            known_names,known_name_encodings = get_all_encodings()
            face_encodings = fr.face_encodings(imgArry)
            # faiss_wrapper(compare_encodings=face_encodings)
            results = real_time_face_recognition(known_name_encodings, known_names, face_encodings)
            os.remove(filename)       
            return jsonify(results) 
    except Exception as e:
        print('EXCEPTIONS:',e)
        return jsonify({"error": e})
    # finally:
        

@app.route('/upload', methods=['POST'])
def upload_encodings():
    global index
    try:
        if 'file' not in request.files:
            return 'No file part'
        if 'filename' not in request.form:
            return 'No filename provided'
        file = request.files['file']
        filename = request.form['filename']
        email = request.form['Email']
        name = request.form['Name']
        face_id = str(uuid.uuid4())
        imgArry = fr.load_image_file(file)
        encoding = fr.face_encodings(imgArry)
        # faiss_wrapper(add_encodings=encoding)
        add_encoding(face_id, filename, encoding[0].tolist())
        response = requests.post(MAIN_SERVER_URL+"/upload",json={"email":email,"name":name,"face_id":face_id},headers={'Content-Type': 'application/json'})
        if response.status_code != 200:
            return jsonify({"error": "Error uploading encodings"})
        return jsonify({"face_id": face_id})
    except Exception as e:
        print(e)
        return jsonify({"error": e})


if __name__ == "__main__":
    app.run(debug=True)