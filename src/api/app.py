import os
from flask import Flask, flash, request, redirect, url_for,send_from_directory,make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ast import literal_eval
import pathlib
import uuid as myuuid
import json
import jsonpickle
import datetime
from sklearn.preprocessing import StandardScaler
from typing import List

# Relative Imports
from api.version import api_version
from stork_a import StorkA, Classifier, ClassifierFactory, Metadata, BlastocystScore, BlastocystGrade, Morphokinetics, InputImage, Result

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'tif', 'tiff'])

# Define Directories

current_file_dir = os.path.dirname(os.path.realpath(__file__))

static_file_dir = os.path.join(current_file_dir, '../static')

upload_dir = os.path.join(current_file_dir, '../uploads')
pathlib.Path(static_file_dir).mkdir(parents=True, exist_ok=True)

output_dir = os.path.join(current_file_dir, '../output')
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

users_dict = literal_eval(os.environ['USERS_DICT'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods = ['GET'])
def serve_index_page():
    return send_from_directory(static_file_dir, 'index.html')

@app.route('/<path:path>', methods = ['GET'])
def serve_assets(path):
    return send_from_directory(static_file_dir, path)

@app.route('/api/healthcheck', methods = ['GET'])
def healthcheck():
    return json.dumps({'status':'Healthy', 'version':api_version()})

@app.route('/login', methods = ['GET'])
def serve_login_page():
    return send_from_directory(static_file_dir, 'login.html')

@app.route('/api/login', methods = ['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username and users_dict[username] == password:
        response = make_response()
        uuid = str(myuuid.uuid4())
        response.set_cookie('stork-auth', uuid, max_age=3600)
        return response
    return json.dumps({}), 401



@app.route('/api/upload', methods = ['POST'])
def abnormal_normal():
    auth_token = None
    auth_header = request.headers.get('Authorization')
    data = json.loads(request.form['data'])
    metadata_input= data_to_metadata_input(data)

    if auth_header is None or auth_header.split(" ")[1] is None:
        flash('No Authorization header')
        return jsonify({}), 401

    # check if the post request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    # 1. Create request directory
    request_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    request_dir = (os.path.join(upload_dir, request_id))
    pathlib.Path(request_dir).mkdir(parents=True, exist_ok=True)

    # For each uploaded image
    for image in request.files.getlist('image'):
        # 2. Save Image
        filename = secure_filename(image.filename)
        image.save(os.path.join(request_dir, filename))

        classifier_factory = ClassifierFactory()
        result = {}
        result['sample'] = image.filename

        classifier = classifier_factory.make(metadata_input, 'Abnormal-Normal')
        result['abnormalNormal'] = getResultsFromClassifier(classifier, request_dir, metadata_input, filename)[0]
        classifier = classifier_factory.make(metadata_input, 'CxA-EUP')
        result['cxAEUP'] = getResultsFromClassifier(classifier, request_dir, metadata_input, filename)[0]
        classifier = classifier_factory.make(metadata_input, 'CxA-Everything')
        result['cxAEverything'] = getResultsFromClassifier(classifier, request_dir, metadata_input, filename)[0]

    json_response = jsonpickle.encode(result, unpicklable=False)
    return json_response, 200, {'Content-Type': 'application/json; charset=utf-8'}

def getResultsFromClassifier(classifier, request_dir, metadata_input, filename):
    stork_a = StorkA()
    metadata = Metadata(**metadata_input)
    classifier.normalize_metadata(metadata)
    input_images = [InputImage(filename, request_dir, 0, metadata)]
    results = stork_a.eval(classifier, input_images)
    for r in results:
        del r.sample

    return results

def data_to_metadata_input(data):
    bs_data = {}
    bg_data = {}
    morphokinetics_data = {}

    if 'blastocystScore' in data and bool(data["blastocystScore"]):
        bs_data[f'BS.{data["blastocystScore"]}'] = 1

    if 'blastocystGrade' in data and bool(data["blastocystGrade"]) \
        and bool(data["blastocystGrade"]["expansion"]) \
        and bool(data["blastocystGrade"]["innerCellMass"]) \
        and bool(data["blastocystGrade"]["trophectoderm"]):
        bg_data[f'Expansion.{data["blastocystGrade"]["expansion"].replace("-", ".").replace("/", ".")}'] = 1
        bg_data[f'ICM.{data["blastocystGrade"]["innerCellMass"].replace("-", ".").replace("/", ".")}'] = 1
        bg_data[f'TE.{data["blastocystGrade"]["trophectoderm"].replace("-", ".").replace("/", ".")}'] = 1

    if 'morphokinetics' in data and bool(data["morphokinetics"]):
        morphokinetics_data = data["morphokinetics"]

    return {
        'age': data['eggAge'],
        'blastocyst_score': BlastocystScore(**bs_data) if len(bs_data) == 1 and any(elem in bs_data for elem in BlastocystScore().get_params()) else None,
        'blastocyst_grade': BlastocystGrade(**bg_data) if len(bg_data) == 3 and all(elem in BlastocystGrade().get_params() for elem in bg_data) else None,
        'morphokinetics': Morphokinetics(**morphokinetics_data) if all(elem in morphokinetics_data for elem in Morphokinetics().get_params()) else None,
    }
