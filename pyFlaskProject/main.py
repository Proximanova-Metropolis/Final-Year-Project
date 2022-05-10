from flask import *
import json
import time
import sentiwordnet
import vader
import senticnet
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/sentiment', methods=['GET'])
@cross_origin()
def home_page():
    data_set = {'page': 'Home', 'Message': 'test msg', 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/analysisold', methods=['POST'])
@cross_origin()
def home_page2():
    request_data = request.get_json()
    txt1 = request_data['text']
    retval=''
    if txt1 is None:
        retval= 'No Response'
    elif len(txt1) > 0 :
        retval = senticnet.sentiment_analyzer_scores(txt1)
    else:
        retval = 'No Response'
    data_set = {'page': 'Home', 'Message': retval, 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/analysis', methods=['POST'])
@cross_origin()
def home_page3():
    request_data = request.get_json()
    txt1 = request_data['text']
    sn = request_data['sn']
    swn = request_data['swn']
    vd = request_data['vd']
    retval1 = ''
    retval2 = ''
    retval3 = ''
    if txt1 is None:
        retval1 = ''
        retval2 = ''
        retval3 = ''
    elif len(txt1) > 0:
        if sn > 0: retval1 = senticnet.sentiment_analyzer_scores(txt1)
        if swn > 0: retval2 = sentiwordnet.sentiment_analyzer_scores(txt1)
        if vd >0: retval3 = vader.sentiment_analyzer_scores(txt1)
    else:
        retval1 = ''
        retval2 = ''
        retval3 = ''

    data_set = {'senticnet1': retval1, 'sentiwordnet1': retval2, 'vader1':retval3 }
    json_dump = json.dumps(data_set)
    return json_dump

if __name__ == '__main__':
    app.run(port=7775)
