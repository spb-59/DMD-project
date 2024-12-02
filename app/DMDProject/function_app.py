import pickle
import uuid
import azure.functions as func
from wfdb.io import rdrecord
from wfdb.processing import normalize_bound
import logging
import pyodbc
import json
from requests_toolbelt.multipart import decoder
import os
from preprocess import denoise
from feature_extraction import extract,make_list
from model import classify
from plotting import make_plots,make_metric 
import datetime


app = func.FunctionApp()


@app.route(route="create_patient", auth_level=func.AuthLevel.FUNCTION)
def create_patient(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_body = req.get_json()

        name= request_body.get('name')
        DOB = request_body.get('DOB')
        contact=request_body.get('contact')

    except ValueError:
        return func.HttpResponse(
            "Invalid  request",
            status_code=400
    )
    try:
        conn_string=os.environ['CONNECTION_STR']
        conn = pyodbc.connect(conn_string)
        cursor=conn.cursor()
    except Exception as e:
        logging.error(f'Cant connect to database: {e}')
        return  func.HttpResponse(
            json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
            status_code=500,
            mimetype="application/json"
        )

    try:
        _id=uuid.uuid4()
        query = 'INSERT INTO Patient (patientID,name,DOB,contact) VALUES (?, ?,?,?)'
        cursor.execute(query, (_id,name,DOB,contact))
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting record: {e}")
        conn.rollback()
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Error creating record"}),
            status_code=400,
            mimetype="application/json"
        )
    
    return func.HttpResponse(
            json.dumps({"status": "Successful", "message": "Error creating record",'patientID':str(_id)}),
            status_code=200,
            mimetype="application/json"
        )


    



    



@app.route(route="insert_record", auth_level=func.AuthLevel.FUNCTION)
def insert_record(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing file upload...")
    # return func.HttpResponse(
    #     json.dumps({"status": "success", 'recordID': 'b0552d75-0928-417c-8c92-e87e084ecf04'}),
    #     status_code=200,
    #     mimetype="application/json"
    # )


    try:
        
        content_type = req.headers.get('Content-Type')
        if not content_type or 'multipart/form-data' not in content_type:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Content-Type must be multipart/form-data"}),
                status_code=400,
                mimetype="application/json"
            )
        body = req.get_body()
        
        multipart_data = decoder.MultipartDecoder(body, content_type)

        file1 = None
        file2 = None
        file_name1=None
        file_name2=None
        for part in multipart_data.parts:
            content_disposition = part.headers.get(b"Content-Disposition")
            if content_disposition:
                disposition_str = content_disposition.decode()
                if 'name="signal"' in disposition_str:
                    file1 = part.content 
                    file_name1 = disposition_str.split('filename="')[1].split('"')[0]
                elif 'name="header"' in disposition_str:
                    file2 = part.content  
                    file_name2 = disposition_str.split('filename="')[1].split('"')[0]
                elif 'name="patientID"' in disposition_str:  
                    user_id = part.content.decode()

        if not file1 or not file2 or not user_id:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Both 'signal' and 'header' files are required"}),
                status_code=400,
                mimetype="application/json"
            )

        
        # Save files to a temporary directory for further processing
        temp_path = os.getenv('TMP', '/tmp')  # Temporary directory in Azure Functions
        signal_path = os.path.join(temp_path, f'{file_name1}')
        header_path = os.path.join(temp_path, f'{file_name2}')


        with open(signal_path, 'wb') as f:
            f.write(file1)
        with open(header_path, 'wb') as f:
            f.write(file2)

        logging.info(f'Upload Complete as {signal_path} & {header_path}')

        base_file_name = file_name2.split('.')[0]  # without extension
        record = rdrecord(temp_path+'/'+base_file_name)
        
        logging.info('Record Loaded')

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": f"Error processing files: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

    sf=record.fs
    record.p_signal=normalize_bound(record.p_signal)
    signal=record.to_dataframe()
    
    signal=denoise(signal,sf)

    signal=signal.to_numpy()
    frames = [signal[i:i+2000] for i in range(0, len(signal), 2000)]

    try:
        conn_string=os.environ['CONNECTION_STR']
        conn = pyodbc.connect(conn_string)
    except Exception as e:
        logging.error(f'Cant connect to database: {e}')
        return  func.HttpResponse(
            json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
            status_code=500,
            mimetype="application/json"
        )


    cursor=conn.cursor()
    logging.info('Start Creating Record')
    record_id = uuid.uuid4()
    try:
        query = 'INSERT INTO Record (patientID, recordID, date) VALUES (?, ?,?)'
        cursor.execute(query, (user_id, record_id, datetime.datetime.now()))
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting record: {e}")
        conn.rollback()
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Error creating record"}),
            status_code=400,
            mimetype="application/json"
        )

    frames_data = [(uuid.uuid4(), pickle.dumps(frame), record_id) for frame in frames]
    try:
        insert_query = "INSERT INTO Frame (frameID, frameData, recordID) VALUES (?, ?, ?)"
        cursor.executemany(insert_query, frames_data)
        frame_ids = [row[0] for row in frames_data]
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting frames: {e}")
        conn.rollback()
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Error inserting frames"}),
            status_code=400,
            mimetype="application/json"
        )
    
    features=[extract(frame) for frame in frames]
    features_data = []
    for index, feature in enumerate(features):
        features_data.append((
            frame_ids[index], uuid.uuid4(), feature['R_N'], feature['R_L'], feature['R_M'], feature['R_P'],
            feature['Lam_min'], feature['Lam_max'], feature['M_u'].mean(), feature['P_u'].mean(),
            feature['M_s'].mean(), feature['P_s'].mean()
        ))
    try:
        feature_query = '''
            INSERT INTO Features (frameID, featureID, R_N, R_L, R_M, R_P, Lam_min, Lam_max, M_u, P_u, M_s, P_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(feature_query, features_data)
        feature_ids = [row[1] for row in features_data]
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting features: {e}")
        conn.rollback()
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Error inserting features"}),
            status_code=400,
            mimetype="application/json"
        )
    predictions=[classify(make_list(feature)) for feature in features]
    results_data = [
        (uuid.uuid4(), int(pred[0]), feature_ids[index]) for index, pred in enumerate(predictions)
    ]
    try:
        result_query = 'INSERT INTO Result (resultID, prediction, featureID) VALUES (?, ?, ?)'
        cursor.executemany(result_query, results_data)
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting results: {e}")
        conn.rollback()
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Error inserting results"}),
            status_code=400,
            mimetype="application/json"
        )

    cursor.close()
    conn.close()

    return func.HttpResponse(
        json.dumps({"status": "success", 'recordID': str(record_id)}),
        status_code=200,
        mimetype="application/json"
    )






@app.route(route="get_metric", auth_level=func.AuthLevel.FUNCTION)
def get_metric(req: func.HttpRequest) -> func.HttpResponse:
    try:
        record_id = req.params.get('record_id')
        
        if not record_id:
            return func.HttpResponse(
                "Missing 'record_id' in request.",
                status_code=400
            )
    except ValueError:
        return func.HttpResponse(
            "Invalid  request",
            status_code=400
        )
    
    try:
        conn_string=os.environ['CONNECTION_STR']

        conn = pyodbc.connect(conn_string)

    except Exception as e:
        logging.error(f'Error connection to databse: {e}')
        return  func.HttpResponse(
            json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
            status_code=500,
        )
    cursor=conn.cursor()

    try:
        query='''
        SELECT 
            *
        FROM 
            Patient p
        JOIN 
            Record rec ON p.patientID = rec.patientID
        JOIN 
            Frame fr ON rec.recordID = fr.recordID
        JOIN 
            Features f ON fr.frameID = f.frameID
        JOIN 
            Result r ON f.featureID = r.featureID
        WHERE 
            rec.recordID = ?;
        '''
        cursor.execute(query,(record_id))

        data=cursor.fetchall()
        data=[dict(zip([column[0] for column in cursor.description], row)) for row in data]
        plots=make_plots(data)
        metric=make_metric(data)
        predictions=[int(p['prediction']) for p in data ]
        comments=[(c['comment'],c['resultID']) for c in data]
        record=(data[0]['recordID'],data[0]['record_comment'],data[0]['date'].strftime("%Y-%m-%d %H:%M:%S"),data[0]['DOB'].strftime("%Y-%m-%d %H:%M:%S"),data[0]['name']) 


    except Exception as e:
        logging.error(f'Error in getting record data, {e}')
        return  func.HttpResponse(
            json.dumps({"status": "error", "message": "Could'nt make plots try again"}),
            status_code=400,
        )
    
    response = {
    "plots": plots,
    "metric": metric,
    "predictions": predictions,
    "comments": comments,
    'record':record
}

    




    return func.HttpResponse(
       json.dumps( response),
        mimetype="application/json",
        status_code=200
    )



@app.route(route="update_frame_comment", auth_level=func.AuthLevel.FUNCTION)
def get_frame_comment(req: func.HttpRequest) -> func.HttpResponse:
        try:
            request_body = req.get_json()

            comment = request_body.get('comment')
            _id = request_body.get('resultID')

        except ValueError:
            return func.HttpResponse(
                "Invalid  request",
                status_code=400
        )


        try:
            conn_string=os.environ['CONNECTION_STR']

            conn = pyodbc.connect(conn_string)

        except Exception as e:
            logging.error(f'Error connection to databse: {e}')
            return  func.HttpResponse(
                json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
                status_code=500,
            )
        cursor=conn.cursor()

        try:
            query='UPDATE Result SET comment=? where resultID=?'
            cursor.execute(query,(comment,_id))
            conn.commit()
        except Exception as e:
            logging.error(f"Error inserting results: {e}")
            conn.rollback()
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Error inserting results"}),
                status_code=400,
                mimetype="application/json"
            )
        return func.HttpResponse(
                json.dumps({"status": "Successful", "message": "Comment updated successfully"}),
                status_code=200,
                mimetype="application/json"
            )
            


@app.route(route="update_record_comment", auth_level=func.AuthLevel.FUNCTION)
def get_record_comment(req: func.HttpRequest) -> func.HttpResponse:
        try:
            request_body = req.get_json()

            comment = request_body.get('comment')
            _id = request_body.get('recordID')

        except ValueError:
            return func.HttpResponse(
                "Invalid  request",
                status_code=400
        )


        try:
            conn_string=os.environ['CONNECTION_STR']

            conn = pyodbc.connect(conn_string)

        except Exception as e:
            logging.error(f'Error connection to databse: {e}')
            return  func.HttpResponse(
                json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
                status_code=500,
            )
        cursor=conn.cursor()

        try:
            query='UPDATE Record SET record_comment=? where recordID=?'
            cursor.execute(query,(comment,_id))
            conn.commit()
        except Exception as e:
            logging.error(f"Error inserting results: {e}")
            conn.rollback()
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Error inserting results"}),
                status_code=400,
                mimetype="application/json"
            )
        return func.HttpResponse(
                json.dumps({"status": "Successful", "message": "Comment updated successfully"}),
                status_code=200,
                mimetype="application/json"
            )
            
@app.route(route="get_search", auth_level=func.AuthLevel.FUNCTION)
def search_patient(req: func.HttpRequest) -> func.HttpResponse:
    try:
        q = req.params.get('q')
        
        if not q:
            return func.HttpResponse(
                "Missing 'record_id' in request.",
                status_code=400
            )
    except ValueError:
        return func.HttpResponse(
            "Invalid  request",
            status_code=400
        )
    try:
            conn_string=os.environ['CONNECTION_STR']

            conn = pyodbc.connect(conn_string)

    except Exception as e:
            logging.error(f'Error connection to databse: {e}')
            return  func.HttpResponse(
                json.dumps({"status": "error", "message": "Couldn't connect to Data base"}),
                status_code=500,
            )
    cursor=conn.cursor()

    try:
            query="""
    SELECT name,DOB,rec.recordID,p.patientID,date
    FROM Patient p
    JOIN Record rec ON p.patientID = rec.patientID
    WHERE p.name LIKE ? 
       OR p.patientID = ? 
       OR rec.recordID = ?
"""

            cursor.execute(query,(q+'%',q,q))
            data=cursor.fetchall()
            logging.info(data)

    except Exception as e:
            logging.error(f"Error inserting results: {e}")
            conn.rollback()
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Error inserting results"}),
                status_code=400,
                mimetype="application/json"
            )
    result=[]    
    for d in data:
        result.append({
            'name':d[0],
            'DOB':d[1].strftime("%Y-%m-%d "),
            'recordID':d[2],
            'patientID':d[3],
            'date':d[4].strftime("%Y-%m-%d %H:%M:%S")
        })

    return func.HttpResponse(
                json.dumps({"status": "Successful", "message": "Comment updated successfully",'data':result}),
                status_code=200,
                mimetype="application/json"
            )



    







    

