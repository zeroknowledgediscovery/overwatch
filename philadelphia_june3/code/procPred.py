import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal
import yaml
import argparse

with open("config.yaml", "r") as stream:
    try:
        CONFIG=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def calculate_and_normalize_probabilities(data, covariance_matrix):
    # Calculate the Gaussian probabilities for each point
    coordinates = np.vstack((data['lat'], data['lon'])).T
    probabilities = np.zeros(len(data))

    for coordinate in coordinates:
        rv = multivariate_normal(coordinate, covariance_matrix)
        probabilities += rv.pdf(coordinates)

    # Normalize the probabilities within the group
    return probabilities / probabilities.sum()

def procPred(file_path, covariance_value=0.0001):
    

    data = pd.read_csv(file_path)

    covariance_matrix = np.array([[covariance_value, 0], [0, covariance_value]])
    data['normalized_probability'] = 0

    for typ, group_data in data.groupby('typ'):
        probabilities = calculate_and_normalize_probabilities(group_data,
                                                              covariance_matrix)
        data.loc[group_data.index, 'probability'] = probabilities

    jsondata=csv_to_json(data,file_path,outfile=file_path.replace('.csv','.json'))
     
    return jsondata


def csv_to_json(data,file_path,outfile):
    modelUpdateAt = datetime.strptime(CONFIG['train_end_date'],
                                      "%Y-%m-%d").isoformat()
    createdAt = datetime.strptime(CONFIG['oos_end'], "%Y-%m-%d").isoformat()
    predictionStartTime = datetime.strptime(file_path.split('/')[-1]
                                            .replace('.csv','')
                                            .replace('prediction_',''),
                                            "%Y-%m-%d").isoformat()
    predictionEndTime = (datetime.strptime(file_path.split('/')[-1]
                                           .replace('.csv','')
                                           .replace('prediction_',''), "%Y-%m-%d")
                         + timedelta(days=1)).isoformat()
    type_mapping = {'PROP': 'property', 'VIOL': 'violent', 'NARC': 'narcotic'}
    
    predictions = []
    for index, row in data.iterrows():
        # Map the event type
        event_type = type_mapping.get(row['typ'], row['typ'])
        
        # Create the JSON structure for each row
        prediction = {
            "prediction": {
                "lat": row['lat'],
                "lon": row['lon'],
                "type": event_type
            },
            "modelUpdateAt": modelUpdateAt,
            "createdAt": createdAt,
            "predictionStartTime": predictionStartTime,
            "predictionEndTime": predictionEndTime,
            "probability": row['probability'] if 'probability' in row else 1.0 
        }
        predictions.append(prediction)
    
    json_data = json.dumps(predictions, indent=2)


    with open(outfile, 'w') as file:
        file.write(json_data)
    
    return json_data

parser = argparse.ArgumentParser(description="Process predictions and generate JSON.")
parser.add_argument("file_path", type=str, help="The file path of the prediction CSV to process.")
args = parser.parse_args()
file_path=args.file_path

procPred(file_path)
