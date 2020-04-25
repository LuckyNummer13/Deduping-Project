

import pandas as pd
import numpy as np
import dedupe
import csv
import os
import unidecode

df = pd.read_csv('myfile.csv')
df['Id'] = df.index
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.drop('index', axis = 1, inplace = True)
df.to_csv('indexed_file.csv', index = False)

def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename, encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        for rows in reader:
            row_id = int(rows['Id'])
            data_d[row_id] = dict(rows)

    return data_d

input_file = 'indexed_file.csv'
output_file = 'deduplicated_file.csv'
settings_file = 'dedupe_settings_file'
training_file = 'dedupe_training_set.json'

data_d = readData(input_file)


if os.path.exists(settings_file):
    with open(settings_file) as f:
        deduper = dedupe.StaticDupe(f)
else:
    fields = [
        {'field': 'company_name', 'type': 'String'}
    ]
    deduper = dedupe.Dedupe(fields)
    if os.path.exists(training_file):
        with open(training_file) as f:
            deduper.prepare_training(data_d,f)
    else:
        deduper.prepare_training(data_d)
        
dedupe.console_label(deduper)
deduper.train(recall = 0.75)

with open(training_file, 'w') as tf:
    deduper.write_training(tf)
    
with open(settings_file, 'wb') as sf:
    deduper.write_settings(sf)
    
print('clustering....')
clustered_dupes = deduper.partition(data_d, 0.7)
print('# duplicate sets', len(clustered_dupes))

cluster_membership = {}
for cluster_id, (records,scores) in enumerate(clustered_dupes):
    for record_id, score in zip(records,scores):
        cluster_membership[record_id] = {
            'Cluster ID': cluster_id,
            'Confidence Score': score
        }
        
with open(output_file, 'w', encoding = 'utf-8', newline = '') as f_output, open(input_file, encoding = 'utf-8') as f_input:

        reader = csv.DictReader(f_input)
        fieldnames = ['Cluster ID', 'Confidence Score'] + reader.fieldnames

        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_id = int(row['Id'])
            row.update(cluster_membership[row_id])
            writer.writerow(row)
            
            