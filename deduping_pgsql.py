# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:07:27 2020

@author: X218102
"""


from __future__ import print_function, division

import dedupe, dedupe.backport
from unidecode import unidecode
import os, re,sys, time
import csv
import itertools
import boto3
from io import StringIO
import psycopg2
from config import config, DBconnection
#import config
import pandas as pd
from sqlalchemy.types import VARCHAR, Float

# ## Setup
current_path = 'C:\Users\xxxxxxx\Desktop\*******\Deduping\postgresql'
output_file =    current_path + '/deduped_list.csv'
settings_file =  current_path + '/deduping_settings'
training_file =  current_path + '/deduping_training.json'

# sys.path.append('C:/shared/Python Scripts/****/****')

# preProcess function 
'''def preProcess(column):
    column = unidecode(str(column))
    column = re.sub('\n', ' ', column)
    column = re.sub('-', ' ', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(",", ' ', column)
    column = re.sub(":", ' ', column)
    column = re.sub(' +', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if not column:
        column = None
    return column'''


def readData(c_results):  #     Read from query results 
    data_d = {}
    for i, row in enumerate(c_results):
        cust = row['customer_id']
        clean_row = dict([(k, preProcess(v)) for (k, v) in row.items()])
        clean_row['customer_id'] = cust
        if cust:
            data_d[cust] = dict(clean_row)
        else: 
            data_d[str(i)] = dict(clean_row)
    return data_d


# get data from database

sql_query_1 = "custom sql query limit 3000"

sql_query_2 ="custom sql query"
 
sql_query ="custome sql query "

# get messy and canonical data
params_local = config(section='pg_local')
with DBconnection(**params_local, dict_cursor=True) as cur:
    print('importing messy data')
    cur.execute(sql_query)1)
    messy_data = readData(cur) 
    print('importing canonical data...') 
    cur.execute(sql_query_2)
    canonical_data = readData(cur)

print('N data 1 records: {}'.format(len(messy_data)))
print('N data 1 records: {}'.format(len(canonical_data)))

# dict(list(canonical_data.items())[0:2])

# get recursion limit 
# sys.getrecursionlimit()
sys.setrecursionlimit(5000)


#### =====   load labelled pairs from database  =====#####
#-- , trim(CONCAT(coalesce(b.city,''), ', ', coalesce(b.state,''), ' ', coalesce(b.zip, ''))) city

psql_labelled = "select a.label, b.customer_id, b.name, b.address \
,b.city, b.state, b.zip, b.country, b.phone \
,c.account_id as customer_id2, c.name as name2, c.address as address2 \
,c.city as city2, c.state as state2, c.zip as zip2, c.country as country2,c.phone as phone2 \
    from dedupe.labeled_for_training a join public.erp_shipto b on a.customer1_id = b.customer_id \
join public.sfdc_accounts c on a.customer2_id = c.account_id "

# get labelled pairs as df
params_local = config(section='pg_local')
with DBconnection(**params_local) as conn:
    df_label = pd.read_sql_query(psql_labelled, conn)
 
labeled_examples = {'match': [], 'distinct': []}
for index, row in df_label.iterrows(): 
    cust1 = dict([(k, preProcess(row[k]) or None) for k in row.keys() if not k.endswith('2')])
    cust1['customer_id'] = row['customer_id']
    cust1.pop('label', None)
    cust2 = dict([(k[:-1], preProcess(row[k]) or None) for k in row.keys() if k.endswith('2')])
    cust2['customer_id'] = row['customer_id2']
    labeled_examples[row['label']].append((cust1, cust2))
#### =================================================



# labeling and training 
if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as sf:
        gazetteer = dedupe.StaticGazetteer(sf)
else:
    fields = [{'field' : 'name', 'type': 'String'},
              {'field' : 'address', 'type': 'String', 'has missing' : True},
              {'field' : 'city', 'type': 'String', 'has missing' : True},
              {'field' : 'state', 'type': 'Exact', 'has missing': True},
              {'field' : 'zip', 'type': 'ShortString', 'has missing' : True},
              {'field' : 'phone', 'type': 'ShortString', 'has missing' : True},
              {'field' : 'country', 'type': 'Exact', 'has missing' : True}
              ]
 
    # Create a new gazetteer object and pass our data model to it.
    gazetteer = dedupe.Gazetteer(fields, num_cores=2)
 
    gazetteer.markPairs(labeled_examples)
    print("labelled pairs are loaded from database")
      
    #print("Prepair for training using sample data")
    gazetteer.prepare_training(messy_data,canonical_data, blocked_proportion=0.6)
    # gazetteer.sample(messy_erp, canonical_sfdc, 15000)   
    # deduper.prepare_training(temp_d, training_file=tf)
    #del messy_erp
          
    # ## Active learning
    dedupe.consoleLabel(gazetteer)    
    gazetteer.train(index_predicates=True)

    # When finished, save our training away to disk
    with open(training_file, 'w') as tf:
        gazetteer.writeTraining(tf)
    
    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_file, 'wb') as sf:
        gazetteer.writeSettings(sf, index=True)

    gazetteer.cleanupTraining()
    
    
    
      
#############################
#    set threshold for matching
gazetteer.index(canonical_data)
# Calc threshold
print('Start calculating threshold')
threshold = gazetteer.threshold(messy_erp, recall_weight=1.1)
print('Threshold: {}'.format(threshold))


#####################################################
####  save the traing label to database   ################
# create a table for manually labeled pairs
conn = psycopg2.connect(**params_local)
c = conn.cursor()
# c.execute("DROP TABLE IF EXISTS dedupe.labeled_for_training")
# c.execute("CREATE TABLE dedupe.labeled_for_training"
#           "(id SERIAL primary key, customer1_id VARCHAR(100) not null, customer2_id VARCHAR(100) not null, "
#           " label VARCHAR(30) ) "  )  

# c.execute("create UNIQUE INDEX idx_id_id2 on dedupe.labeled_for_training (customer1_id, customer2_id)")

import json
with open(training_file) as tf :
    training_pairs = json.load(tf)

# markPairs wants this dict
labeled_examples = {'match': [], 'distinct': []}

postgres_insert ="INSERT INTO dedupe.labeled_for_training (customer1_id, customer2_id, label) VALUES (%s,%s,%s)  ON CONFLICT (customer1_id, customer2_id) DO NOTHING"
# ON CONFLICT (customer1_id, customer2_id) DO NOTHING

# c.execute("SET SCHEMA 'dedupe'")
for pair in training_pairs['distinct']:  
    labeled_examples['distinct'].append((pair[0]['customer_id'], pair[1]['customer_id']))
    record_to_insert = (pair[0]['customer_id'], pair[1]['customer_id'], 'distinct')
    c.execute(postgres_insert_distinct, record_to_insert)

for pair in training_pairs['match']:
    labeled_examples['match'].append((pair[0]['customer_id'], pair[1]['customer_id']))
    record_to_insert = (pair[0]['customer_id'], pair[1]['customer_id'], 'match')
    c.execute(postgres_insert, record_to_insert)

conn.commit()
# conn.rollback()
c.close()
conn.close()

''' work around for insert ignore
for i in range(len(df)):
    try:
        df.iloc[i:i+1].to_sql(name="Table_Name",if_exists='append',con = Engine)
    except IntegrityError:
        pass #or any other action
'''
##########################################################   

def readData2(c_results):
    #     Read from query results 
    for i, row in enumerate(c_results):
        cust = row['customer_id']
        clean_row = dict([(k, preProcess(v)) for (k, v) in row.items()])
        clean_row['customer_id'] = cust
        if cust:
            yield (cust, clean_row) 
            # data_d[clean_row['customer_id']] = dict(clean_row)
        else: 
            yield (str(i), clean_row) 

# matching #########
start_time = time.time()

customer_all = "select a.customer_id, a.name, a.address,a.city, a.state, a.zip, a.country, a.phone \
from  erp_data a where a.region = '**' and a.source not in( '***') "

with DBconnection(**params_local, dict_cursor=True) as cur:
    cur.execute(customer_all)
    full_data = readData2(cur) #  ((row['customer_id'], row) for row in c)

    i = 0 
    step_size = 50000
    results = []
    for step in [step_size]* 300 : 
        i +=1
        chunks = dict(itertools.islice(full_data, step))
        results = results + gazetteer.match(chunks, threshold=threshold)
        
        print(i, len(results))
        if len(chunks) < step_size :
            break; 

print('Scoring ran in', time.time() - start_time, 'seconds, after step=', i)


with DBconnection(**params_local) as conn:
    c=conn.cursor()
    c.execute("DROP TABLE IF EXISTS dedupe.sfdc_data_map")    ##   c.execute("set schema 'dedupe'"
    c.execute("CREATE TABLE dedupe.sfdc_data_map  (id serial primary key, customer_id VARCHAR(100) not null, account_id VARCHAR(100) not null, score float ) "  ) 
    c.execute("create UNIQUE INDEX uniq_sfdc_erp_map on dedupe.sfdc_data_map (customer_id)")
    postgres_insert ="INSERT INTO dedupe.sfdc_data_map (customer_id, account_id, score) VALUES (%s, %s, %s) ON CONFLICT (customer_id) DO NOTHING"
    results_list = []    
    for row in results:
        results_list.append([row['pairs'][0][0],row['pairs'][0][1], row['score'][0]])
        c.execute(postgres_insert, (row['pairs'][0][0],row['pairs'][0][1], float(row['score'][0])))
    conn.commit()
    c.close()

results_df = pd.DataFrame(results_list, columns=["customer_id", "account_id", "score"])

'''
with DBconnection(**params_local) as conn:
    c=conn.cursor()
    c.execute("DROP TABLE IF EXISTS dedupe.sfdc_data_map_bunit")    ##   c.execute("set schema 'dedupe'"
    c.execute("CREATE TABLE dedupe.sfdc_data_map_bunit  (id serial primary key, customer_id VARCHAR(100) not null, account_id VARCHAR(100) not null, score float ) "  ) 
    c.execute("create UNIQUE INDEX uniq_sfdc_data_map_bunit on dedupe.sfdc_data_map_bunit (customer_id)")
    postgres_insert ="INSERT INTO dedupe.sfdc_data_map_bunit (customer_id, account_id, score) VALUES (%s, %s, %s) ON CONFLICT (customer_id) DO NOTHING"
    results_list = []    
    for row in results:
        results_list.append([row['pairs'][0][0],row['pairs'][0][1], row['score'][0]])
        c.execute(postgres_insert, (row['pairs'][0][0],row['pairs'][0][1], float(row['score'][0])))
    conn.commit()
    c.close()
'''


# save the matched pairs to a table in local postgre
#dtype = {"customer_id": VARCHAR(100), "account_id": VARCHAR(100), "score":Float() }
# psql_insert_copy(results_df,"sfdc_data_map", schema='dedupe',  if_exists = 'append', dtype = dtype ) # ,)


# save to s3 bucket
csv_buffer = StringIO()
results_df.to_csv(csv_buffer)
user1 = boto3.session.Session(profile_name='user1')
s3_resource = user1.resource('s3')
mybucket = '***'
s3_resource.Object(
    mybucket, '***').put(Body=csv_buffer.getvalue())




'''
record_pairs = ((list(messy_data.items())[0:2]),)
print(gazetteer.data_model._field_comparators)
print(gazetteer.data_model.distances(record_pairs=record_pairs))
print(gazetteer.classifier.weights)
gazetteer.blocker.predicates
gazetteer.classifier
for field in gazetteer.blocker.index_fields:
    print(field)

'''


'''
#results = gazetteer.match(messy, threshold=threshold)
results = gazetteer.match(messy_data, threshold=0.01)
print('done matching')

cluster_membership = {}
cluster_id = None
for cluster_id, (row,) in enumerate(results):
    cluster, score = row
    for record_id in cluster:
        cluster_membership[record_id] = (cluster_id, score)

if cluster_id :
    unique_id = cluster_id + 1
else :
    unique_id =0
    
 
# with open(output_file, 'w') as f:
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['customer_1_id', 'customer_1_name', 'customer_1_address', 'customer_1_city',  'customer_1_state', 'socre', 
                    'customer_2_id',  'customer_2_name', 'customer_2_address', 'customer_2_city',  'customer_2_state',  'customer_2_zip' ])

    for cluster_id, (row,) in enumerate(results):
        pair, score = row
        row_out = [pair[0],messy_data[pair[0]]['name'],messy_data[pair[0]]['address'],messy_data[pair[0]]['city']  # , messy_data[pair[0]]['state']   
             ,score
             ,pair[1], canonical_data[pair[1]]['name'], canonical_data[pair[1]]['address'] , canonical_data[pair[1]]['city']  # , canonical_data[pair[1]]['state'], canonical_data[pair[1]]['zip']   
             ]
        writer.writerow(row_out) 

print('ouput file completed')        
           
'''




