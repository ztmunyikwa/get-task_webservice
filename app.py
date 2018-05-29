from flask import Flask, request, jsonify
from datetime import datetime
import pickle
import json
import pandas as pd
import os
import numpy as np
import random 
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, BooleanField, TextField, DateTimeField
)
import datetime



####Begin Database stuff
if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('tasks_assigned.db')



class AssignedTask(Model):
    user_id= TextField()
    dwa = TextField()
    industry = IntegerField()
    job = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = DB

DB.create_tables([AssignedTask], safe=True)   ##safe: If set to True, the create table query will include an IF NOT EXISTS clause.


####End database stuff
########################################



########################################
# Normal flask app stuff

app = Flask(__name__)


#use the route() decorator to tell Flask what URL should trigger our function
@app.route('/', methods=['POST'])
def homepage():	
	return jsonify({
		'message': 'This is my homepage'
	}) 



@app.route('/gettask', methods=['POST'])
def gettask():
	df = pd.read_csv('occsDWAsIndustries_full.csv', sep=',')
	##replace recurring industries with one code
	df=df.replace({'NAICS2': 32}, 31)
	df=df.replace({'NAICS2': 33}, 31)
	df=df.replace({'NAICS2': 45}, 44)
	df=df.replace({'NAICS2': 49}, 48)

	#np.random.seed(22)

	#get_random_industry_num = np.random.randint(0,df.shape[0])

	#randomly select an industry for testing
	#industry = df.get_value(get_random_industry_num, 'NAICS2')
	#use payload to take input on industry
	payload=request.get_json()
	industry = int(float(payload['ind']))


	pieces= [df[(df.NAICS2==industry)], df[df.NAICS2==0], df[df.NAICS2==99]]

	df_all= pd.concat(pieces)

	task_set = False

	while task_set==False:
		#use payload to take input on industry
		payload=request.get_json()
		industry = int(float(payload['ind']))
		user_id_qualtrics= payload['userid']

		pieces= [df[(df.NAICS2==industry)], df[df.NAICS2==0], df[df.NAICS2==99]]
		df_all= pd.concat(pieces)



		#np.random.seed( 22 )
		random_dwa =df_all.sample(n=1) 
		dwa_title = random_dwa.iloc[0,3]
		job = random_dwa.iloc[0, 1]

		#if user_id, dwa has already been assigned, redo the loop
		query=AssignedTask.select().where(AssignedTask.user_id == user_id_qualtrics, AssignedTask.dwa == dwa_title)
		if len(query) > 0:
			print(query)
			continue
		else:
			task_set=True
			q = AssignedTask(
		        user_id=user_id_qualtrics,
		        dwa=dwa_title,
		        job=job,
		        industry=industry
		    )
			q.save()

			return jsonify({
		 		'task': dwa_title,
		        'job': job
			})  
# End webserver stuff
########################################






 

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)