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
    dwa_id = TextField()
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
	df = pd.read_csv('occsDWAsIndustries_full_clean.csv', sep=',')
	
	#use payload to take input on industry
	payload=request.get_json()
	try:
		industry = int(float(payload['ind']))
	except ValueError:
		industry = payload['ind']
		print payload['ind']
	user_id_qualtrics= payload['userid']


	task_set = False

	while task_set==False:

		
		try:
			do_filter = payload['filter_on']
		except KeyError:
			do_filter = "False"
		

		if do_filter=="True":
			df_all = df[(df.NAICS2==industry)&(df.rated==True)]
		else:
			df_all= df[(df.NAICS2==industry)]



		random_dwa =df_all.sample(n=1) 
		dwa_title = random_dwa.iloc[0,3]
		dwa_id = random_dwa.iloc[0,2]
		job = random_dwa.iloc[0, 1]


		query_usr_done = AssignedTask.select().where(AssignedTask.user_id == user_id_qualtrics, AssignedTask.dwa == dwa_title).count()
		dwa_times_done = AssignedTask.select().where(AssignedTask.dwa == dwa_title, AssignedTask.dwa_id==dwa_id).count()

		#if dwa has already been assigned assigned to the user, redo the loop
		if query_usr_done > 0:
			continue

		#if the dwa has already been assigned ten times, then redo the loop
		elif dwa_times_done >= 10: 
			continue

		#else, record this assignment, and return the assigned task and job 
		else:
			task_set=True
			q = AssignedTask(
		        user_id=user_id_qualtrics,
		        dwa=dwa_title,
		        dwa_id = dwa_id,
		        job=job,
		        industry=industry
		    )
			q.save()

			return jsonify({
		 		'task': dwa_title,
		        'job': job			})  
# End webserver stuff
########################################






 

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
