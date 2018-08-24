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
    FloatField, BooleanField, TextField, DateTimeField,fn
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
    verified_complete = BooleanField(default=False)

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
	

	payload=request.get_json()

	#set user id
	user_id_qualtrics= payload['userid']
	#set industry
	try:
		industry = int(float(payload['ind']))
	except ValueError:
		industry = random.choice([11,21,22,23,31,42,44,48,51,52,53,54,55,56,61,62,71,81])

	#set the filter status (default is off)
	try:
		do_filter = payload['filter_on']
	except KeyError:
		do_filter = "False"

	if do_filter=="True":
		df_all = df[(df.NAICS2==industry)&(df.rated==True)]
	else:
		df_all= df[(df.NAICS2==industry)]


	task_set = False
	tries = 0 

	while task_set==False:

		if tries!=5:
			random_dwa =df_all.sample(n=1) 
			dwa_title = random_dwa.iloc[0,3]
			dwa_id = random_dwa.iloc[0,2]
			job = random_dwa.iloc[0, 1]

		elif tries==5:	
			#get a list of the dwas that the user has already rated
			query_usr_done= AssignedTask.select().where(AssignedTask.user_id == user_id_qualtrics)
			dwas_ratedby_usr = [rating.dwa for rating in query_usr_done]
			#get a list of the dwas that have already been rated ten times
			query_ten_done= AssignedTask.select().group_by(AssignedTask).having(fn.Count(AssignedTask.dwa) >= 10)
			dwas_ratedten = [rating.dwa for rating in query_ten_done]

			#filter these dwas out of the df_all list 
			dwas_to_filter =  list(set(dwas_ratedby_usr) | set(dwas_ratedten))
			df_selectfrom= df_all[~df_all['DWA Title'].isin(dwas_to_filter)]

			if df_selectfrom.empty:
				df_selectfrom= df[~df['DWA Title'].isin(dwas_to_filter)]

			#if industry subset is empty, randomly select from the second database 
			random_dwa =df_selectfrom.sample(n=1) 
			dwa_title = random_dwa.iloc[0,3]
			dwa_id = random_dwa.iloc[0,2]
			job = random_dwa.iloc[0, 1]





		query_usr_done_count = AssignedTask.select().where(AssignedTask.user_id == user_id_qualtrics, AssignedTask.dwa == dwa_title).count()  ###question for dan...chill to not check for verification here?
		dwa_times_done_count = AssignedTask.select().where(AssignedTask.dwa == dwa_title, AssignedTask.dwa_id==dwa_id, AssignedTask.verified_complete==True).count()





		#if dwa has already been assigned assigned to the user, redo the loop
		if query_usr_done_count > 0:
			tries = tries +1 
			continue

		#if the dwa has already been assigned ten times, then redo the loop
		elif dwa_times_done_count >= 10: 
			tries = tries +1 
			continue

		#else, record this assignment, and return the assigned task and job 
		else:
			task_set=True
			q = AssignedTask(
		        user_id=user_id_qualtrics,
		        dwa=dwa_title,
		        dwa_id = dwa_id,
		        job=job,
		        industry=industry,
		        verified_complete=False
		    )
			q.save()

			return jsonify({
		 		'task': dwa_title,
		        'job': job			
		    })  





@app.route('/verifytask', methods=['POST'])
def verifytask():
	payload=request.get_json()

	user_id_qualtrics= payload['userid']

	try:
		task1 = payload['task1']
		task2 = payload['task2']
		task3 = payload['task3']


	except KeyError:
		return jsonify ({
		'success':False
		})
	

	q = AssignedTask.update(verified_complete=True).where(AssignedTask.user_id==user_id_qualtrics, AssignedTask.dwa==task1)
	q.execute()
	q = AssignedTask.update(verified_complete=True).where(AssignedTask.user_id==user_id_qualtrics, AssignedTask.dwa==task2)
	q.execute()
	q = AssignedTask.update(verified_complete=True).where(AssignedTask.user_id==user_id_qualtrics, AssignedTask.dwa==task3)
	q.execute()

	return jsonify ({
	'success':True
	})

# End webserver stuff
########################################






 

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
