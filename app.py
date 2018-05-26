from flask import Flask, request, jsonify
from datetime import datetime
import pickle
import json
import pandas as pd
import numpy as np
import random 
import datetime

#This will allow us to use a local database called sqlite when we 
#are developing on our laptops and use a more production-ready database 
#called postgresql when deploying to heroku with very little change to our code.

from peewee import (
    SqliteDatabase, Model, DateTimeField, IntegerField, FloatField,
    BooleanField, TextField,
)





########################################
# Begin database stuff
#Create a sqlite databse that will be stored in a file called tasks_assigned.db
DB = SqliteDatabase('tasks_assigned.db')


class AssignedTask(Model):
	#because none of the fields are initialized with primary_key=True, 
	#an auto-incrementing primary key will automatically be created and named “id”.

	user_id = TextField()
	dwa = TextField()
	industry = FloatField()
	job=TextField()
	created_date = DateTimeField(default=datetime.datetime.now)

	class Meta:
		database = DB


DB.create_tables([AssignedTask], safe=True)

# End database stuff
########################################

########################################




# Normal flask app stuff
########################################
# Begin webserver stuff
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

	np.random.seed(22)

	#use payload to take input on industry
	payload=request.get_json()
	industry = int(float(payload['ind']))
	user_agent= payload['useragent']

	pieces= [df[(df.NAICS2==industry)], df[df.NAICS2==0], df[df.NAICS2==99]]
	df_all= pd.concat(pieces)



	np.random.seed( 22 )
	random_dwa =df_all.sample(n=1) 

	dwa_title = random_dwa.iloc[0,3]
	job = random_dwa.iloc[0, 1]

	q = AssignedTask(
        user_id=user_agent,
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
