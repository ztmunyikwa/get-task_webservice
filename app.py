from flask import Flask, request, jsonify
from datetime import datetime
import pickle
import json
import pandas as pd
import numpy as np
import random 


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

	np.random.seed(22)
	#get_random_industry_num = np.random.randint(0,df.shape[0])

	#randomly select an industry for testing
	#industry = df.get_value(get_random_industry_num, 'NAICS2')
	#use payload to take input on industry
	payload=request.get_json()
	industry = int(float(payload['ind']))


	pieces= [df[(df.NAICS2==industry)], df[df.NAICS2==0], df[df.NAICS2==99]]

	df_all= pd.concat(pieces)



	random.seed( 22 )
	random_dwa =df_all.sample(n=1) 

	dwa_title = random_dwa.iloc[0,3]
	job = random_dwa.iloc[0, 1]

	return jsonify({
 		'task': dwa_title,
        'job': job
	})  





 

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
