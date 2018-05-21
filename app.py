from flask import Flask, request, jsonify
from datetime import datetime
#import foo




# Normal flask app stuff

app = Flask(__name__)


#use the route() decorator to tell Flask what URL should trigger our function
@app.route('/', methods=['POST'])
def homepage():
	payload=request.get_json()
	industry = payload['ind']

	if industry==1:
		task = "Make a Cake"
		job= "Baker"
	else:
		task = "Plant a flower"
		job= "Gardener"

	return jsonify({
 		'task': task,
        'job': job
	})   
 

 #    if request.method == 'POST':
	#     return jsonify({
	#         'prediction': prediction
	#     })       
	# else:
 #        return show_the_login_form()


if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
