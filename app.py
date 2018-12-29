from flask import Flask
from flask_restful import Api
from flask import jsonify
import numpy as np
from sklearn.externals import joblib

from resources.loan_application import LoanApplication


app = Flask(__name__)
api = Api(app)


api.add_resource(LoanApplication, '/api/v1/loanapplications/predict') #GET,POST


if __name__ == '__main__': # only run on startup, if this file is called by another file dont run code below
    app.run(port=5001, debug=True)


