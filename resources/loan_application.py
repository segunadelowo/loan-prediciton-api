from flask_restful import Resource, reqparse
from dtos.loan_application import LoanApplicationDto
from sklearn.externals import joblib
from flask import request
import numpy as np

class LoanApplication(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('params', 
        type=str,
        required=True,
        help="Value required."

    )  

    def get(self):
        """Returns the probablity of default for given features.
        """

        if 'gender' in request.args:            
            gender= request.args['gender']
        else:
            return {'message':'gender required'}, 400

        if 'married' in request.args:            
            married= request.args['married']
        else:
            return {'message':'married required'}, 400

        if 'education' in request.args:            
            education= request.args['education']
        else:
            return {'message':'education required'}, 400

        if 'self_employed' in request.args:            
            self_employed= request.args['self_employed']
        else:
            return {'message':'self_employed required'}, 400

        if 'applicant_income' in request.args:            
            applicant_income= request.args['applicant_income']
        else:
            return {'message':'applicant_income required'}, 400    
        
        if 'coapplicant_income' in request.args:            
            coapplicant_income= request.args['coapplicant_income']
        else:
            return {'message':'coapplicant_income required'}, 400

        if 'loan_amount' in request.args:            
            loan_amount= request.args['loan_amount']
        else:
            return {'message':'loan_amount required'}, 400

        if 'loan_amount_term' in request.args:            
            loan_amount_term= request.args['loan_amount_term']
        else:
            return {'message':'loan_amount_term required'}, 400

        if 'credit_history' in request.args:            
            credit_history= request.args['credit_history']
        else:
            return {'message':'credit_history required'}, 400

        if 'property_area' in request.args:            
            property_area= request.args['property_area']
        else:
            return {'message':'property_area required'}, 400


        #gender=male&married=yes&dependents=0&education=Graduate&self_employed=No&applicant_income=5849&coapplicant_income=0.0&loan_amount=128.0&loan_amount_term=360.0&credit_history=1.0&property_area=Urban

        #gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area
        # query=first_name:Bob|age:27

        # read the encoders and the model
        model = joblib.load("../model/LDA.pkl")
        gender_encoder = joblib.load("../model/gender_encoder.pkl")
        married_encoder = joblib.load("../model/married_encoder.pkl")
        education_encoder = joblib.load("../model/education_encoder.pkl")
        self_employed_encoder = joblib.load("../model/self_employed_encoder.pkl")
        property_area_encoder = joblib.load("../model/property_area_encoder.pkl")

        # encoders work on a vector. Wrapping in a list as we only have a single value
        gender_code = gender_encoder.transform([gender])[0]
        married_code = married_encoder.transform([married])[0]
        education_code = education_encoder.transform([education])[0]
        self_employed_code = self_employed_encoder.transform([self_employed])[0]
        property_area_code = property_area_encoder.transform([property_area])[0]
         
         # important to pass the features in the same order as we built the model
        features = [float(gender_code),float(married_code),float(education_code),float(self_employed_code),float(applicant_income),float(coapplicant_income),float(loan_amount),float(loan_amount_term),float(credit_history),float(property_area_code)]
    
        try:
            # probablity for not-defaulting and defaulting
            # Again, wrapping in a list as a list of features is expected
            prob0, prob1 = model.predict_proba([features])[0]
            return prob1
        except Exception as identifier:
            print(identifier)
        
        