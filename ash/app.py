from flask import Flask, request, jsonify
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model_pickle = open("/Users/91789/Documents/GitHub/Flaskpractice/ash/classifier.pkl", "rb")
clf = pickle.load(model_pickle)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/ping",methods=["GET"])
def ping():
    return {"message": "Hi there, I'm working!!"}
# defining the endpoint which will make the prediction
@app.route("/prediction", methods=['POST'])
def prediction():
    """ Returns loan application status using ML model
    """
    try:
        loan_req = request.get_json()
        if not loan_req:
            return jsonify({"error": "Request must be JSON"}), 400

        # Print the request data for debugging
        print(f"Received request data: {loan_req}")

        # Validate required fields
        required_fields = ['Gender', 'Married', 'Credit_History', 'ApplicantIncome', 'LoanAmount']
        for field in required_fields:
            if field not in loan_req:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Process the input data
        Gender = 0 if loan_req['Gender'] == "Male" else 1
        Married = 0 if loan_req['Married'] == "Unmarried" else 1
        Credit_History = 0 if loan_req['Credit_History'] == "Unclear Debts" else 1
        ApplicantIncome = loan_req['ApplicantIncome']
        LoanAmount = loan_req['LoanAmount']

        result = clf.predict([[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])

        pred = "Rejected" if result == 0 else "Approved"

        return jsonify({"loan_approval_status": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)