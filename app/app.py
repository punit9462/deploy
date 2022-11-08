import pickle
from flask import Flask
from flask import request
app = Flask(__name__)


# Sample: http://127.0.0.1/run_ml?monthly_charges=90&internet_service_fiber_optic=0&internet_service_dsl=1&contract_month_to_month=1&contract_one_year=0&time_stamp_jan15_mar15=1
@app.route('/run_ml', methods=['GET'])
def hello_world():
    monthly_charges = float(request.args.get('monthly_charges'))
    internet_service_fiber_optic = float(request.args.get('internet_service_fiber_optic'))
    internet_service_dsl = float(request.args.get('internet_service_dsl'))
    contract_month_to_month = float(request.args.get('contract_month_to_month'))
    contract_one_year = float(request.args.get('contract_one_year'))
    time_stamp_jan15_mar15 = float(request.args.get('time_stamp_jan15_mar15'))

    mean_dict_val = {
        'monthly_charges': 70.19233412382701,
        'internet_service_fiber_optic': 0.4467436760505916,
        'internet_service_dsl': 0.3482889636882905,
        'contract_month_to_month': 0.3160189718482252,
        'contract_one_year': 0.26949459404324766,
        'time_stamp_jan15_mar15': 0.08965728274173806
    }

    var_dict_val = {
        'monthly_charges': 971.6231481205622,
        'internet_service_fiber_optic': 0.24716376395939563,
        'internet_service_dsl': 0.22698376146122726,
        'contract_month_to_month': 0.21615098128021595,
        'contract_one_year': 0.19686725782471284,
        'time_stamp_jan15_mar15': 0.0816188543931061
    }

    def output_label_to_tenure_churn_mapping(out_label):
        mapping = {
            0: (0, 'No'),
            1: (1, 'No'),
            2: (2, 'No'),
            3: (3, 'No'),
            4: (0, 'Yes'),
            5: (1, 'Yes'),
            6: (2, 'Yes'),
            7: (3, 'Yes'),
        }
        return mapping[out_label]

    def predict(inp, loaded_model):
        norm_inp = {}
        for k in inp.keys():
            norm_inp[k] = (inp[k] - mean_dict_val[k]) / var_dict_val[k]
        model_inp = list(norm_inp.values())

        out_label = loaded_model.predict([model_inp])[0]
        tenure, churn = output_label_to_tenure_churn_mapping(out_label)
        return tenure, churn

    inp = {  # high monthly charges = chrun / low_monthly_charges = no_chrun
        "monthly_charges": monthly_charges,                             #90.0,  # 29.85, # 60.0
        "internet_service_fiber_optic": internet_service_fiber_optic,   #0.0,
        "internet_service_dsl": internet_service_dsl,                   #1.0,
        "contract_month_to_month": contract_month_to_month,             #1.0,
        "contract_one_year": contract_one_year,                         #0.0,
        "time_stamp_jan15_mar15": time_stamp_jan15_mar15,               #1.0
    }

    loaded_model = pickle.load(open('customer_churn_model', 'rb'))
    inference_out = predict(inp, loaded_model)
    # list(inference_out.values())

    out_response = {
        "tenure (months in this quarter)": inference_out[0],
        "churn": inference_out[1]
    }
    return out_response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
