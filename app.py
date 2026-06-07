# =============================================================================
# app.py  –  Flask Web Interface for Project B6
# Run: python app.py   →   open http://127.0.0.1:5000
# =============================================================================

from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Load Artefacts ───────────────────────────────────────────────────────────
print("Loading saved models and artefacts...")
scaler         = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
feature_names  = joblib.load('models/feature_names.pkl')

MODELS = {
    'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
    'Decision Tree'      : joblib.load('models/decision_tree.pkl'),
    'Random Forest'      : joblib.load('models/random_forest.pkl'),
    'SVM'                : joblib.load('models/svm.pkl'),
    'KNN'                : joblib.load('models/knn.pkl'),
}
NEEDS_SCALING = {'Logistic Regression', 'SVM', 'KNN'}

# Load metrics summary from CSV
metrics_df = pd.read_csv('plots/metrics_summary.csv', index_col=0)
METRICS_DATA = metrics_df.to_dict(orient='index')
print("All artefacts loaded successfully.")

# ── Categorical option lists ─────────────────────────────────────────────────
WORKCLASS_OPTS   = ['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov']
EDUCATION_OPTS   = ['Bachelors','Some-college','HS-grad','Masters','Doctorate','Prof-school','Assoc-acdm','Assoc-voc','11th','12th','10th','9th','5th-6th']
MARITAL_OPTS     = ['Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent']
OCCUPATION_OPTS  = ['Tech-support','Craft-repair','Sales','Exec-managerial','Prof-specialty','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Protective-serv']
RELATIONSHIP_OPTS= ['Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried']
RACE_OPTS        = ['White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black']
SEX_OPTS         = ['Male','Female']
COUNTRY_OPTS     = ['United-States','Other']


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html',
                           metrics=METRICS_DATA,
                           model_names=list(MODELS.keys()))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page."""
    form_data = {}
    result = None

    options = dict(
        workclass=WORKCLASS_OPTS, education=EDUCATION_OPTS,
        marital_status=MARITAL_OPTS, occupation=OCCUPATION_OPTS,
        relationship=RELATIONSHIP_OPTS, race=RACE_OPTS,
        sex=SEX_OPTS, native_country=COUNTRY_OPTS,
        models=list(MODELS.keys())
    )

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            chosen_model = form_data.get('model_choice', 'Random Forest')

            # Build raw input dict
            raw = {
                'age'            : int(form_data['age']),
                'workclass'      : form_data['workclass'],
                'education'      : form_data['education'],
                'education_num'  : int(form_data['education_num']),
                'marital_status' : form_data['marital_status'],
                'occupation'     : form_data['occupation'],
                'relationship'   : form_data['relationship'],
                'race'           : form_data['race'],
                'sex'            : form_data['sex'],
                'capital_gain'   : int(form_data['capital_gain']),
                'capital_loss'   : int(form_data['capital_loss']),
                'hours_per_week' : int(form_data['hours_per_week']),
                'native_country' : form_data['native_country'],
            }

            # Encode categoricals using saved LabelEncoders
            encoded = {}
            cat_cols = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']
            for col in feature_names:
                if col in cat_cols:
                    le = label_encoders[col]
                    val = raw[col]
                    if val in le.classes_:
                        encoded[col] = int(le.transform([val])[0])
                    else:
                        # unseen label → map to 0
                        encoded[col] = 0
                else:
                    encoded[col] = raw[col]

            X_input = np.array([[encoded[f] for f in feature_names]])

            # Predict with chosen model
            clf = MODELS[chosen_model]
            if chosen_model in NEEDS_SCALING:
                X_input_proc = scaler.transform(X_input)
            else:
                X_input_proc = X_input

            pred   = clf.predict(X_input_proc)[0]
            proba  = clf.predict_proba(X_input_proc)[0]

            # Also predict with all models for comparison
            all_preds = {}
            for mname, mclf in MODELS.items():
                Xi = scaler.transform(X_input) if mname in NEEDS_SCALING else X_input
                p  = mclf.predict_proba(Xi)[0]
                all_preds[mname] = {
                    'prediction': '>50K' if mclf.predict(Xi)[0] == 1 else '<=50K',
                    'prob_high' : round(float(p[1]) * 100, 1),
                    'prob_low'  : round(float(p[0]) * 100, 1),
                }

            result = {
                'prediction'  : '>50K' if pred == 1 else '<=50K',
                'prob_high'   : round(float(proba[1]) * 100, 1),
                'prob_low'    : round(float(proba[0]) * 100, 1),
                'chosen_model': chosen_model,
                'all_preds'   : all_preds,
            }

        except Exception as e:
            result = {'error': str(e)}

    return render_template('predict.html',
                           options=options,
                           form_data=form_data,
                           result=result)


@app.route('/metrics')
def metrics_page():
    """Metrics comparison page with interactive charts."""
    return render_template('metrics.html',
                           metrics=METRICS_DATA,
                           model_names=list(MODELS.keys()))


@app.route('/plots')
def plots_page():
    """Gallery of all saved matplotlib plots."""
    plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
    return render_template('plots.html', plot_files=plot_files)


@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve plot images."""
    return send_from_directory('plots', filename)


@app.route('/api/metrics')
def api_metrics():
    """JSON endpoint for chart data."""
    return jsonify(METRICS_DATA)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Flask server starting...")
    print("  Open browser at:  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, use_reloader=False)
