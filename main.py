import json

from flask import Flask, render_template, request, redirect, session, jsonify
import pymysql
app = Flask(__name__)
app.secret_key="symptiom"
conn = pymysql.connect(host="localhost",db="symptomChecker",password="root",user="root")
cursor = conn.cursor()
import pickle
import pandas as pd
import numpy as np
import requests

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/userLogin")
def userLogin():
    return render_template("userLogin.html")

@app.route("/userReg")
def userReg():
    return render_template("userReg.html")


@app.route("/userRegAction",methods=['post'])
def userRegAction():
    name = request.form.get("name")
    email = request.form.get("email")
    gender = request.form.get("gender")
    location = request.form.get("location")
    age = request.form.get("age")
    password = request.form.get("password")
    cursor.execute("insert into users(name,email,gender,location,age,password) values ('"+str(name)+"','"+str(email)+"','"+str(gender)+"','"+str(location)+"','"+str(age)+"','"+str(password)+"')")
    conn.commit()
    return redirect("/userLogin")


@app.route("/userLoginAction",methods=['post'])
def userLoginAction():
    email = request.form.get("email")
    password = request.form.get("password")
    count = cursor.execute("select * from users where email='"+str(email)+"' and password = '"+str(password)+"'")
    if count > 0:
        user = cursor.fetchone()
        session['role'] = 'user'
        session['userId'] = user[0]
        return redirect("/userHome")
    else:
        return redirect("/userLogin")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")



@app.route("/userHome")
def userHome():
    cursor.execute("select * from users where userId='"+str(session['userId'])+"'")
    user = cursor.fetchone()
    return render_template("userHome.html",user=user)

@app.route("/userBodyMap",methods=['post'])
def userBodyMap():
    age = request.form.get("age")
    gender = request.form.get("gender")
    print(age)
    print(gender)
    message = request.form.get("message")
    return render_template("userBodyMap.html",gender=gender,age=age,message=message)


@app.route("/submit_health_details")
def submit_health_details():
    health_report = request.args.get("health_report")
    bp_report = request.args.get("bp_report")
    temperature = request.args.get("temperature")
    userId = session['userId']
    cursor.execute("insert into health_details(health_report,bp_report,temperature,userId) values ('" + str(health_report) + "','" + str(
        bp_report) + "','" + str(temperature) + "','" + str(userId) + "')")
    conn.commit()
    return render_template("message.html",message="Health Details Updated Successfully")
from joblib import dump, load

@app.route("/disease_prediction",methods=['post'])
def disease_prediction():
    age = request.form.get("age")
    print(age)
    gender= request.form.get("gender")
    muscle_pain = 1 if request.form.get("muscle_pain") else 0
    itching = 1 if request.form.get('itching') else 0
    altered_sensorium = 1 if request.form.get('altered_sensorium') else 0
    dark_urine = 1 if request.form.get('dark_urine') else 0
    high_fever = 1 if request.form.get('high_fever') else 0
    mild_fever = 1 if request.form.get('mild_fever') else 0
    family_history = 1 if request.form.get('family_history') else 0
    nausea = 1 if request.form.get('nausea') else 0
    yellowing_of_eyes = 1 if request.form.get('yellowing_of_eyes') else 0
    sweating = 1 if request.form.get('sweating') else 0
    unsteadiness = 1 if request.form.get('unsteadiness') else 0
    chest_pain = 1 if request.form.get('chest_pain') else 0
    fatigue = 1 if request.form.get('fatigue') else 0
    abdominal_pain = 1 if request.form.get('abdominal_pain') else 0
    joint_pain = 1 if request.form.get('joint_pain') else 0
    diarrhoea = 1 if request.form.get('diarrhoea') else 0
    lack_of_concentration = 1 if request.form.get('lack_of_concentration') else 0
    red_spots_over_body = 1 if request.form.get('red_spots_over_body') else 0
    loss_of_appetite = 1 if request.form.get('loss_of_appetite') else 0
    vomiting = 1 if request.form.get('vomiting') else 0
    symptoms = np.array([[
        muscle_pain,
        itching,
        altered_sensorium,
        dark_urine,
        high_fever,
        mild_fever,
        family_history,
        nausea,
        yellowing_of_eyes,
        sweating,
        unsteadiness,
        chest_pain,
        fatigue,
        abdominal_pain,
        joint_pain,
        diarrhoea,
        lack_of_concentration,
        red_spots_over_body,
        loss_of_appetite,
        vomiting
    ]])
    with open("ML_models/random_forest_rfe_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    disease_chance = model.predict_proba(symptoms)
    disease_chance_score = np.max(disease_chance) * 100
    result = model.predict(symptoms)

    print('predicted disease:', result)
    symptom_list = [
        "muscle_pain",
        "itching",
        "altered_sensorium",
        "dark_urine",
        "high_fever",
        "mild_fever",
        "family_history",
        "nausea",
        "yellowing_of_eyes",
        "sweating",
        "unsteadiness",
        "chest_pain",
        "fatigue",
        "abdominal_pain",
        "joint_pain",
        "diarrhoea",
        "lack_of_concentration",
        "red_spots_over_body",
        "loss_of_appetite",
        "vomiting"
    ]

    real_symptoms = np.array([symptom_list])[symptoms == 1]

    symptoms = symptoms.tolist()[0]  # Convert to Python list

    real_symptoms = [symptom_list[i] for i in range(len(symptom_list)) if symptoms[i] == 1]
    print(len(real_symptoms))
    if len(real_symptoms)<=1:
        return render_template("userBodyMap.html", age=age, gender=gender,
                               message="Select Atleast Two or more symptoms!")
    else:
        cursor.execute("select * from users where userId='"+str(session['userId'])+"'")
        user = cursor.fetchone()
        disease_chance_score = round(disease_chance_score)
        data = {
            "Symptom": ["muscle_pain", "itching", "altered_sensorium", "dark_urine", "high_fever",
                        "mild_fever", "family_history", "nausea", "yellowing_of_eyes", "sweating",
                        "unsteadiness", "chest_pain", "fatigue", "abdominal_pain", "joint_pain",
                        "diarrhoea", "lack_of_concentration", "red_spots_over_body", "loss_of_appetite", "vomiting"],
            "Importance": [0.019439, 0.016014, 0.016001, 0.015811, 0.015623,
                           0.015100, 0.014786, 0.014670, 0.014079, 0.013823,
                           0.013589, 0.013119, 0.012688, 0.012670, 0.012629,
                           0.012459, 0.011938, 0.011681, 0.011402, 0.011396]
        }
        df = pd.DataFrame(data)
        df_selected = df[df["Symptom"].isin(real_symptoms)].copy()
        total_strength = df_selected["Importance"].sum()
        df_selected["Strength (%)"] = (df_selected["Importance"] / total_strength) * 100
        print("hh")
        print(df_selected['Strength (%)'].sum())
        symptoms_strength = df_selected['Strength (%)'].sum()
        if symptoms_strength < 50:
            strength_category = "Low"
        elif 50 <= symptoms_strength < 75:
            strength_category = "Moderate"
        elif 75 <= symptoms_strength < 90:
            strength_category = "High"
        else:
            strength_category ="Strong"
        print(strength_category)
        return render_template("prediction.html",strength_category=strength_category,symptoms_strength=symptoms_strength,real_symptoms=real_symptoms,result=result[0],age=age,gender=gender,user=user,disease_chance_score=disease_chance_score)

@app.route("/medicines",methods=['post'])
def medicines():
    real_symptoms = request.form.getlist('symptoms')
    result = request.form.get("result")
    print(real_symptoms)
    print('symptoms list in medicines.........', real_symptoms)
    url = f'https://api.fda.gov/drug/label.json?search=indications_and_usage:{result}&limit=5'

    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Create a list to store the extracted data
    records = []

    # Extract relevant information
    for i in range(len(response.json().get("results"))):
        drug_name = response.json().get("results")[i].get('openfda').get('generic_name')
        product_type = response.json().get("results")[i].get('openfda').get('product_type')
        brand_name = response.json().get("results")[i].get('openfda').get('brand_name')
        form_of_taking = response.json().get("results")[i].get('openfda').get('route')
        if drug_name is None:
            continue
        records.append([drug_name, product_type, brand_name, form_of_taking])

    data = pd.DataFrame(records,columns=['drug_name','product_type','brand_name','form_of_taking'])
    print(data)
    user_id = session['userId']
    cursor.execute("select * from users where userId='" + str(user_id) + "'")
    user = cursor.fetchall()[0]
    age = user[4]
    gender = user[5]
    return render_template("medicines.html",user=user,result=result,df=data)


# @app.route("/viewDrugDetails")
# def viewDrugDetails():
#     drugName = request.args.get("drugName")
#     result = request.args.get("result")
#     # API URL without a limit
#     url = f'https://api.fda.gov/drug/label.json?search=indications_and_usage:{drugName}'
#     # Fetch the data
#     response = requests.get(url)
#     # Check if the request was successful
#     results2 = []
#     if response.status_code == 200:
#         data = response.json()  # Required keys
#         keys_to_display = ['dosage_and_administration','adverse_reactions','dosage_forms_and_strengths']
#         # keys_to_display =['dosage_and_administration']
#         # dosage_and_administration =data['results'][0]["dosage_and_administration"]
#         # print(dosage_and_administration[:50])
#         # Display data for the required keys
#         for key in keys_to_display:
#             print("vv")
#             print(data['results'][0].get(key))
#             results2.append(data['results'][0].get(key))
#     cursor.execute("select * from users where userId='"+str(session['userId'])+"'")
#     user = cursor.fetchone()
#     print(results2[0][0][:200])
#     return render_template("viewDrugDetails.html",result=result,user=user,results2=results2,drugName=drugName)


# @app.route("/viewDrugDetails")
# def viewDrugDetails():
#     drugName = request.args.get("drugName")
#     result = request.args.get("result")
#
#     # API URL without a limit
#     url = f'https://api.fda.gov/drug/label.json?search=indications_and_usage:{drugName}'
#
#     # Fetch the data
#     response = requests.get(url)
#
#     # Check if the request was successful
#     results2 = []
#     if response.status_code == 200:
#         data = response.json()  # Required keys
#         keys_to_display = ['dosage_and_administration', 'adverse_reactions', 'dosage_forms_and_strengths']
#
#         # Display data for the required keys
#         for key in keys_to_display:
#             content = data['results'][0].get(key, [""])  # Ensure it's a list
#
#             if content and isinstance(content, list):
#                 formatted_content = []
#                 for text in content:
#                     split_text = text.split('.')  # Split by period
#                     sentences = [s.strip() for s in split_text if s.strip()]  # Remove empty spaces
#
#                     # Group sentences into chunks of two
#                     grouped_sentences = ['. '.join(sentences[i:i + 2]) + '.' for i in range(0, len(sentences), 2)]
#
#                     formatted_content.extend(grouped_sentences)
#
#                 results2.append(formatted_content)
#             else:
#                 results2.append(["No data available"])
#
#     cursor.execute("select * from users where userId='" + str(session['userId']) + "'")
#     user = cursor.fetchone()
#
#     return render_template("viewDrugDetails.html", result=result, user=user, results2=results2, drugName=drugName)
@app.route("/viewDrugDetails")
def viewDrugDetails():
    drugName = request.args.get("drugName")
    result = request.args.get("result")

    # API URL without a limit
    url = f'https://api.fda.gov/drug/label.json?search=indications_and_usage:{drugName}'

    # Fetch the data
    response = requests.get(url)

    # Check if the request was successful
    results2 = []
    key_mappings = {
        'dosage_and_administration': 'Dosage and Administration',
        'adverse_reactions': 'Adverse Reactions',
    }

    if response.status_code == 200:
        data = response.json()  # Required keys
        keys_to_display = list(key_mappings.keys())

        # Process and format the data
        for key in keys_to_display:
            content = data['results'][0].get(key, [""])  # Ensure it's a list

            if content and isinstance(content, list):
                formatted_content = []
                for text in content:
                    split_text = text.split('. ')  # Split by period and space
                    sentences = [s.strip() for s in split_text if s.strip()]  # Remove empty spaces

                    # Extract only the first two sentences
                    selected_text = '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else '. '.join(sentences) + '.'

                    formatted_content.append(selected_text)

                # Append as a tuple with heading
                results2.append((key_mappings[key], formatted_content))  # Heading + Content
            else:
                results2.append((key_mappings[key], ["No data available"]))

    cursor.execute("SELECT * FROM users WHERE userId=%s", (session['userId'],))
    user = cursor.fetchone()

    return render_template("viewDrugDetails.html",result=result,user=user,results2=results2,drugName=drugName)

app.run(debug=True)