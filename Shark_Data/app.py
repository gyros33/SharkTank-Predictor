import os
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
import json
import sys

#Used to run model def
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from string import punctuation
import pickle
from sklearn.externals import joblib

#for local use
# from config import master_username, db_password, endpoint, db_instance_name

#insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, os.path.join('static','py'))
from app.static.py.model import run_model


app = Flask(__name__)



#################################################
# Database Setup
#################################################

#For Web Use
master_username = os.environ['master_username']
db_password = os.environ['db_password']
endpoint = os.environ['endpoint']
db_instance_name = os.environ['db_instance_name']

#Connect to Amazon RDS Postgres database
app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{master_username}:{db_password}@{endpoint}:5432/{db_instance_name}"
db = SQLAlchemy(app)

#reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

# Save references to each table
Shark_Table = Base.classes.Shark_Tank
Pitch_Table = Base.classes.Pitch_Table

#Homepage
@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")

#Gets columns of the shark_tank database (no longer used)
@app.route("/names")
def names():
    """Return a list of sample names."""

    # Use Pandas to perform the sql query
    stmt = db.session.query(Shark_Table).statement
    df = pd.read_sql_query(stmt, db.session.bind)

    # Return a list of the column names (sample names)
    return jsonify(list(df))

#Gets a list of pitches and pitch IDs
@app.route("/pitches")
def pitches():
    """Return a list of sample names."""

    # Use Pandas to perform the sql query
    stmt = db.session.query(Shark_Table).statement
    df = pd.read_sql_query(stmt, db.session.bind)

    data ={
        "Id": df.Id.values.tolist(),
        "Pitch Name": df.Title.values.tolist(),
        "Episode Code": df.Episode_Code.values.tolist()
    }
    # Return a list of the column names (sample names)
    return jsonify(data)

#Gathers information needed to render the D3 Graph on the shark page
#this needs improvement to help the filters work correctly. Should psot data based on selected filters
@app.route("/sharks")
def sharks():
    """Return a list of sample names."""

    results = db.session.query(Shark_Table).all()

    data = []
    for result in results:

        #If filter is used, only display pitches where deals were made
        if(result.Deal_Status == "Deal Made"):
            data.append({
                "id": result.Id,
                "stake": result.Exchange_For_Stake,
                "ask": result.Amount_Asked_For,
                "valuation": result.Valuation,
                "deal": result.Deal_Status,
                "dealshark1": result.Deal_Shark_1,
                "dealshark2": result.Deal_Shark_2,
                "dealshark3": result.Deal_Shark_3,
                "dealshark4": result.Deal_Shark_4,
                "dealshark5": result.Deal_Shark_5
            })

        #Otherwise
        else:
            data.append({
                "id": result.Id,
                "stake": result.Exchange_For_Stake,
                "ask": result.Amount_Asked_For,
                "valuation": result.Valuation,
                "deal": result.Deal_Status,
                "dealshark1": result.Shark_1,
                "dealshark2": result.Shark_2,
                "dealshark3": result.Shark_3,
                "dealshark4": result.Shark_4,
                "dealshark5": result.Shark_5
            })

    # Return a list of the column names (sample names)
    return jsonify(data)

#Renders the sharkpage
@app.route("/sharkpage")
def sharkpage():
    return render_template("shark.html")

#Renders the map page
@app.route('/map')
def map():
    return render_template('loc.html')

#This route renders the pitch predictor
@app.route('/funpage', methods=['POST','GET'])
def funpage():
    #Predifine categories. Consider doing this programattically from the Shark_Table
    cats = ['Health / Wellness', 'Lifestyle / Home', 'Software / Tech','Food and Beverage', 'Business Services','Fashion / Beauty', 'Automotive', 'Media / Entertainment','Fitness / Sports / Outdoor', 'Pet Products', 'Green / Clean Tech', 'Travel', 'Children / Education', 'Uncertain / Other']
    
    #Populate the list of previously submitted pitches from teh database
    dbresults = db.session.query(Pitch_Table.Title).all()
    titles = [x[0] for x in dbresults]
    img = 0

    #When data is submitted to the predictor, gather this data
    if request.method == 'POST':
        result = request.form
        input_title = result["title"]
        input_pitch = [result["pitch"]]
        input_amount = int(result["ask"])
        input_exchange = int(result["stake"])
        input_valuation = int(input_amount / (input_exchange / 100))
        input_gender = result["gen"]
        input_category = result["cat"]
        
        #Makes sure that the title is unique for the database, otherwise they will get an error from the js file
        if input_title not in titles:
            new = Pitch_Table(Title=input_title, Category=input_category, Amount_Asked_For=input_amount, Exchange_For_Stake=input_exchange, Valuation=input_valuation, Description=input_pitch[0])
            db.session.add(new)
            db.session.commit()

        #Runs the MultinomialNB trained with the data from shark tank
        x, y = run_model(input_pitch, input_amount, (input_exchange / 100), input_gender, input_category)
        
        #X signifies that the model predicts Deal_Status to be 0
        if x == 0:
            deal_status = "Sorry, I'm out"
            deal_shark = "sad.png"
        #If the deal status is 1, the model also predicts which shark would pick it up
        else:
            deal_status = "You've got a deal!"
            if y[0] == "Barbara Corcoran":
                deal_shark = "barb2.png"
            elif y[0] == "Mark Cuban":
                deal_shark = "cuban1.png"
            elif y[0] == "Lori Greiner":
                deal_shark = "lori1.png"
            elif y[0] == "Robert Herjavec":
                deal_shark = "rob1.png"
            elif y[0] == "Daymond John":
                deal_shark = "daymond1.png"
            else:
                deal_shark = "kevin1.png"
        
        img = 1
        
        return render_template('fun.html', img=img, input_title=input_title, input_pitch=result["pitch"], input_amount=input_amount, input_exchange=input_exchange, input_valuation=input_valuation, input_gender=input_gender, input_category=input_category, deal_status=deal_status, deal_shark=deal_shark, titles=titles, cats=cats)

    return render_template('fun.html', img=img, cats=cats, titles=titles)

#pulls all the user pitches stored in the database
@app.route('/userpitches')
def userpitches():
    results = db.session.query(Pitch_Table).all()

    inputs = []
    for result in results:
        inputs.append({
            "id": result.id,
            "title": result.Title,
            "category": result.Category,
            "ask": result.Amount_Asked_For,
            "exchange": result.Exchange_For_Stake,
            "valuation": result.Valuation,
            "description": result.Description
        })
    return jsonify(inputs)

#Will pull user created pitches in order to populate the pitch predictor tool with already submitted pitches
@app.route('/userpitches/<title>')
def specific_pitch(title):
    results = db.session.query(Pitch_Table).filter(Pitch_Table.Title == title).all()

    inputs = []
    for result in results:
        inputs.append({
            "id": result.id,
            "title": result.Title,
            "category": result.Category,
            "ask": result.Amount_Asked_For,
            "exchange": result.Exchange_For_Stake,
            "valuation": result.Valuation,
            "description": result.Description
        })
    return jsonify(inputs)

#Needs to be reworked to dynamically call data to generate sunburst plot
@app.route('/pitchpage')
def pitchpage():
    return render_template('pitch.html')


if __name__ == "__main__":
    app.run(debug=True)
