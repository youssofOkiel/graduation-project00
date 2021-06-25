from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import pickle
import spacy

import pandas as pd
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer

from difflib import SequenceMatcher

from sklearn.preprocessing import LabelEncoder


app = FastAPI()


class Patient(BaseModel):
    name: str
    text: Optional[str] = None


def load_model(model_path):

    ner = spacy.load(model_path)
    return ner


ner = load_model("custom NER disease v3")


symptoms_model = pickle.load(open("symptoms_model.sav", 'rb'))

loaded_disease_model = pickle.load(open("diseaes_predictor_model.sav", 'rb'))


df = pd.read_csv("overview-of-recordings(symptoms).csv")

data = df[['phrase', 'prompt']]


def clean_txt(docs):
    lemmatizer = WordNetLemmatizer()
    # split into words
    speech_words = nltk.word_tokenize(docs)
    # convert to lower case
    lower_text = [w.lower() for w in speech_words]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in lower_text]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in list(STOP_WORDS)]
    # Stemm all the words in the sentence
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    combined_text = ' '.join(lem_words)
    return combined_text


data['phrase'] = data['phrase'].apply(clean_txt)
X = data['phrase']


vec = TfidfVectorizer(lowercase=False)
vec.fit_transform(X).toarray()


def detectAilment(text):
    text = [text]
    transform_vect = vec.transform(text)

    if symptoms_model.predict(transform_vect) == 0:
        return "Emotional pain"
    elif symptoms_model.predict(transform_vect) == 1:
        return"Hair falling out"
    elif symptoms_model.predict(transform_vect) == 2:
        return"Heart hurts"
    elif symptoms_model.predict(transform_vect) == 3:
        return"Infected wound"
    elif symptoms_model.predict(transform_vect) == 4:
        return"Foot achne"
    elif symptoms_model.predict(transform_vect) == 5:
        return"Shoulder pain"
    elif symptoms_model.predict(transform_vect) == 6:
        return"Injury from sports"
    elif symptoms_model.predict(transform_vect) == 7:
        return"Skin issue"
    elif symptoms_model.predict(transform_vect) == 8:
        return"Stomach ache"
    elif symptoms_model.predict(transform_vect) == 9:
        return"Knee pain"
    elif symptoms_model.predict(transform_vect) == 10:
        return"Joint pain"
    elif symptoms_model.predict(transform_vect) == 11:
        return"Hard to breath"
    elif symptoms_model.predict(transform_vect) == 12:
        return"Head ache"
    elif symptoms_model.predict(transform_vect) == 13:
        return"Body feels weak"
    elif symptoms_model.predict(transform_vect) == 14:
        return"Feeling Dizzy"
    elif symptoms_model.predict(transform_vect) == 15:
        return"Back pain"
    elif symptoms_model.predict(transform_vect) == 16:
        return"Open wound"
    elif symptoms_model.predict(transform_vect) == 17:
        return"Internal pain"
    elif symptoms_model.predict(transform_vect) == 18:
        return"Blurry vision"
    elif symptoms_model.predict(transform_vect) == 19:
        return"Acne"
    elif symptoms_model.predict(transform_vect) == 20:
        return"Muscle pain"
    elif symptoms_model.predict(transform_vect) == 21:
        return"Neck pain"
    elif symptoms_model.predict(transform_vect) == 22:
        return"Cough"
    elif symptoms_model.predict(transform_vect) == 23:
        return"Ear ache"

    else:
        return"Feeling cold"


features = ['itching',
            'skin rash',
            'nodal skin eruptions',
            'continuous sneezing',
            'shivering',
            'chills',
            'joint pain',
            'stomach pain',
            'acidity',
            'ulcers on tongue',
            'muscle wasting',
            'vomiting',
            'burning micturition',
            'spotting  urination',
            'fatigue',
            'weight gain',
            'anxiety',
            'cold hands and feets',
            'mood swings',
            'weight loss',
            'restlessness',
            'lethargy',
            'patches in throat',
            'irregular sugar level',
            'cough',
            'high fever',
            'sunken eyes',
            'breathlessness',
            'sweating',
            'dehydration',
            'indigestion',
            'headache',
            'yellowish skin',
            'dark urine',
            'nausea',
            'loss of appetite',
            'pain behind the eyes',
            'back pain',
            'constipation',
            'abdominal pain',
            'diarrhoea',
            'mild fever',
            'yellow urine',
            'yellowing of eyes',
            'acute liver failure',
            'fluid overload',
            'swelling of stomach',
            'swelled lymph nodes',
            'malaise',
            'blurred and distorted vision',
            'phlegm',
            'throat irritation',
            'redness of eyes',
            'sinus pressure',
            'runny nose',
            'congestion',
            'chest pain',
            'weakness in limbs',
            'fast heart rate',
            'pain during bowel movements',
            'pain in anal region',
            'bloody stool',
            'irritation in anus',
            'neck pain',
            'dizziness',
            'cramps',
            'bruising',
            'obesity',
            'swollen legs',
            'swollen blood vessels',
            'puffy face and eyes',
            'enlarged thyroid',
            'brittle nails',
            'swollen extremeties',
            'excessive hunger',
            'extra marital contacts',
            'drying and tingling lips',
            'slurred speech',
            'knee pain',
            'hip joint pain',
            'muscle weakness',
            'stiff neck',
            'swelling joints',
            'movement stiffness',
            'spinning movements',
            'loss of balance',
            'unsteadiness',
            'weakness of one body side',
            'loss of smell',
            'bladder discomfort',
            'foul smell of urine',
            'continuous feel of urine',
            'passage of gases',
            'internal itching',
            'toxic look (typhos)',
            'depression',
            'irritability',
            'muscle pain',
            'altered sensorium',
            'red spots over body',
            'belly pain',
            'abnormal menstruation',
            'dischromic  patches',
            'watering from eyes',
            'increased appetite',
            'polyuria',
            'family history',
            'mucoid sputum',
            'rusty sputum',
            'lack of concentration',
            'visual disturbances',
            'receiving blood transfusion',
            'receiving unsterile injections',
            'coma',
            'stomach bleeding',
            'distention of abdomen',
            'history of alcohol consumption',
            'fluid overload.1',
            'blood in sputum',
            'prominent veins on calf',
            'palpitations',
            'painful walking',
            'pus filled pimples',
            'blackheads',
            'scurring',
            'skin peeling',
            'silver like dusting',
            'small dents in nails',
            'inflammatory nails',
            'blister',
            'red sore around nose',
            'yellow crust ooze']

diagnoseDict = {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
                'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def checkSymptoms(features, inputSymptoms):
    symptoms = []
    for i in range(132):
        symptoms.append(0)

    for i in inputSymptoms:
        for j in features:
            if similar(i, j) >= 0.6:
                idx = features.index(j)
                symptoms[idx] = 1

    return symptoms


@app.get("/")
def read_root():
    return {"started": "health care system"}


@app.post("/complaintText/")
async def create_item(patient: Patient):
    inputSymptoms = []

    doc = ner(patient.text)
    for ent in doc.ents:
        inputSymptoms.append(ent.text)
    # print(ent.text,"|", ent.start_char, ent.end_char,"|", ent.label_, end="\n\n")

    first_diagnose = detectAilment(patient.text)

    sent_text = nltk.sent_tokenize(patient.text)
    for sentence in sent_text:
        inputSymptoms.append(detectAilment(sentence))

    symptoms = checkSymptoms(features, inputSymptoms)

    encoder = LabelEncoder()
    x_t = encoder.fit_transform(symptoms)
    diagnoseLabel = loaded_disease_model.predict(x_t.reshape(1, -1))

    sec_diagnose = ''
    for value, index in diagnoseDict.items():
        if index == diagnoseLabel:
            # print(value)
            sec_diagnose = value

    return {
        'Symptoms': inputSymptoms,
        '1th diagnose': first_diagnose,
        '2th diagnose': sec_diagnose
    }


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')
