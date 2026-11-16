from pathlib import Path
import pickle
import re
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from keras.models import load_model
from keras.utils import pad_sequences

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "Artifacts" / "BIGRU_Model.h5"
tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"

max_sequence_length = 50

emotion_labels = [
    'sadness',
    'joy',
    'love',
    'anger',
    'fear',
    'surprise'
]
   
 
"""
1.We are going to make some constants

A. Model path(BiGRU)
B.Tokenizer Path
C.MaxSequence Length
D.Emotion Labels
E. Emotion emojis

"""
 
max_sequence_length=50

emotion_emojis={'anger':'😠','fear':'😨','joy':'😂','love':'😍','sadness':'😢','surprise':'😲'}


"""
2.Preprocess the upcoming text
clean raw text so it matches the format used while training
A.Cponvert the text to lower case
B.Remove the special characters
C.Remove the extra spaces
D.Remove apostrophes
"""
def preprocess_text(text:str)->str:
    text=text.lower()
    text=re.sub("'","",text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)    
    text =re.sub(r'^\s+|\s+$', '', text)
    return text

"""
3. Request and Response Schemas
A.Text Input->Input schema the text send by user
B.Prediction Response ->Output Schema the emotion to predict
C.Health Response-(Server health check)
"""

class TextInput(BaseModel):
    text:str=Field(...,
                   min_length=1,max_length=2000,description="The sentence to analyse",
                   json_schema_extra={"example":"I am so happy today!"})




class PredictionResponse(BaseModel):
    text:str=Field(...,description="The input text")
    predicted_emotion:str=Field(...,description="The predicted emotion")
    confidence:float=Field(...,description="The confidence score of the prediction")
    all_probabilities:dict=Field(...,description="The probabilities of all emotions")


class HealthResponse(BaseModel):
    status:str
    model_loaded:bool



"""
4.Model loading and lifespan management
load the model and tokenizer once the server startsup
"""
dl_model={}

dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading the model and tokenizer...")

    dl_model["BiGRU"] = load_model(
        model_path,
        compile=False
    )

    with open(tokenizer_path, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)

    print("Model and tokenizer loaded successfully!")

    yield

    dl_model.clear()

app = FastAPI(lifespan=lifespan)
"""
5. Mount the static model to FastAPI app
A.Enable cross origin resource sharing to allow req from diff origins

"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

"""
6.API Endpoints,
A.Server UI at homepage('/')
B.Health Check Endpoint('/health')
C.Predict Emotion Endpoint ('/predict')
"""

#A.Server UI at homepage('/')
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse(STATIC_DIR / "index.html")


#B
@app.get('/health',response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running",model_loaded=bool(dl_model))


#C
# @app.post('/predict',response_model=PredictionResponse)
# def predict_emotion(text_input:TextInput):
#     #1.clean the inputs

     


#     #2.Convert the words into numeric using tokenizer
#     #3.Pad the seq to ensure uniform length
#     #4.Run prediction using BIGRU model
#     #5.Return the top emotion eith its confidence score and all probabilities for each emotion

#     BiGRU_model=dl_model.get("BiGRU")
#     tokenizer_model=dl_model.get("tokenizer")

#     if BiGRU_model is None or tokenizer_model is None:
#         raise HTTPException(statusCode=503,detail="Model is not loaded yet please try again later.") 

#     #1.
#     cleaned_text=preprocess_text(text_input.text)
#     #2.&3
#     tokenized_text=tokenizer.texts_to_sequences(cleaned_text)
#     padded_sequence=pad_sequences(
# tokenized_text,maxlen=max_sequence_length,
# padding='post',truncating='post'
#     )

#     probabilities=BiGRU_model.predict(padded_sequence)[0]
#     top_emotion_index=int(np.argmax(probabilities))
#     all_probabilities={
#         label:float(prob) for prob,label in zip(probabilities,emotion_labels)
#     }

#     return PredictionResponse(
#         text=text_input.text,
#         predicted_emotion=emotion_labels[top_emotion_index],
#         confidence=float(probabilities[top_emotion_index]),
#         all_probabilities=all_probabilities,
#     )




     

@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    # 1. Get the loaded model and tokenizer
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # 2. Clean the input text
    cleaned_text = preprocess_text(text_input.text)

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Please enter meaningful text."
        )

    # 3. Convert words into numerical sequences
    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # 4. Pad the sequence
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    # 5. Predict emotion
    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # 6. Get the emotion with the highest probability
    top_emotion_index = int(np.argmax(probabilities))

    # 7. Store probabilities for all emotions
    all_probabilities = {
        label: float(prob)
        for label, prob in zip(emotion_labels, probabilities)
    }

    # 8. Return response
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )