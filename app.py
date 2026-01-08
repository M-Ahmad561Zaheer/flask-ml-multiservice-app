from flask import Flask, render_template, request
import os
import time
import torch
import torch.nn as nn
import whisper
import pandas as pd
from PIL import Image
from torchvision import transforms
from gtts import gTTS
from transformers import pipeline

# Data Mining Imports
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Folders Configuration
UPLOAD_FOLDER = "static/uploads"
AUDIO_FOLDER = "static/audio"
MODELS_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.environ["PATH"] += os.pathsep + os.getcwd()

# ---------------- CNN MODEL ARCHITECTURE ----------------
class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ---------------- LOAD MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Voice Models
whisper_model = whisper.load_model("tiny").to(device)

# 2. Image Model
cnn_model = GenderCNN().to(device)
model_path = os.path.join(MODELS_FOLDER, "gender_model.pth")
if os.path.exists(model_path):
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
cnn_model.eval()

# 3. NLP Pipelines
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0 if torch.cuda.is_available() else -1)
text_gen_model = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
translator_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur", device=0 if torch.cuda.is_available() else -1)

# ---------------- PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- CORE ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")

# 1. Image Classification
@app.route("/image", methods=["GET", "POST"])
def image_classification():
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = cnn_model(img_tensor).item()
            result = "Male" if prediction > 0.5 else "Female"
    return render_template("image.html", result=result)

# 2. Voice Sentiment Analysis
@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    result = None
    if request.method == "POST":
        audio = request.files.get("audio")
        if audio:
            path = os.path.join(AUDIO_FOLDER, audio.filename)
            audio.save(path)
            transcription = whisper_model.transcribe(path)["text"]
            label = sentiment_model(transcription)[0]["label"]
            result = f"Transcribed: {transcription} | Sentiment: {label}"
    return render_template("sentiment.html", result=result)

# 3. Voice QA with TTS
@app.route("/qa", methods=["GET", "POST"])
def qa():
    audio_file = None
    if request.method == "POST":
        audio = request.files.get("audio")
        context = request.form.get("context")
        if audio and context:
            path = os.path.join(AUDIO_FOLDER, audio.filename)
            audio.save(path)
            question = whisper_model.transcribe(path)["text"]
            answer = qa_model(question=question, context=context)["answer"]
            
            audio_file = f"answer_{int(time.time())}.mp3"
            tts = gTTS(answer)
            tts.save(os.path.join(AUDIO_FOLDER, audio_file))
    return render_template("qa.html", audio_file=audio_file)

# 4. Text Generation (GPT-2)
@app.route("/generate", methods=["GET", "POST"])
def generate():
    output = None
    if request.method == "POST":
        prompt = request.form.get("prompt")
        output = text_gen_model(prompt, max_new_tokens=50, pad_token_id=50256)[0]["generated_text"]
    return render_template("generate.html", output=output)

# 5. Translation (English to Urdu)
@app.route("/translate", methods=["GET", "POST"])
def translate():
    output = None
    if request.method == "POST":
        text = request.form.get("text")
        output = translator_model(text)[0]["translation_text"]
    return render_template("translate.html", output=output)

# ---------------- DATA MINING ROUTES (FIXED) ----------------

@app.route("/apriori", methods=["GET", "POST"])
def apriori_mining():
    results = None
    if request.method == "POST":
        file = request.files.get("dataset")
        if file:
            df = pd.read_csv(file)
            
            # STEP 1: Sirf categorical columns lein aur one-hot encoding karein
            df_encoded = pd.get_dummies(df)
            
            # STEP 2: MLxtend ki naye version ke liye True/False ko 1/0 ya bool mein convert karna lazmi hai
            df_encoded = df_encoded.astype(bool) 

            try:
                # STEP 3: Apriori run karein
                frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
                
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                    # Frozesets ko list mein convert karein taake HTML display ho sake
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
                    results = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict(orient='records')
                else:
                    results = "EMPTY_RESULTS"
            except Exception as e:
                print(f"Apriori Error: {e}")
                results = f"Error: {str(e)}"
                
    return render_template("apriori.html", results=results)

@app.route("/cluster/<algo>", methods=["GET", "POST"])
def cluster_analysis(algo):
    results = None
    if request.method == "POST":
        file = request.files.get("dataset")
        if file:
            df = pd.read_csv(file)
            # Sirf numbers wali columns chunein
            numeric_df = df.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                # Data ko scale karein (Zaroori for Clustering)
                data_scaled = StandardScaler().fit_transform(numeric_df)
                
                if algo == "kmeans":
                    # K ki value check karein
                    k_val = request.form.get("k_clusters")
                    k = int(k_val) if k_val and k_val.isdigit() else 3
                    model = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data_scaled)
                else:
                    model = DBSCAN(eps=0.5, min_samples=5).fit(data_scaled)
                
                df['Cluster_ID'] = model.labels_
                # Results ko HTML table mein convert karein
                results = df.head(25).to_html(classes="min-w-full border text-sm", index=False)
            else:
                results = "NO_NUMERIC_DATA"
            
    return render_template("clustering.html", results=results, algo=algo.upper())

if __name__ == "__main__":
    app.run(debug=True)