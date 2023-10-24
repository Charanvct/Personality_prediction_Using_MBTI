import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from flask import Flask, render_template, request, redirect, url_for
import joblib

# Create Flask app
app = Flask(__name__)




# Load the pre-trained LightGBM models
def load_lightgbm_model():
    try:
        model = joblib.load("path_of_the_pickle_file_in_your_local_system")
        return model
    except Exception as e:
        print(f"Error loading the models: {e}")
        return None




@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        if csv_file.filename != '':
            # Save the uploaded CSV file to a location
            file_path = 'path_for_storing_the_csv_file_in_your_local_system(use \\ in path instead of \ to overcome error)' + csv_file.filename
            csv_file.save(file_path)

            # Load the pre-trained LightGBM models
            loaded_model = load_lightgbm_model()

            if loaded_model:
                input_data=pd.read_csv(file_path)
                input_posts=input_data['posts']

                vectorizer=TfidfVectorizer(max_features=480, stop_words='english')
                vectorizer.fit(input_posts)
                
                input_data_post=vectorizer.transform(input_posts).toarray()
                prediction=loaded_model.predict(input_data_post)

                if prediction is not None:
                    label_encoder=LabelEncoder()
                    training_labels=["INFJ", "INFP", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP", "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ"]
                    label_encoder.fit(training_labels)

                    personality_labels=label_encoder.inverse_transform(prediction)
                    result=f"Predicted Personality Type: {personality_labels[0]}"
                
                else:
                    result="Error in loading the model or getting the prediction"
            else:
                print("The model is not loaded")

               

    return render_template('homepage.html', result=result)




if __name__ == "__main__":
    app.run(debug=True, port=5000)
