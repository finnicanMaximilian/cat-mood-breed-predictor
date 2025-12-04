Cat Mood & Breed Predictor
AI powered cat recognition using ResNet50 + Flask
Upload a photo -> get your cat's breed + mood with fun, personalized responses.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Overview
The Cat Mood & Breed Predictor is a deep-learning project that uses ResNet50, transfer learning, and a custom and an imported dataset from the Oxford Cat & Dog Big Dataset Sample to classify:
* Cat Breeds
* Cat Moods (Happy, Zoomies, Curious, Eepy (Sleepy), Grumpy)

This project includes:
- A flask web app
- A trained model pipeline
- Frontend image upload + predictions
- Fun dynamic messages (e.g., "OMG Benjamin is a Bengal and has the zoomies!!")
- Fully documented and structured code
- Ready to deply to Render / AWS / Azure / Heroku

This project was built as part of a personal ML portfolio and demostrates end to end deep learning deployment.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Demo

* Add GIF Here

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Features
* Cat breed classifcation using fine-tuned ResNet50
* Cat mood detection with custom mood dataset
* Confidence percentages for each prediction
* Image preview + preprocessed version
* Fast inference using TensorFlow
* Flask backend with clean project structure
* Expandable UI for full portfolio site

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
How It Works
1. Model Architecture
    * Base model: ResNet50 pretrained on ImageNet
    * Top layers added:
        * GlobalAveragePooling2D
        * Dense(128) + Dropout
        * Softmax output layer
    * Fine-tuning on last 15 layers for better domain performance

2. Datasets
    * Breed dataset: ~2600 images across 12 breeds (200 per breed)
    * Mood dataset: ~100 images across 5 moods (20 per mood)

3. Preprocessing
    * Training uses:
        * preprocess_input (ResNet standardization)  
        * Augmentations: rotation, zoom, brightness, flips
    * Predictions also use:
        * preprocess_input
        * No augmentation

4. Deployment
    * Models exported as:
        * models/breed_model.h5
        * models/mood_model.h5
    * Loaded in runtime via utils.py
    * Flask recieves an upload -> saves -> predicts -> returns results

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Structure

cat-mood-predictor/
│
├── app.py                    # Flask application
├── utils.py                  # Loads models + prediction helpers
├── requirements.txt
├── README.md
│
├── models/
│   ├── breed_model.h5
│   └── mood_model.h5
│
├── templates/
│   ├── index.html
│   └── result.html (optional)
│
├── static/
│   ├── styles.css
│   ├── script.js
│   └── uploads/
│
└── training/
    └── train_models.py       # Full training pipeline

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Installation

1. Clone the repository
git clone https://github.com/YOUR_USERNAME/cat-mood-predictor.git
cd cat-mood-predictor

2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

3. Install dependencies
pip install -r requirements.txt

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Running the App
python app.py

Flask will start at:
http://127.0.0.1:5000

Open it in your browser and upload a photo of a cat!

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Training your own models (Optional)
Inside /training/train_models.py:
* Dataset splitting
* Augmentation pipelines
* ResNet50 fine-tuning
* Multipass training (frozen -> unfrozen stages)
* Model checkpoints
* Confusio matrix Generation

Run Training:
python training/train_models.py

This will output new .h5 models into /models

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Example Prediction Output

Breed Prediction:
Bengal (92% confidence)

Mood Prediction:
happy (74% confidence)

Final Response:
"OMG Benjamin is a Bengal and he looks so happy today 😺✨"

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Future Improvements
* Add GIF rendering for preprocessing
* Build a cleaner modern UI with Tailwind or React
* Deploy live version with GPU inference
* Add an "Upload multiple photos" batch mode
* Expand mood dataset to 500+ images
* Add Segmentation (crop cat only before prediction)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contributing
PRs welcome - especially for UI improvements or dataset expansion

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Licence
MIT Licence

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Author
Maximilian Finnican
AI Engineer * ML Enthusiast * Full-stack Builder
Charlotte, NC

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
