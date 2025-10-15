# 📰 AI-Powered Fake News Detector

An **NLP-based Fake News Detection** system using **BiLSTM** and **GloVe embeddings**, deployed as an interactive **Streamlit web app** and containerized with **Docker**.
It intelligently classifies news articles as **“True”** or **“Fake”** using deep learning.

---

## 💡 Project Highlights

* Built a **deep learning model** (BiLSTM) for text classification
* Used **GloVe word embeddings** for semantic understanding
* Achieved strong performance on real-world fake news datasets
* Designed a **Streamlit web app** for live text predictions
* Fully **Dockerized** for deployment on any platform (Render, Hugging Face, etc.)

---

## 🛠️ Tech Stack

**Python**, **TensorFlow / Keras**, **NLP (GloVe)**, **Streamlit**, **Docker**, **Pandas**, **NumPy**

---

## 📁 Project Structure

```bash
Fake-News-Detector/
├── app/
│   └── app_streamlit.py        # Streamlit web application
│
├── models/
│   ├── model.keras             # (Optional) Pretrained model
│   └── tokenizer.json          # Tokenizer for preprocessing
│
├── notebooks/
│   └── True__Fake_News_NLP_GloVe__LSTM.ipynb   # Model training & exploration
│
├── src/
│   ├── preprocess.py           # Text cleaning & tokenization
│   ├── predict.py              # Model inference logic
│   ├── train.py                # Model training script
│
├── tests/
│   └── test_basic.py           # Basic unit tests
│
├── Dockerfile                  # Docker setup for deployment
├── requirements.txt            # Python dependencies
├── run.sh                      # Shell script for running the app
└── README.md                   # Project documentation
```

---

## ⚙️ Setup & Installation

### 🔹 Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/Fake-News-Detector.git
cd Fake-News-Detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app_streamlit.py
```

Then open your browser at 👉 [http://localhost:8501](http://localhost:8501)

---

### 🔹 Option 2: Run with Docker

```bash
# Build the Docker image
docker build -t fake-news-detector .

# Run the container
docker run -p 8501:8501 fake-news-detector
```

Then visit 👉 [http://localhost:8501](http://localhost:8501)

---

## 🧠 Model Training

Train your own model using:

```bash
python src/train.py
```

Or explore the Jupyter notebook:

```
notebooks/True__Fake_News_NLP_GloVe__LSTM.ipynb
```

---

## 📚 Pretrained GloVe Embeddings

Download GloVe embeddings from:
🔗 [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

Place the file (`glove.twitter.27B.100d.txt`) inside a `data/` folder before training.

---

## 🧾 Model Weights

Due to file size constraints, pretrained model weights are **not included** in this repository.

You can:

* Download them separately (add your Google Drive or Hugging Face link here if desired)
* Or retrain the model using `src/train.py` or the notebook.

---

## 🧪 Testing

Run basic tests:

```bash
pytest tests/
```

---

## 🐳 Deployment Options

This app can be deployed easily to:

* [Render](https://render.com)
* [Hugging Face Spaces](https://huggingface.co/spaces)
* [Docker Hub](https://hub.docker.com)
* AWS ECS / Azure Container Apps

---

## 👨‍💻 Author

**Mukaram Ali**
🔗 [LinkedIn](https://www.linkedin.com/in/mukaram-ali-a05061279/)
💻 [GitHub](https://github.com/<your-username>)
📧 [mukaramali@example.com](mailto:mukaramali@example.com) *(optional)*

---

## 🪪 License

**MIT License © 2025 Mukaram Ali**
Feel free to fork, modify, and share with proper credit.

---

## 🌟 Acknowledgements

* [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/) — Stanford NLP
* TensorFlow / Keras Team
* Streamlit for the awesome UI framework