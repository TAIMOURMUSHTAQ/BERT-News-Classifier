# News Topic Classifier Using BERT
A fine-tuned BERT model for classifying news headlines into topic categories with over 94% accuracy.
## 📖 Overview
This project demonstrates how to fine-tune a BERT transformer model for multi-class text classification using the AG News dataset. The implementation includes data preprocessing, model training, evaluation, and deployment options using both Streamlit and Gradio.
The classifier can accurately categorize news headlines into four topics:
•	🌍 World News
•	⚽ Sports
•	💼 Business
•	🔬 Science & Technology

## ✨ Features
•	**BERT Fine-tuning**: Customized bert-base-uncased model for news classification
•	**High Accuracy**: Achieves 94% accuracy on the test set
•	**Comprehensive Evaluation**: Detailed metrics including F1-score, precision, and recall
•	**Two Deployment Options**: Streamlit web app and Gradio interface
•	**Production Ready**: Complete pipeline from data to deployment

## 🚀 Quick Start
### Installation
```bash
# Clone the repository
git clone https://github.com/taimourmushtaq /BERT-News-Classifier.git
cd news-topic-classifier-bert

# Install dependencies
pip install -r requirements.txt
Usage
1.	Train the model:
train_bert_classifier.py
2.	Run the Streamlit app:
streamlit run app.py
3.	Run the Gradio app:
python gradio_app.py

📊 Dataset
The model is trained on the AG News Dataset from Hugging Face, which contains 120,000 training samples and 7,600 testing samples across four news categories.
Class Distribution:
•	World News: 30,000 samples
•	Sports: 30,000 samples
•	Business: 30,000 samples
•	Science/Tech: 30,000 samples
📈 Performance
The fine-tuned BERT model achieves the following performance metrics:
Metric	Score
Accuracy	94.2%
Weighted F1-score	94.1%
Precision	94.3%
Recall	94.2%

🏗️ Model Architecture
The project uses a pre-trained bert-base-uncased model with a classification head fine-tuned on the AG News dataset:
1.	Input: News headlines (max length: 128 tokens)
2.	Base Model: BERT (12-layer, 768-hidden, 12-heads, 110M parameters)
3.	Classification Layer: Linear layer with 4 output units (one per class)
4.	Training: 3 epochs with learning rate 2e-5
🎯 Usage Examples
Python API
from transformers import pipeline
classifier = pipeline("text-classification", 
                      model="your-username/bert-ag-news")
result = classifier("Apple announces new iPhone with advanced AI features")
# Output: {'label': 'SCI/TECH', 'score': 0.95}
Web Interface
The application provides interactive interfaces where users can:
•	Input news headlines for instant classification
•	View confidence scores for all categories
•	Test with example headlines
•	Batch process multiple headlines

📁 Project Structure
news-topic-classifier-bert/
├── models/                    # Saved model files
│   └── bert_ag_news/         # Fine-tuned BERT model
├── notebooks/                # Jupyter notebooks
│   └── News_Classification_with_BERT.ipynb
├── app.py                    # Streamlit application
├── gradio_app.py            # Gradio interface
├── train_bert_classifier.py # Training script
├── requirements.txt         # Dependencies
└── README.md               # Documentation

🔧 Customization
To fine-tune on your own dataset:
1.	Replace the AG News dataset with your custom data
2.	Update the number of classes in the model configuration
3.	Adjust training parameters as needed
4.	Modify the category labels in the deployment apps

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
1.	Fork the project
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request

📧 **Author**
Taimour Mushtaq
🎓 BSCS Student at Federal Urdu University of Arts,Science and Technology, Islamabad Pakistan
🔗 https://www.linkedin.com/in/taimourmushtaq/ |https://github.com/TAIMOURMUSHTAQ

