# mental_stress_detector.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

# Download essential NLTK data if not already installed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class MentalStressDetectorCSV:
    """
    A comprehensive ML system for detecting mental stress and burnout
    from CSV text data using NLP and ensemble methods.
    """

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.dataset = None
        self.X_test = None
        self.y_test = None

    def load_csv_data(self, csv_path=None, text_column='text', label_column='stress_level', df=None):
        print(f"📊 Loading data from CSV...")

        if df is not None:
            df = df.copy()
            print(f"✅ Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        else:
            raise ValueError("Either csv_path or df must be provided")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")

        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")

        if df[text_column].isnull().sum() > 0:
            print(f"⚠️ Missing values in text column found; dropping those rows")
            df = df.dropna(subset=[text_column])

        if df[label_column].isnull().sum() > 0:
            print(f"⚠️ Missing values in label column found; dropping those rows")
            df = df.dropna(subset=[label_column])

        unique_labels = df[label_column].unique()
        print(f"📈 Unique stress levels: {unique_labels}")

        if df[label_column].dtype not in ['int64', 'float64']:
            print("📝 Converting categorical labels to numeric")
            label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            df[label_column] = df[label_column].map(label_mapping)
            print(f"Label mapping: {label_mapping}")

        self.dataset = df
        self.text_column = text_column
        self.label_column = label_column

        print(f"✅ Data prepared with {len(df)} samples")
        print(f"Stress level distribution:\n{df[label_column].value_counts().sort_index()}")

        return df

    def create_sample_csv(self, output_path='sample_stress_data.csv'):
        print(f"📝 Creating sample CSV: {output_path}")

        sample_data = {
            'text': [
                "I'm so overwhelmed with work, I can't sleep at night",
                "Everything feels like too much, I'm constantly anxious",
                "I feel burned out and exhausted all the time",
                "I can't concentrate, my mind is racing with worries",
                "I feel like I'm drowning in responsibilities",
                "I'm so stressed I can't even think straight",
                "I feel like I'm on the verge of a breakdown",
                "I'm constantly worried about everything",
                "I feel like I'm failing at everything",
                "I'm so tired but I can't relax",
                "I'm struggling with anxiety and panic attacks",
                "I feel like I'm losing control of my life",
                "I'm so stressed I can't eat or sleep properly",
                "I feel hopeless and don't know what to do",
                "I'm constantly worried about my health",

                "Work has been challenging lately, but I'm managing",
                "I feel a bit overwhelmed but I'll get through it",
                "Some days are harder than others",
                "I'm feeling a bit stressed but it's manageable",
                "I have some concerns but I'm coping",
                "I feel pressure but I can handle it",
                "I'm a bit worried about the future",
                "I feel some tension but it's not too bad",
                "I have some stress but I'm dealing with it",
                "I feel a bit anxious but I'm okay",
                "I'm feeling a bit tired but generally okay",
                "I have some work stress but I'm handling it",
                "I feel a bit anxious about the presentation",
                "I'm concerned about the project deadline",
                "I feel some pressure but I'm managing",

                "I'm feeling good and positive about things",
                "Life is going well and I'm happy",
                "I feel calm and relaxed today",
                "I'm in a good mood and feeling optimistic",
                "Everything seems to be working out fine",
                "I feel peaceful and content",
                "I'm feeling great and motivated",
                "I have a positive outlook on life",
                "I feel balanced and healthy",
                "I'm feeling confident and secure",
                "I'm feeling wonderful and full of energy",
                "I'm so happy and grateful for everything",
                "I feel amazing and ready to take on anything",
                "I'm feeling confident and successful",
                "I feel peaceful and satisfied with life"
            ],
            'stress_level': [2] * 15 + [1] * 15 + [0] * 15,
            'timestamp': pd.date_range('2024-01-01', periods=45, freq='D'),
            'user_id': [f'user_{i%10}' for i in range(45)]
        }

        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"✅ Sample CSV created with {len(df)} samples")
        print(f"📁 File saved to: {output_path}")

        return output_path

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)

    def extract_features(self, texts):
        print("🔧 Extracting features from text data...")
        processed_texts = [self.preprocess_text(text) for text in texts]

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        features = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"✅ Extracted {features.shape[1]} features")
        return features

    def train_models(self, X, y):
        print("🤖 Training multiple ML models...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': MultinomialNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        model_scores = {}
        trained_models = {}

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            model_scores[name] = accuracy
            trained_models[name] = model

            print(f"     Accuracy: {accuracy:.3f}")

        best_model_name = max(model_scores, key=model_scores.get)
        self.model = trained_models[best_model_name]

        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {model_scores[best_model_name]:.3f})")

        self.X_test = X_test
        self.y_test = y_test

        return trained_models, model_scores

    def evaluate_model(self):
        print("📊 Evaluating model performance...")

        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")

        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['Low Stress', 'Medium Stress', 'High Stress']))

        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low Stress', 'Medium Stress', 'High Stress'],
                    yticklabels=['Low Stress', 'Medium Stress', 'High Stress'])
        plt.title('Confusion Matrix - Mental Stress Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        return accuracy, cm

    def plot_feature_importance(self, top_n=20):
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_

            top_indices = np.argsort(importance)[-top_n:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features for Stress Detection')
            plt.tight_layout()
            plt.show()

        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_

            top_indices = np.argsort(np.abs(coef))[-top_n:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_coef = coef[top_indices]

            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_coef]
            plt.barh(range(len(top_features)), top_coef, color=colors)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Coefficient Value')
            plt.title(f'Top {top_n} Most Important Features for Stress Detection')
            plt.tight_layout()
            plt.show()

    def predict_stress_level(self, text):
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]

        stress_levels = ['Low Stress', 'Medium Stress', 'High Stress']
        predicted_level = stress_levels[prediction]
        confidence = probabilities[prediction]

        return {
            'predicted_level': predicted_level,
            'confidence': confidence,
            'probabilities': {
                'Low Stress': probabilities[0],
                'Medium Stress': probabilities[1],
                'High Stress': probabilities[2]
            }
        }

    def run_complete_pipeline_from_csv(self, csv_path=None, text_column='text', label_column='stress_level', df=None):
        print("🚀 Starting Mental Stress Detection Pipeline from CSV")
        print("=" * 60)

        dataset = self.load_csv_data(csv_path, text_column, label_column, df)

        X = self.extract_features(dataset[text_column].tolist())
        y = dataset[label_column].values

        models, scores = self.train_models(X, y)
        accuracy, cm = self.evaluate_model()
        self.plot_feature_importance()

        print("\n✅ Pipeline completed successfully!")
        return {
            'dataset': dataset,
            'models': models,
            'scores': scores,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
