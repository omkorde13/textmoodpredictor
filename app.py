import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mental_stress_detector import MentalStressDetectorCSV

@st.cache_resource
def download_nltk_data():
    """Downloads NLTK data packages required for the app."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

@st.cache(allow_output_mutation=True)
def load_detector():
    return MentalStressDetectorCSV()


def main():
    download_nltk_data()

    # ... rest of your main function ...
    st.title("Stress Burnout Detection")
    st.title("🧠 Mental Stress & Burnout Detection")

    detector = load_detector()

    menu = ["Home", "Use Sample Data", "Upload CSV", "Custom Text Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(
            """
            This app uses machine learning to detect mental stress levels from text input.

            **Features:**
            - Train and evaluate models on uploaded CSV data
            - Predict stress level from custom text input
            - Display evaluation results and feature importance
            """
        )

    elif choice == "Use Sample Data":
        st.header("Sample Data Analysis")
        if st.button("Run Sample Data Pipeline"):
            sample_csv_path = detector.create_sample_csv()
            results = detector.run_complete_pipeline_from_csv(sample_csv_path)

            st.success("Pipeline completed successfully!")

            st.subheader("Stress Level Distribution")
            st.bar_chart(results['dataset'][detector.label_column].value_counts().sort_index())

            st.subheader("Model Accuracy Scores")
            st.write(results['scores'])

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Low Stress', 'Medium Stress', 'High Stress'],
                        yticklabels=['Low Stress', 'Medium Stress', 'High Stress'], ax=ax)
            st.pyplot(fig)

            st.subheader("Feature Importance")
            detector.plot_feature_importance()
            st.pyplot(plt.gcf())

    elif choice == "Upload CSV":
        st.header("Upload Your CSV Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("Run Pipeline on Uploaded Data"):
                    with st.spinner("Training and evaluating models..."):
                        results = detector.run_complete_pipeline_from_csv(df=df)

                    st.success("Pipeline completed successfully!")

                    st.subheader("Stress Level Distribution")
                    st.bar_chart(results['dataset'][detector.label_column].value_counts().sort_index())

                    st.subheader("Model Accuracy Scores")
                    st.write(results['scores'])

                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Low Stress', 'Medium Stress', 'High Stress'],
                                yticklabels=['Low Stress', 'Medium Stress', 'High Stress'], ax=ax)
                    st.pyplot(fig)

                    st.subheader("Feature Importance")
                    detector.plot_feature_importance()
                    st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Error loading or processing file: {str(e)}")

    elif choice == "Custom Text Prediction":
        st.header("Predict Stress Level from Custom Text")
        user_input = st.text_area("Enter text here:", height=150)

        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please enter some text for prediction.")
            else:
                if detector.model is None:
                    st.info("No trained model found, running sample data pipeline first...")
                    sample_csv_path = detector.create_sample_csv()
                    detector.run_complete_pipeline_from_csv(sample_csv_path)

                result = detector.predict_stress_level(user_input)
                st.write(f"**Predicted Stress Level:** {result['predicted_level']}")
                st.write(f"**Confidence:** {result['confidence']:.3f}")

                levels = list(result['probabilities'].keys())
                probabilities = list(result['probabilities'].values())
                colors = ['green', 'orange', 'red']

                fig, ax = plt.subplots()
                bars = ax.bar(levels, probabilities, color=colors, alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_title('Stress Level Probability Distribution')

                predicted_idx = levels.index(result['predicted_level'])
                bars[predicted_idx].set_edgecolor('black')
                bars[predicted_idx].set_linewidth(3)

                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

                st.pyplot(fig)


if __name__ == "__main__":
    main()
