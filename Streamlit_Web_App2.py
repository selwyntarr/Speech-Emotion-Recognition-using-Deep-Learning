import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import io

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_ltsm_best_weights')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Function to extract audio features from the uploaded audio file
def extract_features(audio_data):
    audio_bytes = io.BytesIO(audio_data)
    y, sr = librosa.load(audio_bytes, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    # Reshape the features to match the expected input shape of the model
    features = np.expand_dims(mfccs_scaled, axis=0)
    features = np.expand_dims(features, axis=-1)
    features = tf.image.resize(features, [162, 1])  # Reshape to (162, 1)
    return features

# Main function for creating the Streamlit app
def main():
    st.title('Speech Emotion Recognition using CNN')
    st.write('Upload an audio file (.wav or .mp3) and check the predicted emotion!')

    audio_file = st.file_uploader('Upload Audio', type=['wav', 'mp3'])

    if audio_file is not None:
        audio_data = audio_file.read()
        st.audio(audio_data, format='audio/wav')

        if st.button('Recognize Emotion'):
            try:
                features = extract_features(audio_data)
                features = np.expand_dims(features, axis=0)
                predicted_probabilities = model.predict(features)[0]
                predicted_emotion = emotion_labels[np.argmax(predicted_probabilities)]
                st.success(f'Predicted Emotion: {predicted_emotion}')
            except Exception as e:
                st.error(f'Error: {e}')

if __name__ == '__main__':
    main()
