"""
app.py - Flask Backend لنظام الكشف عن الأمراض الصوتية
═══════════════════════════════════════════════════════════════
تم نسخ جميع الإعدادات بدقة من أكواد المستخدم الأصلية
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from tensorflow import keras
import joblib
import os
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # السماح للـ Android بالوصول

# ═══════════════════════════════════════════════════════════════
# الإعدادات - منسوخة بدقة من config.py
# ═══════════════════════════════════════════════════════════════

# PARAMÈTRES AUDIO
SAMPLE_RATE = 16000          # ⚠️ بالضبط من أكوادك
DURATION = 6                 # ⚠️ بالضبط من أكوادك
NORMALIZE_AUDIO = True

# PARAMÈTRES FEATURES
N_MELS = 128                 # ⚠️ بالضبط من أكوادك
N_FFT = 1024                 # ⚠️ بالضبط من أكوادك
HOP_LENGTH = 512             # ⚠️ بالضبط من أكوادك
FMIN = 50                    # ⚠️ بالضبط من أكوادك
FMAX = 8000                  # ⚠️ بالضبط من أكوادك
N_MFCC = 13                  # ⚠️ بالضبط من أكوادك
N_CHROMA = 12                # ⚠️ بالضبط من أكوادك

# Shape attendu
EXPECTED_N_FEATURES = 153    # 128 + 13 + 12
EXPECTED_TIME_FRAMES = 186   # ⚠️ بالضبط من أكوادك

# Classes de maladies
DISEASE_CLASSES = [
    'asthma_or_copd',
    'covid',
    'parkinson',
    'voice_disorder'
]

DISEASE_NAMES_FR = {
    'asthma_or_copd': 'Asthme/BPCO',
    'covid': 'COVID-19',
    'parkinson': 'Parkinson',
    'voice_disorder': 'Troubles de la voix'
}

DISEASE_NAMES_AR = {
    'asthma_or_copd': 'الربو/مرض الانسداد الرئوي',
    'covid': 'كوفيد-19',
    'parkinson': 'باركنسون',
    'voice_disorder': 'اضطرابات الصوت'
}

DISEASE_ICONS = {
    'asthma_or_copd': '🫁',
    'covid': '🦠',
    'parkinson': '🧠',
    'voice_disorder': '🗣️'
}

# ═══════════════════════════════════════════════════════════════
# تحميل النموذج والـ Scaler
# ═══════════════════════════════════════════════════════════════

print("🔄 Loading model and scaler...")
try:
    model = keras.models.load_model('unified_model_phase2.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ═══════════════════════════════════════════════════════════════
# الوظائف - منسوخة بدقة من utils.py
# ═══════════════════════════════════════════════════════════════

def preprocess_audio(audio, target_length=None):
    """
    منسوخة بدقة 100% من utils.py
    """
    if target_length is None:
        target_length = SAMPLE_RATE * DURATION
    
    current_length = len(audio)
    
    if current_length < target_length:
        # Padding: répéter le signal
        n_repeats = int(np.ceil(target_length / current_length))
        audio = np.tile(audio, n_repeats)[:target_length]
    elif current_length > target_length:
        # Truncation: prendre le centre
        start = (current_length - target_length) // 2
        audio = audio[start:start + target_length]
    
    assert len(audio) == target_length, "Erreur prétraitement audio"
    return audio


def extract_features_phase1(audio):
    """
    منسوخة بدقة 100% من utils.py
    Extrait EXACTEMENT les mêmes features que Phase 1
    """
    features_list = []
    
    # 1. Mel Spectrogram (128 features)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features_list.append(mel_spec_db)
    
    # 2. MFCC (13 features)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    features_list.append(mfcc)
    
    # 3. Chroma (12 features)
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA
    )
    features_list.append(chroma)
    
    # Combiner verticalement
    features = np.vstack(features_list)
    
    # Vérification critique du shape
    expected_shape = (EXPECTED_N_FEATURES, EXPECTED_TIME_FRAMES)
    
    # Si time dimension différente, forcer à 186 frames
    if features.shape[1] != EXPECTED_TIME_FRAMES:
        if features.shape[1] < EXPECTED_TIME_FRAMES:
            # Pad with zeros
            pad_width = EXPECTED_TIME_FRAMES - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            features = features[:, :EXPECTED_TIME_FRAMES]
    
    # Vérification finale
    assert features.shape == expected_shape, \
        f"❌ Shape invalide! Attendu {expected_shape}, reçu {features.shape}"
    
    return features


def load_audio_file(audio_bytes):
    """
    تحميل الصوت من bytes (من Android)
    منسوخة من utils.py مع تعديل للـ bytes
    """
    try:
        # تحميل من bytes
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
            duration=None
        )
        
        if len(audio) == 0:
            return None
        
        # Normaliser
        if NORMALIZE_AUDIO:
            audio = librosa.util.normalize(audio)
        
        # Prétraiter pour obtenir longueur exacte
        audio = preprocess_audio(audio)
        
        return audio
        
    except Exception as e:
        print(f"⚠️ Erreur chargement audio: {e}")
        return None


def predict_audio(audio):
    """
    منسوخة من 09_unified_prediction.py
    """
    try:
        # 1. Extraction features
        features = extract_features_phase1(audio)
        
        # 2. Normalisation avec scaler
        features_flat = features.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features_flat)
        
        # 3. Reshape pour le modèle
        features_final = features_scaled.reshape(
            1, EXPECTED_N_FEATURES, EXPECTED_TIME_FRAMES, 1
        )
        
        # 4. Prédiction
        predictions = model.predict(features_final, verbose=0)
        
        # 5. Extraction résultats
        if isinstance(predictions, dict):
            binary_key = [k for k in predictions.keys() if k != 'disease_output'][0]
            binary_pred = predictions[binary_key]
            disease_pred = predictions['disease_output']
        else:
            binary_pred, disease_pred = predictions
        
        # 6. Phase 1: Healthy vs Sick
        binary_prob = float(binary_pred[0][0])
        is_healthy = binary_prob < 0.5
        binary_confidence = float((1 - binary_prob) if is_healthy else binary_prob)
        
        if is_healthy:
            return {
                'success': True,
                'healthy': True,
                'binary_confidence': binary_confidence,
                'disease': None,
                'disease_name_fr': None,
                'disease_name_ar': None,
                'disease_confidence': None,
                'icon': '✅'
            }
        
        # 7. Phase 2: Disease classification
        disease_probs = disease_pred[0]
        disease_idx = int(np.argmax(disease_probs))
        disease_name = DISEASE_CLASSES[disease_idx]
        disease_confidence = float(disease_probs[disease_idx])
        
        return {
            'success': True,
            'healthy': False,
            'binary_confidence': binary_confidence,
            'disease': disease_name,
            'disease_name_fr': DISEASE_NAMES_FR[disease_name],
            'disease_name_ar': DISEASE_NAMES_AR[disease_name],
            'disease_confidence': disease_confidence,
            'icon': DISEASE_ICONS[disease_name],
            'all_probabilities': {
                DISEASE_CLASSES[i]: float(disease_probs[i])
                for i in range(len(DISEASE_CLASSES))
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    """صفحة البداية"""
    return jsonify({
        'message': 'Disease Detection API',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Upload audio file for prediction',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """فحص صحة الـ API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    استقبال الصوت من Android والتنبؤ
    """
    try:
        # 1. التحقق من وجود ملف
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # 2. قراءة الملف
        audio_bytes = audio_file.read()
        
        # 3. تحميل ومعالجة الصوت
        audio = load_audio_file(audio_bytes)
        
        if audio is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load audio'
            }), 400
        
        # 4. التنبؤ
        result = predict_audio(audio)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ═══════════════════════════════════════════════════════════════
# تشغيل التطبيق
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
