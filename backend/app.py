"""
VoiceSpot Cloud Backend v4.0
=============================
Two-engine system:
  Engine 1 — CNN model  : fast prediction for trained keywords (yes/no/up/down etc.)
  Engine 2 — Whisper AI : recognizes ANY word, no training needed

How it works:
  1. Audio comes in
  2. CNN predicts keyword + confidence
  3. If CNN confidence >= 0.70 AND word is in keyword list → use CNN result
  4. Otherwise → run Whisper to get the actual spoken word
  5. Check if Whisper result is in keyword watch-list → alert if yes
  6. Always return what was spoken, whether known or unknown
"""

import os, sys, json, sqlite3, threading, tempfile, io
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa

try:
    import keras
except ImportError:
    from tensorflow import keras

# ── Firebase ──────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

_bucket = None
firebase_ok = False

def init_firebase():
    global _bucket, firebase_ok
    try:
        cred_json = os.environ.get("FIREBASE_CREDENTIALS")
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        bucket_name = os.environ.get("FIREBASE_BUCKET", "voicespot-defense.appspot.com")
        if cred_json:
            cred = credentials.Certificate(json.loads(cred_json))
        elif cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
        else:
            print("[FIREBASE] No credentials — cloud storage disabled")
            return
        firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        _bucket = fb_storage.bucket()
        firebase_ok = True
        print(f"[FIREBASE] ✅ Connected — bucket: {bucket_name}")
    except Exception as e:
        print(f"[FIREBASE] Failed: {e}")

init_firebase()

# ── Whisper ───────────────────────────────────────────────────────────────────
whisper_model = None
whisper_ok    = False

def init_whisper():
    global whisper_model, whisper_ok
    try:
        import whisper
        model_size = os.environ.get("WHISPER_MODEL", "base")
        print(f"[WHISPER] Loading '{model_size}' model...")
        whisper_model = whisper.load_model(model_size)
        whisper_ok = True
        print("[WHISPER] ✅ Ready")
    except Exception as e:
        print(f"[WHISPER] Failed to load: {e}")
        print("[WHISPER] Install with: pip install openai-whisper")

init_whisper()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DB_PATH     = os.path.join(BASE_DIR, "predictions.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SAMPLE_RATE = 16000
DURATION    = 1.0
N_MELS      = 124
N_FFT       = 512
HOP_LENGTH  = 83
N_TIME      = 193

# CNN confidence threshold — below this, fall back to Whisper
CNN_THRESHOLD = 0.70

# ── Keywords ──────────────────────────────────────────────────────────────────
KEYWORDS_FILE = os.path.join(MODEL_DIR, "keywords.json")
# Watch-list: any word Whisper hears that matches this list triggers an alert
WATCHLIST_FILE = os.path.join(MODEL_DIR, "watchlist.json")

def load_keywords():
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE) as f:
            return json.load(f)
    return ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']

def save_keywords(kws):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(KEYWORDS_FILE, 'w') as f:
        json.dump(kws, f)

def load_watchlist():
    """Watch-list = words to alert on (Whisper-detected, no CNN needed)."""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return []  # starts empty

def save_watchlist(wl):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(wl, f)

KEYWORDS  = load_keywords()
WATCHLIST = load_watchlist()  # e.g. ["drum", "dishoom", "crossconnection"]

# ── Training status ───────────────────────────────────────────────────────────
train_status = {
    "running": False, "progress": 0,
    "message": "Idle", "last_accuracy": None,
    "last_trained": None, "error": None
}

# ── CNN model ─────────────────────────────────────────────────────────────────
cnn_model = None

def find_model():
    for name in ["keyword_model.keras", "best_model.keras", "keyword_model.h5"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p): return p
    return None

def load_cnn():
    global cnn_model
    mp = find_model()
    if not mp:
        print("[CNN] No model file found")
        return
    try:
        cnn_model = keras.models.load_model(mp)
        print(f"[CNN] ✅ Loaded: {mp} | input={cnn_model.input_shape}")
    except Exception as e:
        try:
            cnn_model = keras.models.load_model(mp, compile=False)
            cnn_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            print(f"[CNN] ✅ Loaded (legacy): {mp}")
        except Exception as e2:
            print(f"[CNN] ❌ Load failed: {e2}")

load_cnn()

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT, keyword TEXT, confidence REAL,
         speaker TEXT, alert_level INTEGER,
         engine TEXT, raw_transcript TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS speakers
        (name TEXT PRIMARY KEY, embedding TEXT, enrolled_at TEXT)""")
    conn.commit(); conn.close()

init_db()

# ── Audio helpers ─────────────────────────────────────────────────────────────
def bytes_to_audio(audio_bytes):
    """Decode any audio format → numpy float32 at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        try: os.unlink(tmp_path)
        except: pass
    return audio

def save_temp_wav(audio_bytes):
    """Save audio bytes as a temp .wav file — needed by Whisper."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes); return tmp.name

def make_spectrogram(audio):
    target = int(SAMPLE_RATE * DURATION)
    audio = audio[:target] if len(audio)>=target else np.pad(audio,(0,target-len(audio)))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-8)
    mel_db = mel_db[:,:N_TIME] if mel_db.shape[1]>=N_TIME \
             else np.pad(mel_db,((0,0),(0,N_TIME-mel_db.shape[1])))
    return mel_db[...,np.newaxis].astype(np.float32)

def get_alert_level(confidence):
    if confidence>=0.95: return 5
    if confidence>=0.85: return 4
    if confidence>=0.70: return 3
    if confidence>=0.50: return 2
    return 1

def cosine_sim(a, b):
    a,b = np.array(a),np.array(b)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

# ── Whisper transcription ─────────────────────────────────────────────────────
def transcribe_with_whisper(audio_bytes):
    """
    Use Whisper to transcribe audio bytes → text.
    Returns dict: {text, words, language, segments}
    """
    if not whisper_ok or whisper_model is None:
        return None

    tmp_path = save_temp_wav(audio_bytes)
    try:
        result = whisper_model.transcribe(
            tmp_path,
            word_timestamps=True,   # get per-word timing
            language=None,          # auto-detect language (English, Hindi, Chinese...)
            fp16=False
        )
        return result
    except Exception as e:
        print(f"[WHISPER] Transcribe error: {e}")
        return None
    finally:
        try: os.unlink(tmp_path)
        except: pass

def extract_whisper_keyword(result):
    """
    Extract the most prominent single word from Whisper result.
    For short 1-second clips, this is usually just one word.
    """
    if not result: return None, 0.0
    text = result.get('text', '').strip().lower()
    # Clean punctuation
    import re
    text = re.sub(r'[^a-z0-9\s]', '', text).strip()
    if not text: return None, 0.0
    # For short clips, take the first/main word
    words = text.split()
    if not words: return None, 0.0
    # Try to get confidence from segments
    confidence = 0.75  # default Whisper confidence
    segs = result.get('segments', [])
    if segs:
        # Average log probability → confidence estimate
        avg_logprob = np.mean([s.get('avg_logprob', -1) for s in segs])
        confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
    return words[0], confidence

# ── Firebase helpers ──────────────────────────────────────────────────────────
def upload_to_firebase(audio_bytes, keyword, filename):
    if not firebase_ok or not _bucket: return None
    try:
        blob = _bucket.blob(f"keywords/{keyword}/{filename}")
        blob.upload_from_string(audio_bytes, content_type='audio/wav')
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"[FIREBASE] Upload failed: {e}"); return None

def download_keyword_samples(keyword):
    if not firebase_ok or not _bucket: return 0
    local_dir = os.path.join(DATASET_DIR, keyword.lower())
    os.makedirs(local_dir, exist_ok=True)
    blobs = list(_bucket.list_blobs(prefix=f"keywords/{keyword.lower()}/"))
    count = 0
    for blob in blobs:
        fname = os.path.basename(blob.name)
        if fname:
            blob.download_to_filename(os.path.join(local_dir, fname))
            count += 1
    print(f"[FIREBASE] Downloaded {count} files for '{keyword}'")
    return count

# ── Auto-train (background thread) ───────────────────────────────────────────
def run_training(keywords_to_train):
    global cnn_model, KEYWORDS, train_status
    train_status.update(running=True, progress=5, message="Starting...", error=None)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle as sk_shuffle

        train_status.update(progress=10, message="Downloading audio from Firebase...")
        for kw in keywords_to_train:
            download_keyword_samples(kw)

        train_status.update(progress=25, message="Loading audio files...")
        X, y = [], []
        label_map = {kw:i for i,kw in enumerate(keywords_to_train)}

        for kw in keywords_to_train:
            folder = os.path.join(DATASET_DIR, kw.lower())
            if not os.path.exists(folder): continue
            files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            for fname in files:
                try:
                    audio, _ = librosa.load(os.path.join(folder,fname), sr=SAMPLE_RATE, mono=True)
                    X.append(make_spectrogram(audio)); y.append(label_map[kw])
                    aug = audio + 0.004*np.random.randn(len(audio))
                    X.append(make_spectrogram(aug.astype(np.float32))); y.append(label_map[kw])
                except: pass

        if len(X) < 10:
            raise ValueError(f"Not enough data: {len(X)} samples")

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        X, y = sk_shuffle(X, y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15,
                                                           random_state=42, stratify=y)

        train_status.update(progress=40,
            message=f"Training on {len(X_train)} samples ({len(keywords_to_train)} keywords)...")

        inp = keras.Input(shape=(N_MELS,N_TIME,1))
        x = keras.layers.Conv2D(32,(3,3),padding='same',use_bias=False)(inp)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32,(3,3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2,2))(x); x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(64,(3,3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64,(3,3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2,2))(x); x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(128,(3,3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.GlobalAveragePooling2D()(x); x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128,use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x); x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.4)(x)
        out = keras.layers.Dense(len(keywords_to_train), activation='softmax')(x)
        new_model = keras.Model(inp, out)
        new_model.compile(optimizer=keras.optimizers.Adam(3e-3),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        class ProgressCB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pct = 40 + int((epoch+1)/30*50)
                acc = (logs or {}).get('val_accuracy',0)*100
                train_status.update(progress=min(pct,90),
                    message=f"Epoch {epoch+1}/30 — val_acc={acc:.1f}%")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_out = os.path.join(MODEL_DIR, "keyword_model.keras")

        new_model.fit(X_train, y_train, validation_data=(X_val,y_val),
            epochs=30, batch_size=64, verbose=0,
            callbacks=[
                ProgressCB(),
                keras.callbacks.ModelCheckpoint(model_out, monitor='val_accuracy',
                    save_best_only=True, verbose=0),
                keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8,
                    restore_best_weights=True, verbose=0),
            ])

        train_status.update(progress=95, message="Saving model...")
        KEYWORDS = keywords_to_train
        save_keywords(KEYWORDS)
        load_cnn()

        best_val = max(new_model.history.history.get('val_accuracy',[0]))
        train_status.update(running=False, progress=100,
            message=f"✅ Done! Accuracy: {best_val*100:.1f}%",
            last_accuracy=round(best_val*100,1),
            last_trained=datetime.now().isoformat(), error=None)

    except Exception as e:
        import traceback; traceback.print_exc()
        train_status.update(running=False, progress=0,
                            message="❌ Training failed", error=str(e))

# ═══════════════════════════════════════════════════════════════
#  DUAL-ENGINE PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════════
def dual_engine_predict(audio_bytes, speaker='Unknown'):
    """
    The core intelligence:
      1. Try CNN first (fast, for trained keywords)
      2. If CNN not confident enough → use Whisper (any word)
      3. Check result against watchlist → set alert level
    Returns a full result dict.
    """
    audio = bytes_to_audio(audio_bytes)

    cnn_result     = None
    whisper_result = None
    engine_used    = 'none'
    keyword        = 'unknown'
    confidence     = 0.0
    transcript     = ''
    in_watchlist   = False
    in_cnn_list    = False
    all_preds      = {}

    # ── Engine 1: CNN ─────────────────────────────────────────
    if cnn_model is not None:
        spec  = make_spectrogram(audio)
        spec  = np.expand_dims(spec, axis=0)
        preds = cnn_model.predict(spec, verbose=0)[0]
        top   = int(np.argmax(preds))
        cnn_conf = float(preds[top])
        cnn_word = KEYWORDS[top]
        all_preds = {KEYWORDS[i]: float(preds[i]) for i in range(len(KEYWORDS))}

        if cnn_conf >= CNN_THRESHOLD:
            # CNN is confident — trust it
            keyword    = cnn_word
            confidence = cnn_conf
            engine_used = 'CNN'
            in_cnn_list = True
        else:
            cnn_result = {'word': cnn_word, 'conf': cnn_conf}
            # CNN not confident — fall through to Whisper

    # ── Engine 2: Whisper (if CNN failed or not available) ────
    if engine_used != 'CNN':
        if whisper_ok:
            w_result = transcribe_with_whisper(audio_bytes)
            if w_result:
                transcript = w_result.get('text', '').strip()
                w_word, w_conf = extract_whisper_keyword(w_result)
                if w_word:
                    keyword     = w_word
                    confidence  = w_conf
                    engine_used = 'Whisper'
                    whisper_result = w_result
                    # Check if whisper detected a CNN keyword
                    if keyword in KEYWORDS:
                        in_cnn_list = True
                        # Update all_preds to show this
                        if keyword in all_preds:
                            all_preds[keyword] = max(all_preds.get(keyword,0), confidence)
            else:
                # Whisper returned nothing — use low-confidence CNN result
                if cnn_result:
                    keyword     = cnn_result['word']
                    confidence  = cnn_result['conf']
                    engine_used = 'CNN (low confidence)'
        else:
            # No Whisper — just use CNN result even if low confidence
            if cnn_result:
                keyword     = cnn_result['word']
                confidence  = cnn_result['conf']
                engine_used = 'CNN (no Whisper)'

    # ── Check watchlist ───────────────────────────────────────
    if keyword and keyword.lower() in [w.lower() for w in WATCHLIST]:
        in_watchlist = True

    # ── Alert level ───────────────────────────────────────────
    # Higher alert if word is in watchlist OR CNN keyword list
    base_alert = get_alert_level(confidence)
    if in_watchlist:
        alert_level = max(base_alert, 4)  # minimum L4 for watchlist words
    elif in_cnn_list:
        alert_level = base_alert
    else:
        alert_level = min(base_alert, 2)  # unknown words get max L2

    # ── Log to DB ─────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp,keyword,confidence,speaker,alert_level,engine,raw_transcript) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), keyword, confidence, speaker,
         alert_level, engine_used, transcript))
    conn.commit(); conn.close()

    return {
        'keyword':        keyword,
        'confidence':     round(confidence, 3),
        'alert_level':    alert_level,
        'speaker':        speaker,
        'engine':         engine_used,
        'transcript':     transcript,
        'in_watchlist':   in_watchlist,
        'in_keyword_list': in_cnn_list,
        'all_predictions': {k: round(v,3) for k,v in all_preds.items()},
        'timestamp':      datetime.now().isoformat()
    }

# ═══════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/health')
def health():
    return jsonify({
        'status': 'online',
        'cnn_loaded':   cnn_model is not None,
        'whisper_ready': whisper_ok,
        'firebase':     firebase_ok,
        'keywords':     KEYWORDS,
        'watchlist':    WATCHLIST,
        'timestamp':    datetime.now().isoformat()
    })

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'cnn_loaded':   cnn_model is not None,
        'whisper_ready': whisper_ok,
        'firebase_connected': firebase_ok,
        'keywords':     KEYWORDS,
        'watchlist':    WATCHLIST,
        'engines':      {
            'cnn':     'ready' if cnn_model else 'not loaded',
            'whisper': 'ready' if whisper_ok  else 'not installed'
        }
    })

# ── 1. PREDICT (dual-engine) ──────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if cnn_model is None and not whisper_ok:
        return jsonify({'error': 'No prediction engine available. Load model or install Whisper.'}), 503
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_bytes  = request.files['audio'].read()
    speaker_name = request.form.get('speaker', 'Unknown')

    try:
        result = dual_engine_predict(audio_bytes, speaker_name)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── 2. WATCHLIST — add/remove words to monitor (no audio samples needed) ─────
@app.route('/watchlist', methods=['GET'])
def get_watchlist():
    return jsonify({'watchlist': WATCHLIST,
                    'count': len(WATCHLIST),
                    'message': 'Words in this list will trigger alerts when Whisper detects them'})

@app.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """
    Add any word to the watchlist — no audio samples needed!
    Whisper will detect it automatically when someone says it.
    Example: POST {'keyword': 'drum'}
    """
    global WATCHLIST
    data = request.json or {}
    keyword = data.get('keyword', '').strip().lower()
    if not keyword:
        return jsonify({'error': 'keyword required'}), 400
    if keyword not in WATCHLIST:
        WATCHLIST.append(keyword)
        save_watchlist(WATCHLIST)
    return jsonify({
        'status': 'added',
        'keyword': keyword,
        'watchlist': WATCHLIST,
        'message': f'"{keyword}" added to watchlist. Whisper will now detect it automatically — no audio samples needed!'
    })

@app.route('/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    global WATCHLIST
    data = request.json or {}
    keyword = data.get('keyword', '').strip().lower()
    WATCHLIST = [w for w in WATCHLIST if w.lower() != keyword]
    save_watchlist(WATCHLIST)
    return jsonify({'status': 'removed', 'keyword': keyword, 'watchlist': WATCHLIST})

# ── 3. SEARCH keyword in long audio file ─────────────────────────────────────
@app.route('/search_keyword', methods=['POST'])
def search_keyword():
    """
    Search for any keyword inside a long audio file.
    Uses Whisper to transcribe the full audio → then searches for the word in text.
    No model training needed — works for ANY word in ANY language.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    keyword_to_find = request.form.get('keyword', '').strip().lower()

    audio_bytes = request.files['audio'].read()

    try:
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes); tmp_path = tmp.name

        if whisper_ok:
            # ── Whisper approach: transcribe full audio first ──────────
            print(f"[SEARCH] Transcribing with Whisper...")
            result = whisper_model.transcribe(
                tmp_path,
                word_timestamps=True,
                language=None,  # auto-detect
                fp16=False
            )
            full_transcript = result.get('text', '')
            language        = result.get('language', 'unknown')

            # Find all word-level timestamps
            detections = []
            for seg in result.get('segments', []):
                for w in seg.get('words', []):
                    word = w.get('word','').strip().lower()
                    import re
                    word = re.sub(r'[^a-z0-9]', '', word)
                    if not word: continue

                    # Check if this word matches search keyword
                    if not keyword_to_find or word == keyword_to_find or \
                       keyword_to_find in word or word in keyword_to_find:
                        t = w.get('start', 0)
                        detections.append({
                            'word':       word,
                            'time_sec':   round(t, 2),
                            'time_label': f"{int(t//60)}:{int(t%60):02d}",
                            'end_sec':    round(w.get('end', t+0.5), 2),
                            'confidence': round(w.get('probability', 0.8), 3)
                        })

            found = len(detections) > 0
            os.unlink(tmp_path)

            return jsonify({
                'method':           'whisper_word_timestamps',
                'keyword_searched': keyword_to_find or '(all words)',
                'found':            found,
                'count':            len(detections),
                'detections':       detections,
                'full_transcript':  full_transcript,
                'language_detected': language,
                'message': (
                    f'✅ Found "{keyword_to_find}" {len(detections)} time(s)!'
                    if found and keyword_to_find else
                    f'✅ Transcription complete — {len(detections)} words detected'
                    if not keyword_to_find else
                    f'❌ "{keyword_to_find}" not found in this audio.'
                )
            })

        else:
            # ── Fallback: CNN sliding window ──────────────────────────
            full_audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            os.unlink(tmp_path)
            total_dur = len(full_audio)/SAMPLE_RATE
            window    = int(SAMPLE_RATE * DURATION)
            step      = int(SAMPLE_RATE * 0.5)
            detections = []

            if cnn_model is None:
                return jsonify({'error': 'Neither Whisper nor CNN model available'}), 503

            if keyword_to_find and keyword_to_find not in KEYWORDS:
                return jsonify({
                    'error': f'"{keyword_to_find}" not in CNN keywords and Whisper not available.',
                    'tip': 'Install openai-whisper for unlimited keyword search.',
                    'cnn_keywords': KEYWORDS
                }), 400

            for start in range(0, len(full_audio)-window+1, step):
                chunk = full_audio[start:start+window]
                spec  = np.expand_dims(make_spectrogram(chunk), 0)
                preds = cnn_model.predict(spec, verbose=0)[0]
                pred_dict = {KEYWORDS[i]: float(preds[i]) for i in range(len(KEYWORDS))}
                check_kw  = keyword_to_find if keyword_to_find else KEYWORDS[int(np.argmax(preds))]
                conf      = pred_dict.get(check_kw, 0)
                if conf >= 0.65:
                    t = start/SAMPLE_RATE
                    detections.append({'word': check_kw, 'time_sec': round(t,2),
                                       'time_label': f"{int(t//60)}:{int(t%60):02d}",
                                       'confidence': round(conf,3)})

            # Merge nearby
            merged = []
            for d in detections:
                if merged and d['time_sec']-merged[-1]['time_sec'] < 1.0:
                    if d['confidence'] > merged[-1]['confidence']: merged[-1] = d
                else: merged.append(d)

            return jsonify({
                'method': 'cnn_sliding_window',
                'keyword_searched': keyword_to_find or '(all)',
                'found': len(merged)>0, 'count': len(merged),
                'detections': merged,
                'audio_duration_sec': round(total_dur,2),
                'note': 'Install openai-whisper for better accuracy and any-word search'
            })

    except Exception as e:
        import traceback; traceback.print_exc()
        try: os.unlink(tmp_path)
        except: pass
        return jsonify({'error': str(e)}), 500

# ── 4. ADD KEYWORD (with or without audio samples) ───────────────────────────
@app.route('/add_keyword', methods=['POST'])
def add_keyword():
    """
    Add a new keyword.
    - If audio samples provided → upload to Firebase + retrain CNN
    - If NO samples → just add to watchlist (Whisper detects it automatically!)
    """
    global KEYWORDS, WATCHLIST
    keyword = request.form.get('keyword', '').strip().lower()
    if not keyword:
        return jsonify({'error': 'keyword required'}), 400

    files = request.files.getlist('audio')
    local_dir = os.path.join(DATASET_DIR, keyword)
    os.makedirs(local_dir, exist_ok=True)

    saved_count = 0
    for f in files:
        audio_bytes = f.read()
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = f"{keyword}_{ts}.wav"
        with open(os.path.join(local_dir, fname), 'wb') as lf:
            lf.write(audio_bytes)
        upload_to_firebase(audio_bytes, keyword, fname)
        saved_count += 1

    # Always add to watchlist (Whisper will catch it even without CNN training)
    if keyword not in WATCHLIST:
        WATCHLIST.append(keyword)
        save_watchlist(WATCHLIST)

    training_started = False
    if saved_count >= 5 and not train_status['running']:
        # Enough samples — retrain CNN to include this keyword
        if keyword not in KEYWORDS:
            KEYWORDS.append(keyword)
            save_keywords(KEYWORDS)
        t = threading.Thread(target=run_training, args=(list(KEYWORDS),), daemon=True)
        t.start()
        training_started = True
        msg = (f'"{keyword}" added! {saved_count} samples uploaded to Firebase. '
               f'CNN retraining started + Whisper will detect it immediately.')
    elif saved_count > 0:
        msg = (f'"{keyword}" added with {saved_count} samples. '
               f'Need {5-saved_count} more samples to retrain CNN. '
               f'Whisper will detect it right now without retraining!')
    else:
        # No samples at all — just watchlist
        msg = (f'"{keyword}" added to watchlist. '
               f'Whisper will automatically detect it in any audio — no samples needed!')

    return jsonify({
        'status':           'success',
        'keyword':          keyword,
        'samples_uploaded': saved_count,
        'added_to_watchlist': True,
        'cnn_retrain_started': training_started,
        'keywords':         KEYWORDS,
        'watchlist':        WATCHLIST,
        'message':          msg
    })

# ── 5. Train status ───────────────────────────────────────────────────────────
@app.route('/train_status')
def get_train_status():
    return jsonify(train_status)

@app.route('/retrain', methods=['POST'])
def retrain():
    if train_status['running']:
        return jsonify({'error': 'Already training', 'progress': train_status['progress']}), 409
    kws = (request.json or {}).get('keywords', KEYWORDS)
    threading.Thread(target=run_training, args=(list(kws),), daemon=True).start()
    return jsonify({'status': 'started', 'keywords': kws})

# ── 6. Keywords & watchlist info ──────────────────────────────────────────────
@app.route('/keywords')
def get_keywords():
    details = []
    for kw in KEYWORDS:
        folder = os.path.join(DATASET_DIR, kw)
        count  = len([f for f in os.listdir(folder) if f.endswith('.wav')]) \
                 if os.path.exists(folder) else 0
        details.append({'keyword': kw, 'sample_count': count, 'engine': 'CNN'})
    for kw in WATCHLIST:
        if kw not in KEYWORDS:
            details.append({'keyword': kw, 'sample_count': 0, 'engine': 'Whisper'})
    return jsonify({'keywords': KEYWORDS, 'watchlist': WATCHLIST, 'details': details})

# ── 7. Speaker enroll / verify ────────────────────────────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Need audio and name'}), 400
    name  = request.form['name'].strip()
    audio = bytes_to_audio(request.files['audio'].read())
    mfcc  = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).mean(axis=1).tolist()
    conn  = sqlite3.connect(DB_PATH)
    row   = conn.execute("SELECT embedding FROM speakers WHERE name=?", (name,)).fetchone()
    if row:
        merged = ((np.array(json.loads(row[0]))+np.array(mfcc))/2).tolist()
        conn.execute("UPDATE speakers SET embedding=? WHERE name=?", (json.dumps(merged), name))
    else:
        conn.execute("INSERT INTO speakers VALUES (?,?,?)",
                     (name, json.dumps(mfcc), datetime.now().isoformat()))
    conn.commit(); conn.close()
    return jsonify({'status': 'enrolled', 'speaker': name})

@app.route('/verify', methods=['POST'])
def verify():
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Need audio and name'}), 400
    name  = request.form['name'].strip()
    audio = bytes_to_audio(request.files['audio'].read())
    mfcc  = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).mean(axis=1).tolist()
    conn  = sqlite3.connect(DB_PATH)
    row   = conn.execute("SELECT embedding FROM speakers WHERE name=?", (name,)).fetchone()
    conn.close()
    if not row: return jsonify({'verified': False, 'reason': 'Not enrolled'})
    sim = cosine_sim(json.loads(row[0]), mfcc)
    return jsonify({'verified': sim>0.85, 'similarity': round(sim,3), 'speaker': name})

@app.route('/speakers')
def speakers():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name,enrolled_at FROM speakers").fetchall()
    conn.close()
    return jsonify({'speakers': [{'name':r[0],'enrolled_at':r[1]} for r in rows]})

# ── 8. Analytics & history ────────────────────────────────────────────────────
@app.route('/analytics')
def analytics():
    conn   = sqlite3.connect(DB_PATH)
    stats  = conn.execute("SELECT keyword,COUNT(*),AVG(confidence) FROM predictions GROUP BY keyword").fetchall()
    recent = conn.execute("SELECT timestamp,keyword,confidence,speaker,alert_level,engine FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return jsonify({
        'stats':  [{'keyword':r[0],'count':r[1],'avg_confidence':round(r[2],3)} for r in stats],
        'recent': [{'timestamp':r[0],'keyword':r[1],'confidence':round(r[2],3),
                    'speaker':r[3],'alert_level':r[4],'engine':r[5]} for r in recent]
    })

@app.route('/history')
def history():
    limit = int(request.args.get('limit', 50))
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT timestamp,keyword,confidence,speaker,alert_level,engine,raw_transcript "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return jsonify({'history': [
        {'timestamp':r[0],'keyword':r[1],'confidence':round(r[2],3),
         'speaker':r[3],'alert_level':r[4],'engine':r[5],'transcript':r[6]}
        for r in rows]})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("VOICESPOT CLOUD BACKEND v4.0")
    print(f"CNN     : {'✅ Loaded' if cnn_model else '❌ Not found'}")
    print(f"Whisper : {'✅ Ready' if whisper_ok else '❌ Not installed'}")
    print(f"Firebase: {'✅ Connected' if firebase_ok else '⚠️  Disabled'}")
    print(f"Keywords: {KEYWORDS}")
    print(f"Watchlist: {WATCHLIST}")
    print("="*60)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
