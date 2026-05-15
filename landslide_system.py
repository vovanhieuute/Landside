"""
=================================================================
  LANDSLIDE PREDICTION SYSTEM
  File: landslide_system.py
  Chạy trên: Raspberry Pi 4
  
  Gồm 3 phần:
  1. TRAIN   — train RF/LSTM/SVM, export model
  2. PREDICT — load model, đọc Serial, dự đoán realtime
  3. DEMO    — test không cần ESP32 (dùng dữ liệu giả)
  
  Cách dùng:
    python3 landslide_system.py train    # train và lưu model
    python3 landslide_system.py predict  # chạy dự đoán realtime
    python3 landslide_system.py demo     # test nhanh không cần ESP32
=================================================================
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Đường dẫn lưu model ──────────────────────────────────────────
MODEL_DIR  = os.path.expanduser("~/landslide_models")
DATA_FILE  = os.path.expanduser("~/landslide_data.csv")
SERIAL_PORT = "/dev/ttyUSB0"   # đổi nếu cần: ls /dev/tty*
BAUD_RATE   = 115200

LABELS = {0: "AN TOAN", 1: "CANH BAO", 2: "NGUY HIEM!"}
FEATURES = ["vmc_surface", "vmc_deep", "pitch", "roll"]

os.makedirs(MODEL_DIR, exist_ok=True)


# =================================================================
#  PHẦN 1 — TRAIN
# =================================================================

def generate_data(n=3000):
    """Tạo dữ liệu mô phỏng nếu chưa có data thực."""
    rng = np.random.default_rng(42)
    n3  = n // 3

    # Label 0 — an toàn
    vs0 = rng.uniform(0.15, 0.28, n3)
    vd0 = vs0 + rng.uniform(0.02, 0.06, n3)
    p0  = rng.uniform(-0.4, 0.4, n3)
    r0  = rng.uniform(-0.3, 0.3, n3)

    # Label 1 — cảnh báo nhỏ
    vs1 = rng.uniform(0.28, 0.34, n3)
    vd1 = vs1 + rng.uniform(0.03, 0.07, n3)
    p1  = rng.choice([-1,1], n3) * rng.uniform(0.5, 1.1, n3)
    r1  = rng.uniform(-0.6, 0.6, n3)

    # Label 2 — nguy hiểm
    vs2 = rng.uniform(0.34, 0.48, n3)
    vd2 = vs2 + rng.uniform(0.02, 0.05, n3)
    p2  = rng.choice([-1,1], n3) * rng.uniform(1.1, 3.1, n3)
    r2  = rng.uniform(-1.2, 1.2, n3)

    X = np.column_stack([
        np.concatenate([vs0, vs1, vs2]) + rng.normal(0, 0.008, n),
        np.concatenate([vd0, vd1, vd2]) + rng.normal(0, 0.006, n),
        np.concatenate([p0,  p1,  p2])  + rng.normal(0, 0.05,  n),
        np.concatenate([r0,  r1,  r2])  + rng.normal(0, 0.04,  n),
    ]).clip(-5, 5).astype(np.float32)

    y = np.array([0]*n3 + [1]*n3 + [2]*n3)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def load_real_data():
    """Load dữ liệu thực từ CSV.
    
    CSV cần có cột: vmc_surface, vmc_deep, pitch, roll, label
    """
    if not os.path.exists(DATA_FILE):
        print(f"[!] Không tìm thấy {DATA_FILE} — dùng dữ liệu mô phỏng")
        return None, None
    df = pd.read_csv(DATA_FILE).dropna()
    print(f"[OK] Đọc được {len(df)} mẫu từ {DATA_FILE}")
    return df[FEATURES].values.astype(np.float32), df["label"].values.astype(int)


def train_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    print("\n--- Train Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    path = os.path.join(MODEL_DIR, "model_rf.pkl")
    joblib.dump(rf, path)
    print(f"  Saved: {path}")
    print(f"  Feature importance: {dict(zip(FEATURES, rf.feature_importances_.round(3)))}")
    return rf


def train_svm(X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("\n--- Train SVM ---")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    svm    = SVC(kernel="rbf", class_weight="balanced",
                 probability=True, random_state=42)
    svm.fit(X_sc, y_train)

    joblib.dump(svm,    os.path.join(MODEL_DIR, "model_svm.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_svm.pkl"))
    print(f"  Saved: model_svm.pkl + scaler_svm.pkl")
    return svm, scaler


def train_lstm(X_train, y_train, X_val, y_val):
    """Train LSTM và export TFLite để chạy nhanh trên RPi."""
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        print("  [!] TensorFlow chưa cài — bỏ qua LSTM")
        print("      Cài: pip3 install tensorflow")
        return None, None

    print("\n--- Train LSTM ---")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train).reshape(-1, 1, 4).astype(np.float32)
    X_v_sc  = scaler.transform(X_val).reshape(-1, 1, 4).astype(np.float32)
    y_tr    = tf.keras.utils.to_categorical(y_train, 3)
    y_v     = tf.keras.utils.to_categorical(y_val,   3)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(1, 4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3,  activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    es = tf.keras.callbacks.EarlyStopping(patience=5,
                                          restore_best_weights=True,
                                          verbose=0)
    model.fit(X_tr_sc, y_tr,
              validation_data=(X_v_sc, y_v),
              epochs=30, batch_size=128,
              callbacks=[es], verbose=1)

    # Export TFLite — nhanh hơn nhiều trên RPi
    converter   = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path  = os.path.join(MODEL_DIR, "model_lstm.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
    print(f"  Saved: model_lstm.tflite + scaler_lstm.pkl")
    return model, scaler


def evaluate(name, model, X_test, y_test, scaler=None, is_lstm=False):
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    if is_lstm:
        import tflite_runtime.interpreter as tflite
        import joblib
        interp = tflite.Interpreter(
            model_path=os.path.join(MODEL_DIR, "model_lstm.tflite"))
        interp.allocate_tensors()
        inp = interp.get_input_details()
        out = interp.get_output_details()
        sc  = joblib.load(os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
        X_sc = sc.transform(X_test).reshape(-1, 1, 4).astype(np.float32)
        y_pred = []
        for i in range(len(X_sc)):
            interp.set_tensor(inp[0]["index"], X_sc[i:i+1])
            interp.invoke()
            y_pred.append(np.argmax(interp.get_tensor(out[0]["index"])))
        y_pred = np.array(y_pred)
    elif scaler is not None:
        X_sc   = scaler.transform(X_test)
        y_pred = model.predict(X_sc)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 macro : {f1:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["An toan","Canh bao","Nguy hiem"],
          zero_division=0))
    return acc, f1


def run_train():
    from sklearn.model_selection import train_test_split

    print("=" * 55)
    print("  TRAIN RF / LSTM / SVM")
    print("=" * 55)

    # 1. Load hoặc tạo data
    X, y = load_real_data()
    if X is None:
        print("[*] Tạo dữ liệu mô phỏng 3000 mẫu...")
        X, y = generate_data(3000)

    # 2. Chia tập
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    print(f"\nTrain={len(X_tr)} Val={len(X_val)} Test={len(X_te)}")
    print(f"Phân phối train: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    # 3. Train
    rf             = train_rf(X_tr, y_tr)
    svm, scaler_svm = train_svm(X_tr, y_tr)
    train_lstm(X_tr, y_tr, X_val, y_val)

    # 4. Đánh giá
    print("\n" + "="*55)
    print("  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*55)
    evaluate("Random Forest", rf, X_te, y_te)
    evaluate("SVM",           svm, X_te, y_te, scaler=scaler_svm)
    try:
        evaluate("LSTM (TFLite)", None, X_te, y_te, is_lstm=True)
    except Exception as e:
        print(f"  [!] Không đánh giá được LSTM: {e}")

    print("\n[OK] Train xong! Models lưu tại:", MODEL_DIR)


# =================================================================
#  PHẦN 2 — PREDICT (đọc từ ESP32 qua Serial)
# =================================================================

class Predictor:
    """Load 3 model một lần, gọi predict() liên tục."""

    def __init__(self):
        import joblib
        print("[*] Đang load models...")

        # RF
        self.rf = joblib.load(os.path.join(MODEL_DIR, "model_rf.pkl"))
        print("  RF loaded")

        # SVM
        self.svm        = joblib.load(os.path.join(MODEL_DIR, "model_svm.pkl"))
        self.scaler_svm = joblib.load(os.path.join(MODEL_DIR, "scaler_svm.pkl"))
        print("  SVM loaded")

        # LSTM TFLite
        try:
            import tflite_runtime.interpreter as tflite
            self.interp = tflite.Interpreter(
                model_path=os.path.join(MODEL_DIR, "model_lstm.tflite"))
            self.interp.allocate_tensors()
            self.inp    = self.interp.get_input_details()
            self.out    = self.interp.get_output_details()
            import joblib as jb
            self.scaler_lstm = jb.load(
                os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
            self.lstm_ok = True
            print("  LSTM TFLite loaded")
        except Exception as e:
            self.lstm_ok = False
            print(f"  LSTM skip: {e}")

        print("[OK] Sẵn sàng dự đoán!\n")

    def predict_one(self, vmc_s, vmc_d, pitch, roll):
        """
        Trả về: (final_label, detail_dict)
        """
        X = np.array([[vmc_s, vmc_d, pitch, roll]], dtype=np.float32)

        # RF
        lrf  = int(self.rf.predict(X)[0])
        prf  = float(self.rf.predict_proba(X)[0][lrf])

        # SVM
        X_sc = self.scaler_svm.transform(X)
        lsvm = int(self.svm.predict(X_sc)[0])
        psvm = float(self.svm.predict_proba(X_sc)[0][lsvm])

        # LSTM
        llstm, plstm = lrf, prf  # fallback = RF nếu LSTM không load được
        if self.lstm_ok:
            X_lt = self.scaler_lstm.transform(X).reshape(1,1,4).astype(np.float32)
            self.interp.set_tensor(self.inp[0]["index"], X_lt)
            self.interp.invoke()
            prob  = self.interp.get_tensor(self.out[0]["index"])[0]
            llstm = int(np.argmax(prob))
            plstm = float(prob[llstm])

        # Ensemble voting (đa số phiếu)
        votes  = [lrf, lsvm, llstm]
        final  = max(set(votes), key=votes.count)

        return final, {
            "rf":   (lrf,   round(prf,   2)),
            "svm":  (lsvm,  round(psvm,  2)),
            "lstm": (llstm, round(plstm, 2)),
        }


def send_gmail_alert(node_id, tilt, j2, j3, rain, final):
    """Gui email canh bao qua Gmail SMTP."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    GMAIL_USER = "your_email@gmail.com"   # <- doi email cua ban
    GMAIL_PASS = "your_app_password"       # <- doi App Password Gmail
    ALERT_TO   = ["recipient@gmail.com"]   # <- doi email nhan

    levels = {0: "AN TOAN", 1: "CANH BAO", 2: "NGUY HIEM!"}
    subject = f"[SAT LO] {levels[final]} - {node_id} - {time.strftime('%d/%m %H:%M')}"
    body = (
        f"=== CANH BAO SAT LO DAT ===\n\n"
        f"Node        : {node_id}\n"
        f"Muc canh bao: {levels[final]}\n"
        f"Goc nghieng : {tilt:.1f} do\n"
        f"Do am mat   : {j2:.0f}%\n"
        f"Do am sau   : {j3:.0f}%\n"
        f"Mua         : {'Co' if rain else 'Khong'}\n"
        f"Thoi gian   : {time.strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        f"Vui long kiem tra khu vuc ngay!"
    )
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From']    = GMAIL_USER
        msg['To']      = ', '.join(ALERT_TO)
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, ALERT_TO, msg.as_string())
        print(f"  [EMAIL] Da gui canh bao!")
    except Exception as e:
        print(f"  [EMAIL] Loi: {e}")


def run_predict():
    import serial

    p = Predictor()

    print(f"[*] Ket noi Serial {SERIAL_PORT} @ {BAUD_RATE}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"[!] Khong mo duoc cong Serial: {e}")
        print(f"    Kiem tra: ls /dev/tty*")
        return

    time.sleep(2)
    print("[OK] Serial ket noi!\n")
    print("=" * 60)
    print("  DANG GIAM SAT — nhan Ctrl+C de dung")
    print("=" * 60)

    node_states     = {}
    last_alert_time = {}
    ALERT_COOLDOWN  = 600  # 10 phut

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("{"):
                continue

            data     = json.loads(line)
            msg_type = data.get("type", "")

            # ── Nhan du lieu tu Gateway ──────────────────────────
            if msg_type == "data":
                raw   = data.get("raw", "")
                parts = raw.split(",")
                if len(parts) < 6:
                    continue

                # Format: N01,tilt,j2%,j3%,rain,alert
                node_id    = parts[0]
                tilt       = float(parts[1])
                j2         = float(parts[2])
                j3         = float(parts[3])
                rain       = int(parts[4])
                alert_node = int(parts[5])

                # Chuyen sang dac trung AI (0.0-1.0)
                vmc_s = j2 / 100.0
                vmc_d = j3 / 100.0
                pitch = tilt
                roll  = 0.0

                # Du doan AI
                t0 = time.time()
                final, detail = p.predict_one(vmc_s, vmc_d, pitch, roll)
                ms = (time.time() - t0) * 1000

                # Luu trang thai node
                node_states[node_id] = {
                    "tilt": tilt, "j2": j2, "j3": j3,
                    "rain": rain, "alert_node": alert_node,
                    "alert_ai": final,
                    "time": time.strftime('%H:%M:%S')
                }

                # In ket qua
                status   = LABELS[final]
                rain_str = "Co mua" if rain else "Khong mua"
                print(f"\n[{node_id}] {time.strftime('%H:%M:%S')}")
                print(f"  VMC_mat={vmc_s:.3f}  VMC_sau={vmc_d:.3f}  "
                      f"Pitch={pitch:.2f}  Roll={roll:.2f}")
                print(f"  RF={detail['rf'][0]}({detail['rf'][1]*100:.0f}%)  "
                      f"SVM={detail['svm'][0]}({detail['svm'][1]*100:.0f}%)  "
                      f"LSTM={detail['lstm'][0]}({detail['lstm'][1]*100:.0f}%)")
                print(f"  Node={alert_node}  AI={final}"
                      f"  >>> {status}  [{ms:.0f}ms]")
                print(f"  Tilt={tilt:.1f} J2={j2:.0f}% J3={j3:.0f}% {rain_str}")

                # Gui Gmail neu can canh bao
                now  = time.time()
                last = last_alert_time.get(node_id, 0)
                if (final >= 1 or alert_node >= 1) and \
                   (now - last > ALERT_COOLDOWN):
                    print(f"  !!! GUI GMAIL CANH BAO !!!")
                    send_gmail_alert(node_id, tilt, j2, j3, rain, final)
                    last_alert_time[node_id] = now

            elif msg_type == "heartbeat":
                print(f"[HB] Gateway hoat dong | count={data.get('count',0)}")

            elif msg_type == "timeout":
                print(f"[!] {data.get('node','?')} MAT KET NOI!")

            elif msg_type == "status":
                n1 = "ON" if data.get("n1") else "OFF"
                n2 = "ON" if data.get("n2") else "OFF"
                n3 = "ON" if data.get("n3") else "OFF"
                print(f"[STATUS] N1={n1} N2={n2} N3={n3} "
                      f"Alert={data.get('globalAlert',0)}")

        except json.JSONDecodeError:
            pass
        except KeyboardInterrupt:
            print("\n[*] Dung giam sat.")
            ser.close()
            break
        except Exception as e:
            print(f"  Loi: {e}")
            time.sleep(1)


# =================================================================
#  PHẦN 3 — DEMO (test không cần ESP32)
# =================================================================

def run_demo():
    """Test nhanh toàn bộ pipeline không cần ESP32."""

    # Tự train nếu chưa có model
    rf_path = os.path.join(MODEL_DIR, "model_rf.pkl")
    if not os.path.exists(rf_path):
        print("[*] Chưa có model — tự train trước...")
        run_train()

    p = Predictor()

    # Các tình huống test
    scenarios = [
        (0.18, 0.22,  0.2,  0.1, "Dat kho, troi nang"),
        (0.31, 0.36,  0.7,  0.3, "Dat am, mua nhe"),
        (0.38, 0.41,  0.95, 0.4, "Dat uot, mua vua"),
        (0.43, 0.46,  1.8,  0.7, "Dat rat uot, sap sat lo"),
        (0.46, 0.49,  2.5,  1.1, "DAT BAO HOA - SAT LO!"),
    ]

    print("=" * 60)
    print("  DEMO DU DOAN — 5 TINH HUONG")
    print("=" * 60)

    for vs, vd, pt, ro, mo_ta in scenarios:
        final, detail = p.predict_one(vs, vd, pt, ro)
        print(f"\n[{mo_ta}]")
        print(f"  VMC_mat={vs}  VMC_sau={vd}  Pitch={pt}°  Roll={ro}°")
        print(f"  RF={detail['rf'][0]}  SVM={detail['svm'][0]}  "
              f"LSTM={detail['lstm'][0]}")
        print(f"  >>> KET QUA CUOI: {LABELS[final]}")

    # Mô phỏng stream dữ liệu liên tục
    print("\n" + "=" * 60)
    print("  MO PHONG STREAM DU LIEU (5 giay) — nhấn Ctrl+C de dung")
    print("=" * 60)

    import random
    try:
        step = 0
        while True:
            # Tăng dần VMC và pitch để giả lập mưa
            t   = (step % 30) / 30.0
            vs  = round(0.15 + t * 0.35 + random.gauss(0, 0.01), 3)
            vd  = round(vs   + 0.04 + random.gauss(0, 0.008), 3)
            pt  = round(t * 2.8 * random.choice([-1, 1])
                        + random.gauss(0, 0.05), 3)
            ro  = round(random.gauss(0, 0.2), 3)

            final, detail = p.predict_one(vs, vd, pt, ro)
            marker = "***" if final >= 2 else (">>>" if final == 1 else "   ")
            print(f"  {marker} VMC={vs:.3f}/{vd:.3f} Pitch={pt:.2f} "
                  f"=> {LABELS[final]}")

            step += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Demo kết thúc.")


# =================================================================
#  MAIN
# =================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "train":
        run_train()

    elif mode == "predict":
        # Kiểm tra model đã có chưa
        if not os.path.exists(os.path.join(MODEL_DIR, "model_rf.pkl")):
            print("[!] Chưa có model — chạy train trước:")
            print("    python3 landslide_system.py train")
            sys.exit(1)
        run_predict()

    elif mode == "demo":
        run_demo()

    else:
        print("Dùng: python3 landslide_system.py [train|predict|demo]")
        print("  train   — train RF/LSTM/SVM và lưu model")
        print("  predict — chạy dự đoán realtime từ ESP32 Serial")
        print("  demo    — test không cần ESP32")
