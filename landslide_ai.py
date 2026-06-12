# =============================================
# landslide_ai.py
# AI Module - Hệ thống cảnh báo sạt lở đất
# Tác giả: Võ Văn Hiếu - MSSV: 22139021
# Mô hình: Ensemble RF+SVM+XGB (Soft Voting)
#
# Chạy: python3 landslide_ai.py
# Cài:  pip3 install scikit-learn imbalanced-learn
#            xgboost pandas matplotlib seaborn
# =============================================

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    roc_auc_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ══════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════
DB_PATH        = "landslide.db"
CSV_PATH       = "train_data.csv"
LABEL_NAMES    = ['AN TOAN', 'CANH BAO', 'NGUY HIEM']
RANDOM_STATE   = 42
MODEL_SAVE     = "rf_model.pkl"      # model phan loai hien tai
MODEL_SAVE_5P  = "rf_model_5p.pkl"   # model du bao som 5 phut
FORECAST_STEPS = 1   # 1 buoc ~ 5 phut

# ══════════════════════════════════════════════
# 1. ĐỌC DỮ LIỆU
# ══════════════════════════════════════════════
def load_data():
    print("\n" + "="*55)
    print("  BUOC 1: DOC DU LIEU")
    print("="*55)
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM sensor_data ORDER BY timestamp ASC",
            conn)
        conn.close()
        print(f"  Nguon: SQLite ({DB_PATH})")
    except:
        df = pd.read_csv(CSV_PATH)
        print(f"  Nguon: CSV ({CSV_PATH})")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'label' not in df.columns:
        df['label'] = df['alert']

    print(f"  Tong mau : {len(df):,}")
    print(f"  So node  : {df['node_id'].nunique()}")
    print(f"\n  Phan phoi nhan:")
    for lbl, name in enumerate(LABEL_NAMES):
        cnt = (df['label'] == lbl).sum()
        print(f"    {name:<12}: {cnt:>6} ({cnt/len(df)*100:.1f}%)")
    return df

# ══════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ══════════════════════════════════════════════
def preprocess(df, forecast_steps=0):
    print("\n" + "="*55)
    title = f"DU BAO SOM {forecast_steps*5} PHUT" if forecast_steps > 0 else "HIEN TAI"
    print(f"  BUOC 2: TIEN XU LY ({title})")
    print("="*55)

    df = df.copy()
    features = ['tilt', 'pitch', 'roll', 'j2', 'j3', 'rain']

    df['tilt_abs']      = df['tilt'].abs()
    df['roll_abs']      = df['roll'].abs()
    df['moisture_avg']  = (df['j2'] + df['j3']) / 2
    df['moisture_diff'] = (df['j2'] - df['j3']).abs()
    df['tilt_moisture'] = df['tilt_abs'] * df['moisture_avg']

    features_ext = features + [
        'tilt_abs', 'roll_abs',
        'moisture_avg', 'moisture_diff', 'tilt_moisture'
    ]
    df[features_ext] = df[features_ext].fillna(
        df[features_ext].median())

    if forecast_steps > 0:
        df['y'] = df['label'].shift(-forecast_steps)
        df = df.dropna(subset=['y'])
        df['y'] = df['y'].astype(int)
        print(f"  Label dich truoc {forecast_steps} buoc (~{forecast_steps*5} phut)")
    else:
        df['y'] = df['label']

    X = df[features_ext].values
    y = df['y'].values

    print(f"  So features : {len(features_ext)}")
    print(f"  So mau      : {len(X):,}")
    return X, y, features_ext

# ══════════════════════════════════════════════
# 3. SMOTE
# ══════════════════════════════════════════════
def apply_smote(X_train, y_train):
    print("\n  [SMOTE] Can bang du lieu...")
    print(f"    Truoc: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"    Sau  : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

# ══════════════════════════════════════════════
# 4. ENSEMBLE RF + SVM + XGB
# ══════════════════════════════════════════════
def train_ensemble(X_train, X_test, y_train, y_test,
                   title_suffix="Hien tai"):
    print(f"\n  [ENSEMBLE] RF + SVM + XGBoost — {title_suffix}")

    # Random Forest
    print("  [RF] Huan luyen...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)

    # SVM
    print("  [SVM] Huan luyen...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    svm = SVC(kernel='rbf', C=10, class_weight='balanced',
              probability=True, random_state=RANDOM_STATE)
    svm.fit(X_tr_s, y_train)
    svm_prob = svm.predict_proba(X_te_s)

    # XGBoost
    print("  [XGB] Huan luyen...")
    from collections import Counter
    counts = Counter(y_train)
    total  = len(y_train)
    scale  = {c: total / (len(counts) * v)
              for c, v in counts.items()}
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train,
            sample_weight=[scale[y] for y in y_train])
    xgb_prob = xgb.predict_proba(X_test)

    # Soft Voting
    ensemble_prob = (rf_prob + svm_prob + xgb_prob) / 3
    y_pred = np.argmax(ensemble_prob, axis=1)

    # Đánh giá
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    try:
        auc_score = roc_auc_score(
            y_test, ensemble_prob,
            multi_class='ovr', average='macro')
    except:
        auc_score = 0.0

    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"  F1 macro : {f1*100:.2f}%")
    print(f"  AUC      : {auc_score:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.title(f'Confusion Matrix — Ensemble {title_suffix}')
    plt.ylabel('Thuc te')
    plt.xlabel('Du doan')
    plt.tight_layout()
    fname = f"cm_ensemble_{title_suffix.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Luu] {fname}")

    # ROC
    plt.figure(figsize=(7, 5))
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), ensemble_prob.ravel())
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2,
             label=f'Ensemble RF+SVM+XGB (AUC={auc_val:.3f})')
    plt.plot([0,1],[0,1],'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — Ensemble {title_suffix}')
    plt.legend()
    plt.tight_layout()
    fname = f"roc_ensemble_{title_suffix.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Luu] {fname}")

    return rf, svm, xgb, scaler, acc, f1

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "╔"+"═"*53+"╗")
    print("║   AI MODULE — HE THONG CANH BAO SAT LO DAT     ║")
    print("║   Vo Van Hieu — MSSV: 22139021                 ║")
    print("║   Mo hinh: Ensemble RF + SVM + XGBoost         ║")
    print("╚"+"═"*53+"╝")

    df = load_data()
    if df is None:
        return

    # ── PHẦN A: Phân loại hiện tại ────────────
    print("\n\n" + "▓"*55)
    print("  PHAN A: PHAN LOAI TRANG THAI HIEN TAI")
    print("▓"*55)

    X, y, feat = preprocess(df, forecast_steps=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    rf, svm, xgb, scaler, acc, f1 = train_ensemble(
        X_train_sm, X_test, y_train_sm, y_test,
        title_suffix="Hien tai")

    # Lưu model hiện tại
    model_dict = {
        'model'   : (rf, svm, xgb),
        'scaler'  : scaler,
        'type'    : 'ensemble',
        'features': feat
    }
    with open(MODEL_SAVE, 'wb') as f_out:
        pickle.dump(model_dict, f_out)
    print(f"\n  [MODEL] Da luu: {MODEL_SAVE}")

    # ── PHẦN B: Dự báo sớm 5 phút ─────────────
    print("\n\n" + "▓"*55)
    print(f"  PHAN B: DU BAO SOM {FORECAST_STEPS*5} PHUT TRUOC")
    print("▓"*55)

    X_f, y_f, _ = preprocess(df, forecast_steps=FORECAST_STEPS)

    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_f, y_f, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_f)
    X_tr_f_sm, y_tr_f_sm = apply_smote(X_tr_f, y_tr_f)

    rf_f, svm_f, xgb_f, scaler_f, acc_f, f1_f = train_ensemble(
        X_tr_f_sm, X_te_f, y_tr_f_sm, y_te_f,
        title_suffix=f"Du_bao_{FORECAST_STEPS*5}p")

    # Luu model du bao 5p
    model_dict_5p = {
        'model'   : (rf_f, svm_f, xgb_f),
        'scaler'  : scaler_f,
        'type'    : 'ensemble',
        'features': feat
    }
    with open(MODEL_SAVE_5P, 'wb') as f_out:
        pickle.dump(model_dict_5p, f_out)
    print(f"  [MODEL] Da luu: {MODEL_SAVE_5P}")

    # ── TỔNG KẾT ──────────────────────────────
    print(f"\n{'='*55}")
    print(f"  TONG KET")
    print(f"{'='*55}")
    print(f"  [Hien tai]")
    print(f"    Accuracy: {acc*100:.2f}%")
    print(f"    F1 macro: {f1*100:.2f}%")
    print(f"  [Du bao {FORECAST_STEPS*5} phut]")
    print(f"    Accuracy: {acc_f*100:.2f}%")
    print(f"    F1 macro: {f1_f*100:.2f}%")
    print(f"  Model luu : {MODEL_SAVE} + {MODEL_SAVE_5P}")
    print(f"\n  HOAN THANH!\n")

# ══════════════════════════════════════════════
# PREDICT REAL-TIME
# ══════════════════════════════════════════════
def predict_realtime(model_dict, tilt, pitch, roll,
                     j2, j3, rain):
    tilt_abs      = abs(tilt)
    roll_abs      = abs(roll)
    moisture_avg  = (j2 + j3) / 2
    moisture_diff = abs(j2 - j3)
    tilt_moisture = tilt_abs * moisture_avg

    X = np.array([[tilt, pitch, roll, j2, j3, rain,
                   tilt_abs, roll_abs, moisture_avg,
                   moisture_diff, tilt_moisture]])

    rf, svm, xgb = model_dict['model']
    scaler = model_dict['scaler']
    X_s = scaler.transform(X)

    prob = (rf.predict_proba(X)[0] +
            svm.predict_proba(X_s)[0] +
            xgb.predict_proba(X)[0]) / 3

    label = int(np.argmax(prob))
    conf  = float(prob[label]) * 100
    names = ['AN TOAN', 'CANH BAO', 'NGUY HIEM']
    return label, names[label], round(conf, 1)

if __name__ == "__main__":
    main()
