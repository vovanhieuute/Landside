# =============================================
# landslide_ai.py
# AI Module - Hệ thống cảnh báo sạt lở đất
# Tác giả: Võ Văn Hiếu - MSSV: 22139021
# Mô hình: XGBoost
#
# Chạy: python3 landslide_ai.py
# Cài:  pip3 install xgboost scikit-learn
#            imbalanced-learn pandas matplotlib seaborn
# =============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing  import label_binarize
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
CSV_PATH       = "train_data_24k.csv"
LABEL_NAMES    = ['AN TOAN', 'CANH BAO', 'NGUY HIEM']
RANDOM_STATE   = 42
MODEL_SAVE     = "rf_model.pkl"
MODEL_SAVE_5P  = "rf_model_5p.pkl"
FORECAST_STEPS = 33  # 33 bước × ~9.5s ≈ 313s ≈ 5 phút thực sự

# ══════════════════════════════════════════════
# 1. ĐỌC DỮ LIỆU TỪ CSV
# ══════════════════════════════════════════════
def load_data():
    print("\n" + "="*55)
    print("  BUOC 1: DOC DU LIEU")
    print("="*55)

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

    # Features cơ bản
    df['tilt_abs']      = df['tilt'].abs()
    df['roll_abs']      = df['roll'].abs()
    df['moisture_avg']  = (df['j2'] + df['j3']) / 2
    df['moisture_diff'] = (df['j2'] - df['j3']).abs()
    df['tilt_moisture'] = df['tilt_abs'] * df['moisture_avg']

    # Features xu hướng — cải tiến cho dự báo sớm
    df['tilt_diff']     = df['tilt'].diff().fillna(0)   # tilt đang tăng/giảm
    df['j2_diff']       = df['j2'].diff().fillna(0)     # VMC j2 tăng/giảm
    df['j3_diff']       = df['j3'].diff().fillna(0)     # VMC j3 tăng/giảm
    df['rain_count']    = df['rain'].rolling(3, min_periods=1).sum()  # mưa liên tiếp
    df['tilt_trend']    = df['tilt'].rolling(3, min_periods=1).mean() # xu hướng tilt
    df['moisture_trend']= df['moisture_avg'].rolling(3, min_periods=1).mean()

    features_ext = features + [
        'tilt_abs', 'roll_abs',
        'moisture_avg', 'moisture_diff', 'tilt_moisture',
        'tilt_diff', 'j2_diff', 'j3_diff',
        'rain_count', 'tilt_trend', 'moisture_trend'
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
    counts = np.bincount(y_train)
    ratio  = counts.min() / counts.max()
    if ratio >= 0.8:
        print("\n  [SMOTE] Data da can bang, bo qua SMOTE")
        return X_train, y_train
    print("\n  [SMOTE] Can bang du lieu...")
    print(f"    Truoc: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"    Sau  : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

# ══════════════════════════════════════════════
# 4. XGBOOST
# ══════════════════════════════════════════════
def train_xgboost(X_train, X_test, y_train, y_test,
                  title_suffix="Hien tai"):
    print(f"\n  [XGB] Huan luyen XGBoost — {title_suffix}")

    from collections import Counter
    counts = Counter(y_train)
    total  = len(y_train)
    scale  = {c: total / (len(counts) * v)
              for c, v in counts.items()}

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train,
            sample_weight=[scale[y] for y in y_train])

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    try:
        auc_score = roc_auc_score(
            y_test, y_prob,
            multi_class='ovr', average='macro')
    except:
        auc_score = 0.0

    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"  F1 macro : {f1*100:.2f}%")
    print(f"  AUC      : {auc_score:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES, digits=4))

    # Feature Importance
    feat_names = ['tilt','pitch','roll','j2','j3','rain',
                  'tilt_abs','roll_abs','moisture_avg',
                  'moisture_diff','tilt_moisture',
                  'tilt_diff','j2_diff','j3_diff',
                  'rain_count','tilt_trend','moisture_trend']
    print("  Feature Importance:")
    for name, imp in sorted(
            zip(feat_names, xgb.feature_importances_),
            key=lambda x: -x[1]):
        print(f"    {name:<15}: {imp*100:.1f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.title(f'Confusion Matrix — XGBoost {title_suffix}')
    plt.ylabel('Thuc te')
    plt.xlabel('Du doan')
    plt.tight_layout()
    fname = f"cm_xgb_{title_suffix.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Luu] {fname}")

    # ROC Curve
    plt.figure(figsize=(7, 5))
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2,
             label=f'XGBoost (AUC={auc_val:.3f})')
    plt.plot([0,1],[0,1],'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — XGBoost {title_suffix}')
    plt.legend()
    plt.tight_layout()
    fname = f"roc_xgb_{title_suffix.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Luu] {fname}")

    return xgb, acc, f1

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "╔"+"═"*53+"╗")
    print("║   AI MODULE — HE THONG CANH BAO SAT LO DAT     ║")
    print("║   Vo Van Hieu — MSSV: 22139021                 ║")
    print("║   Mo hinh: XGBoost                             ║")
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

    xgb, acc, f1 = train_xgboost(
        X_train_sm, X_test, y_train_sm, y_test,
        title_suffix="Hien tai")

    model_dict = {
        'model'   : xgb,
        'scaler'  : None,
        'type'    : 'xgb',
        'features': feat
    }
    with open(MODEL_SAVE, 'wb') as f_out:
        pickle.dump(model_dict, f_out)
    print(f"\n  [MODEL] Da luu: {MODEL_SAVE}")

    # ── PHẦN B: Dự báo sớm 5 phút ─────────────
    print(f"\n\n" + "▓"*55)
    print(f"  PHAN B: DU BAO SOM {FORECAST_STEPS} BUOC ({FORECAST_STEPS*9.5/60:.1f} PHUT)")
    print("▓"*55)

    X_f, y_f, _ = preprocess(df, forecast_steps=FORECAST_STEPS)
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_f, y_f, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_f)
    X_tr_f_sm, y_tr_f_sm = apply_smote(X_tr_f, y_tr_f)

    xgb_f, acc_f, f1_f = train_xgboost(
        X_tr_f_sm, X_te_f, y_tr_f_sm, y_te_f,
        title_suffix=f"Du bao {FORECAST_STEPS}buoc(~5p)")

    model_dict_5p = {
        'model'   : xgb_f,
        'scaler'  : None,
        'type'    : 'xgb',
        'features': feat
    }
    with open(MODEL_SAVE_5P, 'wb') as f_out:
        pickle.dump(model_dict_5p, f_out)
    print(f"  [MODEL] Da luu: {MODEL_SAVE_5P}")

    # ── TỔNG KẾT ──────────────────────────────
    print(f"\n{'='*55}")
    print(f"  TONG KET")
    print(f"{'='*55}")
    print(f"  [Hien tai]     Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")
    print(f"  [Du bao {FORECAST_STEPS}b~5p] Accuracy: {acc_f*100:.2f}%  F1: {f1_f*100:.2f}%")
    print(f"  Model luu : {MODEL_SAVE} + {MODEL_SAVE_5P}")
    print(f"\n  HOAN THANH!\n")

# ══════════════════════════════════════════════
# PREDICT REAL-TIME (dùng trong landslide_system.py)
# ══════════════════════════════════════════════
# Buffer lưu giá trị trước để tính xu hướng
_prev_buffer = {
    'tilt': [], 'j2': [], 'j3': [], 'rain': [], 'moisture_avg': []
}

def predict_realtime(model_dict, tilt, pitch, roll,
                     j2, j3, rain):
    global _prev_buffer

    tilt_abs      = abs(tilt)
    roll_abs      = abs(roll)
    moisture_avg  = (j2 + j3) / 2
    moisture_diff = abs(j2 - j3)
    tilt_moisture = tilt_abs * moisture_avg

    # Cập nhật buffer (giữ 3 giá trị gần nhất)
    for key, val in [('tilt', tilt), ('j2', j2), ('j3', j3),
                     ('rain', rain), ('moisture_avg', moisture_avg)]:
        _prev_buffer[key].append(val)
        if len(_prev_buffer[key]) > 3:
            _prev_buffer[key].pop(0)

    # Tính xu hướng
    buf = _prev_buffer
    tilt_diff      = tilt - buf['tilt'][-2] if len(buf['tilt']) >= 2 else 0
    j2_diff        = j2   - buf['j2'][-2]   if len(buf['j2'])   >= 2 else 0
    j3_diff        = j3   - buf['j3'][-2]   if len(buf['j3'])   >= 2 else 0
    rain_count     = sum(buf['rain'])
    tilt_trend     = sum(buf['tilt']) / len(buf['tilt'])
    moisture_trend = sum(buf['moisture_avg']) / len(buf['moisture_avg'])

    X = np.array([[tilt, pitch, roll, j2, j3, rain,
                   tilt_abs, roll_abs, moisture_avg,
                   moisture_diff, tilt_moisture,
                   tilt_diff, j2_diff, j3_diff,
                   rain_count, tilt_trend, moisture_trend]])

    xgb   = model_dict['model']
    prob  = xgb.predict_proba(X)[0]
    label = int(np.argmax(prob))
    conf  = float(prob[label]) * 100
    names = ['AN TOAN', 'CANH BAO', 'NGUY HIEM']
    return label, names[label], round(conf, 1)

if __name__ == "__main__":
    main()