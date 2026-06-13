# =============================================
# landslide_system.py
# Raspberry Pi: Serial → SQLite → Firebase + AI
# Tác giả: Võ Văn Hiếu - MSSV: 22139021
#
# Chạy: python3 landslide_system.py
# Lệnh: csv | perf | status | exit
# =============================================

import serial, json, time, pickle, threading
import numpy as np, sqlite3, psutil, smtplib
import warnings
warnings.filterwarnings('ignore')

from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# ══════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════
SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
DB_PATH       = "landslide.db"
FIREBASE_CRED = "serviceAccountKey.json"
FIREBASE_URL  = "https://landside-cf537-default-rtdb.firebaseio.com"
CSV_PATH      = "train_data.csv"

EMAIL_SENDER   = "cauuthovohoaichau@gmail.com"
EMAIL_PASSWORD = "sqdr xpsb tvjs flgf"
EMAIL_RECEIVER = [
    "vovanhieuute@gmail.com",
    "22139021@student.hcmute.edu.vn",
]
EMAIL_ENABLED  = True
EMAIL_COOLDOWN = 0   # Gửi email ngay lập tức

AI_MODEL_PATH    = "rf_model.pkl"
AI_MODEL_PATH_5P = "rf_model_5p.pkl"
AI_ENABLED       = True
LABEL_NAMES      = ['AN TOAN', 'CANH BAO', 'NGUY HIEM']

# ══════════════════════════════════════════════
# BIẾN TOÀN CỤC
# ══════════════════════════════════════════════
ai_model    = None
ai_model_5p = None

node_last_seen = {"N01": 0,  "N02": 0,  "N03": 0}
node_online    = {"N01": False, "N02": False, "N03": False}

perf = {
    "N01": {"recv":0,"lost":0,"seq_prev":-1,"latencies":[],"e2e":[]},
    "N02": {"recv":0,"lost":0,"seq_prev":-1,"latencies":[],"e2e":[]},
    "N03": {"recv":0,"lost":0,"seq_prev":-1,"latencies":[],"e2e":[]},
}

# ══════════════════════════════════════════════
# AI — LOAD & PREDICT
# ══════════════════════════════════════════════
def load_ai_model():
    global ai_model, ai_model_5p
    try:
        with open(AI_MODEL_PATH, 'rb') as f:
            ai_model = pickle.load(f)
        print(f"[AI] Load model OK: {AI_MODEL_PATH}")
    except:
        print(f"[AI] Chua co model — chay landslide_ai.py de train")
        ai_model = None
    try:
        with open(AI_MODEL_PATH_5P, 'rb') as f:
            ai_model_5p = pickle.load(f)
        print(f"[AI] Load model 5p OK: {AI_MODEL_PATH_5P}")
    except:
        ai_model_5p = None

def predict_ai_model(model_dict, tilt, pitch, roll, j2, j3, rain):
    if model_dict is None:
        return -1, "CHUA CO MODEL", 0.0
    try:
        tilt_abs      = abs(tilt)
        roll_abs      = abs(roll)
        moisture_avg  = (j2 + j3) / 2
        moisture_diff = abs(j2 - j3)
        tilt_moisture = tilt_abs * moisture_avg
        X = np.array([[tilt, pitch, roll, j2, j3, rain,
                       tilt_abs, roll_abs, moisture_avg,
                       moisture_diff, tilt_moisture]])
        xgb   = model_dict['model']
        prob  = xgb.predict_proba(X)[0]
        label = int(np.argmax(prob))
        conf  = float(prob[label]) * 100
        return label, LABEL_NAMES[label], round(conf, 1)
    except Exception as e:
        print(f"[AI] Loi predict: {e}")
        return -1, "LOI", 0.0

# ══════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════
def send_alert_email(node_id, alert, pitch, tilt, j2, j3, rain):
    if not EMAIL_ENABLED: return
    if alert == 0: return
    colors = {1:"#FF8C00", 2:"#FF0000"}
    levels = {1:"CANH BAO", 2:"NGUY HIEM"}
    emoji  = {1:"⚠️", 2:"🚨"}
    color  = colors.get(alert, "#FF8C00")
    subject = f"{emoji.get(alert,'⚠️')} [{levels.get(alert,'?')}] Sat lo dat - {node_id}"
    body = f"""
<html><body style="font-family:Arial;background:#f5f5f5;padding:20px">
<div style="background:#fff;border-radius:8px;padding:24px;max-width:500px;
     margin:auto;border-left:6px solid {color}">
<h2 style="color:{color};margin-top:0">
  {emoji.get(alert,'⚠️')} CANH BAO SAT LO DAT
</h2>
<table style="width:100%;border-collapse:collapse">
  <tr><td style="padding:8px;color:#666">Node</td>
      <td style="padding:8px;font-weight:bold">{node_id}</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">Muc canh bao</td>
      <td style="padding:8px;font-weight:bold;color:{color}">
        {levels.get(alert,'?')}</td></tr>
  <tr><td style="padding:8px;color:#666">Goc nghieng</td>
      <td style="padding:8px">Pitch={pitch:.1f}°  Tilt={tilt:.1f}°</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">Do am dat</td>
      <td style="padding:8px">J2={j2:.0f}%  J3={j3:.0f}%</td></tr>
  <tr><td style="padding:8px;color:#666">Mua</td>
      <td style="padding:8px">{"CO MUA" if rain else "KHONG MUA"}</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">Thoi gian</td>
      <td style="padding:8px">{time.strftime("%d/%m/%Y %H:%M:%S")}</td></tr>
</table>
{('<p style="color:red;font-weight:bold">⚠ NGUY HIEM — Kiem tra khu vuc ngay!</p>' if alert==2 else '<p style="color:#FF8C00">Theo doi chat tinh trang mai doc.</p>')}
<p style="color:#999;font-size:12px;margin-top:16px">
  He thong canh bao sat lo dat — HCMUTE — Vo Van Hieu 22139021
</p>
</div></body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = ", ".join(EMAIL_RECEIVER)
        msg.attach(MIMEText(body, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"  [EMAIL] Gui OK (muc {alert})")
    except Exception as e:
        print(f"  [EMAIL] Loi: {e}")

def send_alert_email_early(node_id, ai_label, ai_name,
                            ai_conf, pitch, tilt, j2, j3, rain):
    if not EMAIL_ENABLED: return
    colors = {1:"#FF8C00", 2:"#FF0000"}
    color  = colors.get(ai_label, "#FF8C00")
    subject = f"🔮 [CANH BAO SOM] AI du bao {ai_name} trong 5 phut - {node_id}"
    body = f"""
<html><body style="font-family:Arial;background:#f5f5f5;padding:20px">
<div style="background:#fff;border-radius:8px;padding:24px;max-width:500px;
     margin:auto;border-left:6px solid {color}">
<h2 style="color:{color};margin-top:0">
  🔮 CANH BAO SOM — AI DU BAO
</h2>
<p style="color:#666">AI du bao tinh trang
  <b style="color:{color}">{ai_name}</b>
  trong vong <b>5 phut toi</b>
  (do tin cay: {ai_conf:.1f}%)
</p>
<table style="width:100%;border-collapse:collapse">
  <tr><td style="padding:8px;color:#666">Node</td>
      <td style="padding:8px;font-weight:bold">{node_id}</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">AI du bao</td>
      <td style="padding:8px;font-weight:bold;color:{color}">
        {ai_name} ({ai_conf:.1f}%)</td></tr>
  <tr><td style="padding:8px;color:#666">Goc nghieng hien tai</td>
      <td style="padding:8px">Pitch={pitch:.1f}°  Tilt={tilt:.1f}°</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">Do am dat</td>
      <td style="padding:8px">J2={j2:.0f}%  J3={j3:.0f}%</td></tr>
  <tr><td style="padding:8px;color:#666">Mua</td>
      <td style="padding:8px">{"CO MUA" if rain else "KHONG MUA"}</td></tr>
  <tr style="background:#f9f9f9">
      <td style="padding:8px;color:#666">Thoi gian</td>
      <td style="padding:8px">{time.strftime("%d/%m/%Y %H:%M:%S")}</td></tr>
</table>
<p style="color:{color};font-weight:bold;margin-top:16px">
  ⚠ Hay kiem tra khu vuc mai doc ngay!
</p>
<p style="color:#999;font-size:12px">
  He thong canh bao sat lo dat — HCMUTE — Vo Van Hieu 22139021
</p>
</div></body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = ", ".join(EMAIL_RECEIVER)
        msg.attach(MIMEText(body, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"  [EMAIL-SOM] Gui OK -> {ai_name} ({ai_conf:.1f}%)")
    except Exception as e:
        print(f"  [EMAIL-SOM] Loi: {e}")

# ══════════════════════════════════════════════
# SQLITE
# ══════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            node_id    TEXT    NOT NULL,
            seq        INTEGER DEFAULT -1,
            pitch      REAL    DEFAULT 0,
            tilt       REAL    DEFAULT 0,
            roll       REAL    DEFAULT 0,
            j2         REAL    DEFAULT 0,
            j3         REAL    DEFAULT 0,
            rain       INTEGER DEFAULT 0,
            alert      INTEGER DEFAULT 0,
            label      INTEGER DEFAULT -1,
            latency_ms REAL    DEFAULT 0,
            e2e_ms     REAL    DEFAULT 0
        )''')
    conn.commit()
    existing = {r[1] for r in conn.execute("PRAGMA table_info(sensor_data)")}
    for col, sql in {
        "pitch"     : "ALTER TABLE sensor_data ADD COLUMN pitch      REAL    DEFAULT 0",
        "tilt"      : "ALTER TABLE sensor_data ADD COLUMN tilt       REAL    DEFAULT 0",
        "roll"      : "ALTER TABLE sensor_data ADD COLUMN roll       REAL    DEFAULT 0",
        "label"     : "ALTER TABLE sensor_data ADD COLUMN label      INTEGER DEFAULT -1",
        "seq"       : "ALTER TABLE sensor_data ADD COLUMN seq        INTEGER DEFAULT -1",
        "latency_ms": "ALTER TABLE sensor_data ADD COLUMN latency_ms REAL    DEFAULT 0",
        "e2e_ms"    : "ALTER TABLE sensor_data ADD COLUMN e2e_ms     REAL    DEFAULT 0",
    }.items():
        if col not in existing:
            conn.execute(sql); conn.commit()
    print("[DB] SQLite OK!")
    return conn

def save_db(conn, node_id, seq, pitch, tilt, roll,
            j2, j3, rain, alert, latency_ms, e2e_ms):
    try:
        if tilt > 3.0 or j2 > 75 or j3 > 75:   label = 2
        elif tilt > 1.5 or j2 > 65 or j3 > 65: label = 1
        else:                                    label = 0
        conn.execute(
            "INSERT INTO sensor_data "
            "(timestamp,node_id,seq,pitch,tilt,roll,"
            "j2,j3,rain,alert,label,latency_ms,e2e_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime('%Y-%m-%d %H:%M:%S'),
             node_id, seq, pitch, tilt, roll,
             j2, j3, rain, alert, label,
             latency_ms, e2e_ms))
        conn.commit()
    except Exception as e:
        print(f"  [DB] Loi: {e}")

def export_csv(conn):
    try:
        df = pd.read_sql_query(
            "SELECT timestamp,node_id,seq,pitch,roll,tilt,"
            "j2,j3,rain,alert,label,latency_ms,e2e_ms "
            "FROM sensor_data ORDER BY timestamp ASC", conn)
        df.to_csv(CSV_PATH, index=False)
        print(f"[CSV] Xuat {len(df)} mau -> {CSV_PATH}")
        for i, n in enumerate(LABEL_NAMES):
            print(f"  L{i} {n}: {(df['label']==i).sum()}")
        print()
    except Exception as e:
        print(f"[CSV] Loi: {e}")

# ══════════════════════════════════════════════
# HIỆU NĂNG
# ══════════════════════════════════════════════
def print_perf_report():
    print("\n" + "="*55)
    print("  BAO CAO HIEU NANG")
    print("="*55)
    total_recv = 0
    total_lost = 0
    for nid, p in perf.items():
        recv = p["recv"]
        if recv == 0: continue
        lost = p["lost"]
        sent = recv + lost
        total_recv += recv
        total_lost += lost
        print(f"\n  [{nid}]")
        print(f"  Goi nhan    : {recv}")
        print(f"  Goi mat     : {lost}")
        print(f"  Tong gui    : {sent}")
        print(f"  PLR         : {lost/sent*100:.2f}%")
        print(f"  Reliability : {recv/sent*100:.2f}%")
        lats = p["latencies"]
        if lats:
            print(f"  Latency TB  : {sum(lats)/len(lats):.1f}ms")
            print(f"  Latency Min : {min(lats):.1f}ms")
            print(f"  Latency Max : {max(lats):.1f}ms")
        e2es = p["e2e"]
        if e2es:
            print(f"  E2E TB      : {sum(e2es)/len(e2es):.1f}ms")

    # Tổng 3 node
    if total_recv > 0:
        total_sent = total_recv + total_lost
        print(f"\n  [TONG 3 NODE]")
        print(f"  Tong goi nhan: {total_recv}")
        print(f"  Tong goi mat : {total_lost}")
        print(f"  Tong goi gui : {total_sent}")
        print(f"  PLR tong     : {total_lost/total_sent*100:.2f}%")
        print(f"  Reliability  : {total_recv/total_sent*100:.2f}%")

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    print(f"\n  CPU: {cpu:.1f}%  "
          f"RAM: {ram.used//1024//1024}/"
          f"{ram.total//1024//1024}MB ({ram.percent:.1f}%)")
    print("="*55+"\n")

# ══════════════════════════════════════════════
# FIREBASE
# ══════════════════════════════════════════════
def init_firebase():
    try:
        cred = credentials.Certificate(FIREBASE_CRED)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print("[Firebase] Ket noi OK!")
        return True
    except Exception as e:
        print(f"[Firebase] Loi: {e}")
        return False

def push_firebase(node_id, pitch, tilt, roll,
                  j2, j3, rain, alert,
                  ai_label=-1, ai_name="", ai_conf=0.0,
                  ai5_label=-1, ai5_name="", ai5_conf=0.0):
    try:
        node_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch'    : round(pitch, 1),
            'tilt'     : round(tilt,  1),
            'roll'     : round(roll,  1),
            'j2'       : round(j2,    1),
            'j3'       : round(j3,    1),
            'rain'     : rain,
            'alert'    : alert,
            'status'   : 'online',
        }
        if ai_label >= 0:
            node_data['ai'] = {
                'label': ai_label,
                'name' : ai_name,
                'conf' : round(ai_conf, 1)
            }
        if ai5_label >= 0:
            node_data['ai5p'] = {
                'label': ai5_label,
                'name' : ai5_name,
                'conf' : round(ai5_conf, 1)
            }
        db.reference(f'landslide/nodes/{node_id}').set(node_data)
        db.reference('landslide/global').update({
            'lastUpdate' : time.strftime('%Y-%m-%d %H:%M:%S'),
            'globalAlert': alert,
            'aiAlert'    : ai_label if ai_label >= 0 else alert,
            'ai5pAlert'  : ai5_label if ai5_label >= 0 else -1,
        })
        print(f"  [Firebase] {node_id} OK "
              f"alert={alert} AI={ai_name}({ai_conf:.0f}%)")
    except Exception as e:
        print(f"  [Firebase] Loi: {e}")

def set_node_offline_firebase(nid):
    try:
        db.reference(f'landslide/nodes/{nid}').update({
            'status':'offline',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch':0,'tilt':0,'roll':0,'j2':0,'j3':0,'rain':0,'alert':0
        })
    except: pass

# ══════════════════════════════════════════════
# THREADS
# ══════════════════════════════════════════════
def check_node_timeout(fb_ok):
    while True:
        now = time.time()
        for nid in ["N01","N02","N03"]:
            last = node_last_seen[nid]
            if last > 0 and now - last > 300 and node_online[nid]:
                node_online[nid] = False
                print(f"[TIMEOUT] {nid} mat ket noi!\n")
                if fb_ok:
                    threading.Thread(
                        target=set_node_offline_firebase,
                        args=(nid,), daemon=True).start()
        time.sleep(5)

def resource_monitor():
    last = time.time()
    while True:
        if time.time() - last >= 60:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            print(f"[RES] CPU={cpu:.1f}%  "
                  f"RAM={ram.used//1024//1024}MB "
                  f"({ram.percent:.1f}%)")
            last = time.time()
        time.sleep(5)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    load_ai_model()
    print("="*55)
    print("  LANDSLIDE GATEWAY — Raspberry Pi 4")
    print("  csv   -> xuat CSV")
    print("  perf  -> hieu nang")
    print("  status-> so ban ghi DB")
    print("  exit  -> thoat")
    print("="*55+"\n")

    conn  = init_db()
    fb_ok = init_firebase()

    print(f"[*] Mo cong {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"[!] Loi Serial: {e}")
        return

    time.sleep(2)
    print("[OK] San sang!\n")

    threading.Thread(target=check_node_timeout,
                     args=(fb_ok,), daemon=True).start()
    threading.Thread(target=resource_monitor,
                     daemon=True).start()

    def keyboard_thread():
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == "csv":
                    export_csv(conn)
                elif cmd == "perf":
                    print_perf_report()
                elif cmd == "status":
                    rows = conn.execute(
                        "SELECT COUNT(*) FROM sensor_data"
                    ).fetchone()
                    print(f"[DB] Tong: {rows[0]} ban ghi\n")
                elif cmd in ("exit","quit"):
                    print("[*] Thoat...")
                    export_csv(conn)
                    print_perf_report()
                    import os; os._exit(0)
            except:
                break

    threading.Thread(target=keyboard_thread,
                     daemon=True).start()

    last_auto_csv = time.time()

    while True:
        try:
            if time.time() - last_auto_csv >= 3600:
                export_csv(conn)
                last_auto_csv = time.time()

            line = ser.readline().decode(
                "utf-8", errors="ignore").strip()
            if not line or not line.startswith("{"):
                continue

            data     = json.loads(line)
            msg_type = data.get("type","")

            if msg_type == "data":
                t_recv  = time.time()
                node_id = data.get("node","")
                seq     = int(data.get("seq",  -1))
                pitch   = float(data.get("pitch", 0))
                tilt    = float(data.get("tilt",  0))
                roll    = float(data.get("roll",  0))
                j2      = float(data.get("j2",    0))
                j3      = float(data.get("j3",    0))
                rain    = int(data.get("rain",    0))
                alert   = int(data.get("alert",   0))

                if not node_id: continue

                tilt = abs(tilt)
                node_last_seen[node_id] = t_recv
                node_online[node_id]    = True

                latency_ms  = max((time.time()-t_recv)*1000, 0)
                t_e2e_start = time.time()

                p = perf[node_id]
                p["recv"] += 1
                if seq >= 0:
                    prev = p["seq_prev"]
                    if prev >= 0:
                        if seq < prev:
                            print(f"  [PLR] {node_id} seq reset")
                        elif seq > prev + 1:
                            lost = seq - prev - 1
                            p["lost"] += lost
                            print(f"  [PLR] {node_id} mat {lost} goi")
                    p["seq_prev"] = seq
                if latency_ms > 0:
                    p["latencies"].append(latency_ms)

                levels = {0:"AN TOAN",1:"CANH BAO",2:"NGUY HIEM!"}
                print(f"[{node_id}] {time.strftime('%H:%M:%S')} seq={seq}")
                print(f"  Pitch={pitch:.1f} Tilt={tilt:.1f} Roll={roll:.1f} "
                      f"J2={j2:.0f}% J3={j3:.0f}% "
                      f"Mua={'Co' if rain else 'Khong'}")
                print(f"  >>> {levels.get(alert,'?')}")

                save_db(conn, node_id, seq, pitch, tilt,
                        roll, j2, j3, rain, alert, latency_ms, 0)
                print(f"  [DB] Luu OK")

                # AI dự đoán hiện tại
                ai_label, ai_name, ai_conf = -1, "", 0.0
                if AI_ENABLED and ai_model is not None:
                    ai_label, ai_name, ai_conf = predict_ai_model(
                        ai_model, tilt, pitch, roll, j2, j3, rain)
                    if ai_label >= 0:
                        print(f"  [AI] HIEN TAI : {ai_name} "
                              f"(conf={ai_conf:.1f}%)")
                        if ai_label != alert:
                            print(f"  [AI] ! Khac sensor: "
                                  f"AI={ai_label} Sensor={alert}")

                # AI dự báo 5p
                ai5_label, ai5_name, ai5_conf = -1, "", 0.0
                if AI_ENABLED and ai_model_5p is not None:
                    ai5_label, ai5_name, ai5_conf = predict_ai_model(
                        ai_model_5p, tilt, pitch, roll, j2, j3, rain)
                    if ai5_label >= 0:
                        print(f"  [AI] DU BAO 5P: {ai5_name} "
                              f"(conf={ai5_conf:.1f}%)")
                        if ai5_label > alert:
                            print(f"  [AI] *** CANH BAO SOM! ***")
                            threading.Thread(
                                target=send_alert_email_early,
                                args=(node_id, ai5_label, ai5_name,
                                      ai5_conf, pitch, tilt,
                                      j2, j3, rain),
                                daemon=True).start()

                # Email cảnh báo sensor
                if alert > 0:
                    threading.Thread(
                        target=send_alert_email,
                        args=(node_id, alert, pitch,
                              tilt, j2, j3, rain),
                        daemon=True).start()

                # Firebase
                if fb_ok:
                    def push_and_measure(
                            nid, p_, t_, r_, j2_, j3_, ra_, al_,
                            ail_, ain_, aic_,
                            ai5l_, ai5n_, ai5c_, t_start_):
                        push_firebase(nid, p_, t_, r_, j2_, j3_, ra_, al_,
                                      ail_, ain_, aic_, ai5l_, ai5n_, ai5c_)
                        e2e = (time.time()-t_start_)*1000
                        perf[nid]["e2e"].append(e2e)
                        print(f"  [E2E] {nid}: {e2e:.0f}ms")

                    threading.Thread(
                        target=push_and_measure,
                        args=(node_id, pitch, tilt, roll,
                              j2, j3, rain, alert,
                              ai_label, ai_name, ai_conf,
                              ai5_label, ai5_name, ai5_conf,
                              t_e2e_start),
                        daemon=True).start()
                print()

            elif msg_type == "status":
                n1 = "ON" if data.get("n1") else "OFF"
                n2 = "ON" if data.get("n2") else "OFF"
                n3 = "ON" if data.get("n3") else "OFF"
                ga = data.get("globalAlert", 0)
                print(f"[STATUS] N1={n1} N2={n2} N3={n3} Alert={ga}\n")

            elif msg_type == "timeout":
                nid = data.get("node","?")
                print(f"[!] {nid} MAT KET NOI!\n")
                if fb_ok:
                    threading.Thread(
                        target=set_node_offline_firebase,
                        args=(nid,), daemon=True).start()

            elif msg_type == "heartbeat":
                count = data.get("count", 0)
                print(f"[HB] Gateway OK | {count} ban tin\n")

        except json.JSONDecodeError:
            pass
        except KeyboardInterrupt:
            print("\n[*] Thoat...")
            export_csv(conn)
            print_perf_report()
            ser.close(); conn.close()
            break
        except Exception as e:
            print(f"[ERR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()