# =============================================
# landslide_gateway.py
# Raspberry Pi: Serial → SQLite → Firebase
# Chạy: python3 landslide_gateway.py
# Cài:  pip3 install pyserial firebase-admin pandas
# Cần:  serviceAccountKey.json cùng thư mục
# =============================================

import serial
import json
import time
import sqlite3
import threading
import firebase_admin
from firebase_admin import credentials, db

# ── Cấu hình ─────────────────────────────────
SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
DB_PATH       = "landslide.db"
FIREBASE_CRED = "serviceAccountKey.json"
FIREBASE_URL  = "https://landside-cf537-default-rtdb.firebaseio.com"
CSV_PATH      = "train_data.csv"

# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            node_id   TEXT    NOT NULL,
            pitch     REAL    DEFAULT 0,
            tilt      REAL    DEFAULT 0,
            roll      REAL    DEFAULT 0,
            j2        REAL    DEFAULT 0,
            j3        REAL    DEFAULT 0,
            rain      INTEGER DEFAULT 0,
            alert     INTEGER DEFAULT 0,
            label     INTEGER DEFAULT -1
        )
    ''')
    conn.commit()

    # ── Tự động migrate nếu bảng cũ thiếu cột ─
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(sensor_data)")
    }
    migrations = {
        "pitch" : "ALTER TABLE sensor_data ADD COLUMN pitch REAL    DEFAULT 0",
        "tilt"  : "ALTER TABLE sensor_data ADD COLUMN tilt  REAL    DEFAULT 0",
        "roll"  : "ALTER TABLE sensor_data ADD COLUMN roll  REAL    DEFAULT 0",
        "label" : "ALTER TABLE sensor_data ADD COLUMN label INTEGER DEFAULT -1",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
            conn.commit()
            print(f"[DB] Da them cot: {col}")

    print("[DB] SQLite OK!")
    return conn

# ─────────────────────────────────────────────
def save_db(conn, node_id, pitch, tilt, roll,
            j2, j3, rain, alert):
    try:
        # Gán nhãn tự động theo pitch
        p = abs(pitch)
        if   p < 0.5:  label = 0   # An toan
        elif p < 1.5:  label = 1   # Small
        else:          label = 2   # Medium

        conn.execute(
            "INSERT INTO sensor_data "
            "(timestamp,node_id,pitch,tilt,roll,"
            "j2,j3,rain,alert,label) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.strftime('%Y-%m-%d %H:%M:%S'),
             node_id, pitch, tilt, roll,
             j2, j3, rain, alert, label)
        )
        conn.commit()
    except Exception as e:
        print(f"  [DB] Loi luu: {e}")

# ─────────────────────────────────────────────
def export_csv(conn):
    try:
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT timestamp, node_id, "
            "pitch, roll, tilt, j2, j3, rain, alert, label "
            "FROM sensor_data "
            "ORDER BY timestamp ASC",
            conn
        )
        df.to_csv(CSV_PATH, index=False)
        total = len(df)
        c0 = len(df[df['label'] == 0])
        c1 = len(df[df['label'] == 1])
        c2 = len(df[df['label'] == 2])
        print(f"\n[CSV] Xuat thanh cong: {CSV_PATH}")
        print(f"  Tong    : {total} mau")
        print(f"  Label 0 (An toan) : {c0}")
        print(f"  Label 1 (Small)   : {c1}")
        print(f"  Label 2 (Medium)  : {c2}\n")
    except ImportError:
        print("[CSV] Cai pandas: pip3 install pandas")
    except Exception as e:
        print(f"[CSV] Loi: {e}")

# ─────────────────────────────────────────────
def init_firebase():
    try:
        cred = credentials.Certificate(FIREBASE_CRED)
        firebase_admin.initialize_app(cred, {
            'databaseURL': FIREBASE_URL
        })
        print("[Firebase] Ket noi OK!")
        return True
    except Exception as e:
        print(f"[Firebase] Loi: {e}")
        return False

# ─────────────────────────────────────────────
def push_firebase(node_id, pitch, tilt, roll,
                  j2, j3, rain, alert):
    try:
        levels = {0: "AN_TOAN", 1: "CANH_BAO", 2: "NGUY_HIEM"}

        db.reference(f'landslide/nodes/{node_id}').set({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch'    : round(pitch, 1),
            'tilt'     : round(tilt,  1),
            'roll'     : round(roll,  1),
            'j2'       : round(j2,    1),
            'j3'       : round(j3,    1),
            'rain'     : rain,
            'alert'    : alert,
            'status'   : 'online',
            'status_text': levels.get(alert, 'UNKNOWN')
        })

        db.reference('landslide/global').update({
            'lastUpdate'  : time.strftime('%Y-%m-%d %H:%M:%S'),
            'globalAlert' : alert
        })

        print(f"  [Firebase] Push OK -> {node_id} alert={alert}")
    except Exception as e:
        print(f"  [Firebase] Loi push: {e}")

# ─────────────────────────────────────────────
def set_node_offline_firebase(nid):
    try:
        db.reference(f'landslide/nodes/{nid}').update({
            'status'   : 'offline',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch'    : 0,
            'tilt'     : 0,
            'roll'     : 0,
            'j2'       : 0,
            'j3'       : 0,
            'rain'     : 0,
            'alert'    : 0,
            'status_text': 'OFFLINE'
        })
        print(f"  [Firebase] {nid} -> offline")
    except Exception as e:
        print(f"  [Firebase] Loi set offline: {e}")

# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  LANDSLIDE GATEWAY — Raspberry Pi 4")
    print("  Serial -> SQLite -> Firebase")
    print("  Lenh: 'csv'    -> xuat train_data.csv")
    print("  Lenh: 'status' -> xem so ban ghi")
    print("  Lenh: 'exit'   -> thoat")
    print("=" * 50 + "\n")

    conn  = init_db()
    fb_ok = init_firebase()

    print(f"[*] Mo cong {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"[!] Loi Serial: {e}")
        print("    Thu: ls /dev/tty*")
        return

    time.sleep(2)
    print("[OK] San sang nhan du lieu!\n")

    # ── Thread nhận lệnh từ bàn phím ──────────
    def keyboard_thread():
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == "csv":
                    export_csv(conn)
                elif cmd in ("exit", "quit"):
                    print("[*] Dung chuong trinh.")
                    import os; os._exit(0)
                elif cmd == "status":
                    rows = conn.execute(
                        "SELECT COUNT(*) FROM sensor_data"
                    ).fetchone()
                    c0 = conn.execute(
                        "SELECT COUNT(*) FROM sensor_data WHERE label=0"
                    ).fetchone()[0]
                    c1 = conn.execute(
                        "SELECT COUNT(*) FROM sensor_data WHERE label=1"
                    ).fetchone()[0]
                    c2 = conn.execute(
                        "SELECT COUNT(*) FROM sensor_data WHERE label=2"
                    ).fetchone()[0]
                    print(f"\n[DB] Tong: {rows[0]} ban ghi")
                    print(f"  Label 0: {c0}")
                    print(f"  Label 1: {c1}")
                    print(f"  Label 2: {c2}\n")
            except:
                break

    t_kb = threading.Thread(target=keyboard_thread, daemon=True)
    t_kb.start()

    # ── Xuất CSV tự động mỗi 1 giờ ────────────
    last_auto_csv = time.time()

    while True:
        try:
            # Auto export CSV mỗi 1 giờ
            if time.time() - last_auto_csv >= 3600:
                export_csv(conn)
                last_auto_csv = time.time()

            line = ser.readline().decode(
                "utf-8", errors="ignore").strip()

            if not line or not line.startswith("{"):
                continue

            data     = json.loads(line)
            msg_type = data.get("type", "")

            # ── Nhận data từ Gateway ──────────────
            if msg_type == "data":
                node_id = data.get("node",  "")
                pitch   = float(data.get("pitch", 0))
                tilt    = float(data.get("tilt",  0))
                roll    = float(data.get("roll",  0))
                j2      = float(data.get("j2",    0))
                j3      = float(data.get("j3",    0))
                rain    = int(data.get("rain",    0))
                alert   = int(data.get("alert",   0))

                if not node_id:
                    continue

                tilt = abs(tilt)

                levels = {0: "AN TOAN",
                          1: "CANH BAO",
                          2: "NGUY HIEM!"}

                print(f"[{node_id}] {time.strftime('%H:%M:%S')}")
                print(f"  Pitch={pitch:.1f}  Tilt={tilt:.1f}  "
                      f"Roll={roll:.1f}  "
                      f"J2={j2:.0f}%  J3={j3:.0f}%  "
                      f"Mua={'Co' if rain else 'Khong'}")
                print(f"  >>> {levels.get(alert, '?')}")

                # 1. Lưu SQLite
                save_db(conn, node_id, pitch, tilt, roll,
                        j2, j3, rain, alert)
                print(f"  [DB] Luu OK")

                # 2. Push Firebase
                if fb_ok:
                    t = threading.Thread(
                        target=push_firebase,
                        args=(node_id, pitch, tilt, roll,
                              j2, j3, rain, alert),
                        daemon=True
                    )
                    t.start()

                print()

            # ── Heartbeat ─────────────────────────
            elif msg_type == "heartbeat":
                count = data.get("count", 0)
                print(f"[HB] Gateway OK | {count} ban tin\n")
                if fb_ok:
                    try:
                        db.reference('landslide/gateway').set({
                            'status'   : 'online',
                            'lastHB'   : time.strftime('%H:%M:%S'),
                            'msgCount' : count
                        })
                    except:
                        pass

            # ── Node timeout ──────────────────────
            elif msg_type == "timeout":
                nid = data.get("node", "?")
                print(f"[!] {nid} MAT KET NOI!\n")
                if fb_ok:
                    t = threading.Thread(
                        target=set_node_offline_firebase,
                        args=(nid,),
                        daemon=True
                    )
                    t.start()

            # ── Status ────────────────────────────
            elif msg_type == "status":
                n1 = "ON" if data.get("n1") else "OFF"
                n2 = "ON" if data.get("n2") else "OFF"
                n3 = "ON" if data.get("n3") else "OFF"
                ga = data.get("globalAlert", 0)
                print(f"[STATUS] N1={n1} N2={n2} "
                      f"N3={n3} Alert={ga}\n")

        except json.JSONDecodeError:
            pass
        except KeyboardInterrupt:
            print("\n[*] Dung chuong trinh.")
            print("[*] Xuat CSV truoc khi thoat...")
            export_csv(conn)
            ser.close()
            conn.close()
            break
        except Exception as e:
            print(f"[ERR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()