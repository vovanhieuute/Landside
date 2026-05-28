# =============================================
# landslide_gateway.py
# Raspberry Pi: Serial → SQLite → Firebase
# Chạy: python3 landslide_gateway.py
# Cài:  pip3 install pyserial firebase-admin
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

# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            node_id   TEXT    NOT NULL,
            tilt      REAL    DEFAULT 0,
            roll      REAL    DEFAULT 0,
            j2        REAL    DEFAULT 0,
            j3        REAL    DEFAULT 0,
            rain      INTEGER DEFAULT 0,
            alert     INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    print("[DB] SQLite OK!")
    return conn

# ─────────────────────────────────────────────
def save_db(conn, node_id, tilt, roll,
            j2, j3, rain, alert):
    try:
        conn.execute(
            "INSERT INTO sensor_data "
            "(timestamp,node_id,tilt,roll,j2,j3,rain,alert) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (time.strftime('%Y-%m-%d %H:%M:%S'),
             node_id, tilt, roll, j2, j3, rain, alert)
        )
        conn.commit()
    except Exception as e:
        print(f"  [DB] Loi luu: {e}")

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
def push_firebase(node_id, tilt, roll,
                  j2, j3, rain, alert):
    try:
        levels = {0:"AN_TOAN", 1:"CANH_BAO", 2:"NGUY_HIEM"}

        # Dữ liệu node
        db.reference(f'landslide/nodes/{node_id}').set({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tilt'     : round(tilt, 1),
            'roll'     : round(roll, 1),
            'j2'       : round(j2,   1),
            'j3'       : round(j3,   1),
            'rain'     : rain,
            'alert'    : alert,
            'status'   : levels.get(alert, 'UNKNOWN')
        })

        # Global alert
        db.reference('landslide/global').update({
            'lastUpdate'  : time.strftime('%Y-%m-%d %H:%M:%S'),
            'globalAlert' : alert
        })

        print(f"  [Firebase] Push OK → {node_id} alert={alert}")
    except Exception as e:
        print(f"  [Firebase] Loi push: {e}")

# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  LANDSLIDE GATEWAY — Raspberry Pi 4")
    print("  Serial → SQLite → Firebase")
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

    while True:
        try:
            line = ser.readline().decode(
                "utf-8", errors="ignore").strip()

            if not line or not line.startswith("{"):
                continue

            data     = json.loads(line)
            msg_type = data.get("type", "")

            # ── Nhận data từ Gateway v2 ───────────
            if msg_type == "data":
                # Format mới: fields riêng lẻ
                node_id = data.get("node",  "")
                tilt    = float(data.get("tilt",  0))
                roll    = float(data.get("roll",  0))
                j2      = float(data.get("j2",    0))
                j3      = float(data.get("j3",    0))
                rain    = int(data.get("rain",    0))
                alert   = int(data.get("alert",   0))

                if not node_id:
                    continue

                levels = {0:"AN TOAN",
                          1:"CANH BAO",
                          2:"NGUY HIEM!"}

                # In terminal
                print(f"[{node_id}] {time.strftime('%H:%M:%S')}")
                print(f"  Tilt={tilt:.1f}  Roll={roll:.1f}  "
                      f"J2={j2:.0f}%  J3={j3:.0f}%  "
                      f"Mua={'Co' if rain else 'Khong'}")
                print(f"  >>> {levels.get(alert, '?')}")

                # 1. Lưu SQLite
                save_db(conn, node_id, tilt, roll,
                        j2, j3, rain, alert)
                print(f"  [DB] Luu OK")

                # 2. Push Firebase (thread riêng)
                if fb_ok:
                    t = threading.Thread(
                        target=push_firebase,
                        args=(node_id, tilt, roll,
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
                    try:
                        db.reference(
                            f'landslide/nodes/{nid}'
                        ).update({'status': 'offline'})
                    except:
                        pass

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
            ser.close()
            conn.close()
            break
        except Exception as e:
            print(f"[ERR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()