# =============================================
# landslide_gateway.py
# Raspberry Pi: Serial → SQLite → Firebase
# Đo: Latency, PLR, Reliability, E2E, CPU/RAM
# =============================================

import serial
import json
import time
import sqlite3
import threading
import psutil
import firebase_admin
from firebase_admin import credentials, db

# ── Cấu hình ─────────────────────────────────
SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
DB_PATH       = "landslide.db"
FIREBASE_CRED = "serviceAccountKey.json"
FIREBASE_URL  = "https://landside-cf537-default-rtdb.firebaseio.com"
CSV_PATH      = "train_data.csv"

# ── Theo dõi node ─────────────────────────────
node_last_seen = {"N01": 0, "N02": 0, "N03": 0}
node_online    = {"N01": False, "N02": False, "N03": False}

# ── Đo hiệu năng ─────────────────────────────
perf = {
    "N01": {"sent": 0, "recv": 0, "lost": 0,
            "seq_first": -1, "seq_last": -1,
            "seq_expected": 0, "latencies": [],
            "e2e": []},
    "N02": {"sent": 0, "recv": 0, "lost": 0,
            "seq_first": -1, "seq_last": -1,
            "seq_expected": 0, "latencies": [],
            "e2e": []},
    "N03": {"sent": 0, "recv": 0, "lost": 0,
            "seq_first": -1, "seq_last": -1,
            "seq_expected": 0, "latencies": [],
            "e2e": []},
}

# ── SQLite ────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            node_id   TEXT    NOT NULL,
            seq       INTEGER DEFAULT -1,
            pitch     REAL    DEFAULT 0,
            tilt      REAL    DEFAULT 0,
            roll      REAL    DEFAULT 0,
            j2        REAL    DEFAULT 0,
            j3        REAL    DEFAULT 0,
            rain      INTEGER DEFAULT 0,
            alert     INTEGER DEFAULT 0,
            label     INTEGER DEFAULT -1,
            latency_ms REAL   DEFAULT 0,
            e2e_ms    REAL    DEFAULT 0
        )
    ''')
    conn.commit()

    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(sensor_data)")
    }
    migrations = {
        "pitch"     : "ALTER TABLE sensor_data ADD COLUMN pitch      REAL    DEFAULT 0",
        "tilt"      : "ALTER TABLE sensor_data ADD COLUMN tilt       REAL    DEFAULT 0",
        "roll"      : "ALTER TABLE sensor_data ADD COLUMN roll       REAL    DEFAULT 0",
        "label"     : "ALTER TABLE sensor_data ADD COLUMN label      INTEGER DEFAULT -1",
        "seq"       : "ALTER TABLE sensor_data ADD COLUMN seq        INTEGER DEFAULT -1",
        "latency_ms": "ALTER TABLE sensor_data ADD COLUMN latency_ms REAL    DEFAULT 0",
        "e2e_ms"    : "ALTER TABLE sensor_data ADD COLUMN e2e_ms     REAL    DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
            conn.commit()
            print(f"[DB] Da them cot: {col}")

    print("[DB] SQLite OK!")
    return conn

# ── Lưu DB ───────────────────────────────────
def save_db(conn, node_id, seq, pitch, tilt, roll,
            j2, j3, rain, alert, latency_ms, e2e_ms):
    try:
        p = abs(pitch)
        if   p < 0.5:  label = 0
        elif p < 1.5:  label = 1
        else:          label = 2

        conn.execute(
            "INSERT INTO sensor_data "
            "(timestamp,node_id,seq,pitch,tilt,roll,"
            "j2,j3,rain,alert,label,latency_ms,e2e_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime('%Y-%m-%d %H:%M:%S'),
             node_id, seq, pitch, tilt, roll,
             j2, j3, rain, alert, label,
             latency_ms, e2e_ms)
        )
        conn.commit()
    except Exception as e:
        print(f"  [DB] Loi luu: {e}")

# ── Xuất CSV ─────────────────────────────────
def export_csv(conn):
    try:
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT timestamp, node_id, seq, "
            "pitch, roll, tilt, j2, j3, rain, "
            "alert, label, latency_ms, e2e_ms "
            "FROM sensor_data ORDER BY timestamp ASC",
            conn
        )
        df.to_csv(CSV_PATH, index=False)
        total = len(df)
        c0 = len(df[df['label'] == 0])
        c1 = len(df[df['label'] == 1])
        c2 = len(df[df['label'] == 2])
        print(f"\n[CSV] Xuat: {CSV_PATH}")
        print(f"  Tong    : {total} mau")
        print(f"  Label 0 : {c0}")
        print(f"  Label 1 : {c1}")
        print(f"  Label 2 : {c2}\n")
    except ImportError:
        print("[CSV] pip3 install pandas")
    except Exception as e:
        print(f"[CSV] Loi: {e}")

# ── In báo cáo hiệu năng ─────────────────────
def print_perf_report():
    print("\n" + "=" * 55)
    print("  BAO CAO HIEU NANG HE THONG")
    print("=" * 55)

    for nid, p in perf.items():
        recv = p["recv"]
        sent = p["sent"]
        lost = p["lost"]
        if recv == 0:
            continue

        plr  = (lost / (sent) * 100) if sent > 0 else 0
        rel  = (recv / sent * 100)   if sent > 0 else 0
        # Giới hạn hợp lệ
        plr  = max(0.0, min(plr,  100.0))
        rel  = max(0.0, min(rel,  100.0))

        lats = p["latencies"]
        avg_lat = sum(lats) / len(lats) if lats else 0
        max_lat = max(lats)             if lats else 0

        e2es = p["e2e"]
        avg_e2e = sum(e2es) / len(e2es) if e2es else 0
        max_e2e = max(e2es)             if e2es else 0

        print(f"\n  [{nid}]")
        print(f"  Goi gui (seq)      : {sent}")
        print(f"  Goi nhan           : {recv}")
        print(f"  Goi mat            : {lost}")
        print(f"  PLR                : {plr:.2f}%")
        print(f"  Reliability        : {rel:.2f}%")
        if lats:
            print(f"  Latency trung binh : {avg_lat:.1f} ms")
            print(f"  Latency max        : {max_lat:.1f} ms")
        if e2es:
            print(f"  E2E trung binh     : {avg_e2e:.1f} ms")
            print(f"  E2E max            : {max_e2e:.1f} ms")

    # CPU/RAM
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    print(f"\n  CPU Usage : {cpu:.1f}%")
    print(f"  RAM Usage : {ram.used//1024//1024} MB / "
          f"{ram.total//1024//1024} MB "
          f"({ram.percent:.1f}%)")
    print("=" * 55 + "\n")

# ── Firebase ─────────────────────────────────
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

def push_firebase(node_id, pitch, tilt, roll,
                  j2, j3, rain, alert):
    try:
        t_before = time.time()
        levels = {0: "AN_TOAN", 1: "CANH_BAO", 2: "NGUY_HIEM"}
        db.reference(f'landslide/nodes/{node_id}').set({
            'timestamp'  : time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch'      : round(pitch, 1),
            'tilt'       : round(tilt,  1),
            'roll'       : round(roll,  1),
            'j2'         : round(j2,    1),
            'j3'         : round(j3,    1),
            'rain'       : rain,
            'alert'      : alert,
            'status'     : 'online',
            'status_text': levels.get(alert, 'UNKNOWN')
        })
        db.reference('landslide/global').update({
            'lastUpdate' : time.strftime('%Y-%m-%d %H:%M:%S'),
            'globalAlert': alert
        })
        rt = (time.time() - t_before) * 1000
        print(f"  [Firebase] Push OK -> {node_id} "
              f"alert={alert} RT={rt:.0f}ms")
    except Exception as e:
        print(f"  [Firebase] Loi push: {e}")

def set_node_offline_firebase(nid):
    try:
        db.reference(f'landslide/nodes/{nid}').update({
            'status'     : 'offline',
            'timestamp'  : time.strftime('%Y-%m-%d %H:%M:%S'),
            'pitch'      : 0, 'tilt': 0, 'roll': 0,
            'j2'         : 0, 'j3' : 0, 'rain': 0,
            'alert'      : 0, 'status_text': 'OFFLINE'
        })
        print(f"  [Firebase] {nid} -> offline")
    except Exception as e:
        print(f"  [Firebase] Loi set offline: {e}")

# ── Thread kiểm tra timeout node ─────────────
def check_node_timeout(fb_ok):
    while True:
        now = time.time()
        for nid in ["N01", "N02", "N03"]:
            last = node_last_seen[nid]
            if last > 0 and now - last > 35:
                if node_online[nid]:
                    node_online[nid] = False
                    print(f"[TIMEOUT] {nid} mat ket noi!\n")
                    if fb_ok:
                        threading.Thread(
                            target=set_node_offline_firebase,
                            args=(nid,), daemon=True
                        ).start()
        time.sleep(5)

# ── Thread log CPU/RAM mỗi 60 giây ───────────
def resource_monitor():
    last = time.time()
    while True:
        if time.time() - last >= 60:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            print(f"[RES] CPU={cpu:.1f}%  "
                  f"RAM={ram.used//1024//1024}MB/"
                  f"{ram.total//1024//1024}MB "
                  f"({ram.percent:.1f}%)")
            last = time.time()
        time.sleep(5)

# ── Main ─────────────────────────────────────
def main():
    print("=" * 55)
    print("  LANDSLIDE GATEWAY — Raspberry Pi 4")
    print("  Lenh: csv    -> xuat CSV")
    print("  Lenh: perf   -> bao cao hieu nang")
    print("  Lenh: status -> so ban ghi DB")
    print("  Lenh: exit   -> thoat")
    print("=" * 55 + "\n")

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

    # Threads
    threading.Thread(
        target=check_node_timeout,
        args=(fb_ok,), daemon=True).start()
    threading.Thread(
        target=resource_monitor,
        daemon=True).start()

    # Keyboard thread
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
                elif cmd in ("exit", "quit"):
                    print("[*] Thoat...")
                    export_csv(conn)
                    print_perf_report()
                    import os; os._exit(0)
            except:
                break

    threading.Thread(
        target=keyboard_thread, daemon=True).start()

    last_auto_csv = time.time()

    while True:
        try:
            # Auto CSV mỗi 1 giờ
            if time.time() - last_auto_csv >= 3600:
                export_csv(conn)
                last_auto_csv = time.time()

            line = ser.readline().decode(
                "utf-8", errors="ignore").strip()
            if not line or not line.startswith("{"):
                continue

            data     = json.loads(line)
            msg_type = data.get("type", "")

            # ── DATA ──────────────────────────────
            if msg_type == "data":
                t_recv   = time.time()
                node_id  = data.get("node",  "")
                seq      = int(data.get("seq",   -1))
                ts_gw    = int(data.get("ts",     0))
                pitch    = float(data.get("pitch", 0))
                tilt     = float(data.get("tilt",  0))
                roll     = float(data.get("roll",  0))
                j2       = float(data.get("j2",    0))
                j3       = float(data.get("j3",    0))
                rain     = int(data.get("rain",    0))
                alert    = int(data.get("alert",   0))

                if not node_id:
                    continue

                tilt = abs(tilt)

                # Cập nhật last seen
                node_last_seen[node_id] = t_recv
                node_online[node_id]    = True

                # Đo latency (gateway → Pi)
                # ts_gw là millis() từ ESP32 (ms kể từ khi boot)
                # KHÔNG thể trừ trực tiếp với Unix timestamp
                # → Dùng thời gian nhận serial thay thế
                # Latency thực = thời gian từ lúc gói đến Serial
                # đến lúc Python xử lý xong (~vài ms)
                latency_ms = (time.time() - t_recv) * 1000
                if latency_ms < 0:
                    latency_ms = 0

                # Đo E2E (tính từ lúc push Firebase xong)
                t_e2e_start = time.time()

                # Đo PLR qua seq number
                p = perf[node_id]
                p["recv"] += 1
                if seq >= 0:
                    if p["seq_first"] == -1:
                        p["seq_first"] = seq
                        p["seq_expected"] = seq + 1
                    else:
                        if seq > p["seq_expected"]:
                            lost = seq - p["seq_expected"]
                            p["lost"] += lost
                            print(f"  [PLR] {node_id} mat "
                                  f"{lost} goi "
                                  f"(seq {p['seq_expected']}"
                                  f"→{seq})")
                        p["seq_expected"] = seq + 1
                    p["seq_last"] = seq
                    # Tổng gói gửi = seq cuối - seq đầu + 1
                    p["sent"] = p["seq_last"] - p["seq_first"] + 1

                if latency_ms > 0:
                    p["latencies"].append(latency_ms)

                levels = {0:"AN TOAN", 1:"CANH BAO",
                          2:"NGUY HIEM!"}
                print(f"[{node_id}] {time.strftime('%H:%M:%S')} "
                      f"seq={seq}")
                print(f"  Pitch={pitch:.1f} Tilt={tilt:.1f} "
                      f"Roll={roll:.1f} "
                      f"J2={j2:.0f}% J3={j3:.0f}% "
                      f"Mua={'Co' if rain else 'Khong'}")
                if latency_ms > 0:
                    print(f"  Latency={latency_ms:.0f}ms")
                print(f"  >>> {levels.get(alert, '?')}")

                # Lưu DB
                save_db(conn, node_id, seq, pitch, tilt,
                        roll, j2, j3, rain, alert,
                        latency_ms, 0)
                print(f"  [DB] Luu OK")

                # Push Firebase + đo E2E
                if fb_ok:
                    def push_and_measure(nid, p_, t_, r_,
                                         j2_, j3_, ra_, al_,
                                         t_start_):
                        push_firebase(nid, p_, t_, r_,
                                      j2_, j3_, ra_, al_)
                        e2e = (time.time() - t_start_) * 1000
                        perf[nid]["e2e"].append(e2e)
                        print(f"  [E2E] {nid}: {e2e:.0f}ms")

                    threading.Thread(
                        target=push_and_measure,
                        args=(node_id, pitch, tilt, roll,
                              j2, j3, rain, alert,
                              t_e2e_start),
                        daemon=True
                    ).start()

                print()

            # ── HEARTBEAT ─────────────────────────
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

            # ── TIMEOUT ───────────────────────────
            elif msg_type == "timeout":
                nid = data.get("node", "?")
                print(f"[!] {nid} MAT KET NOI!\n")
                if fb_ok:
                    threading.Thread(
                        target=set_node_offline_firebase,
                        args=(nid,), daemon=True
                    ).start()

            # ── STATUS ────────────────────────────
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
            print("\n[*] Thoat...")
            export_csv(conn)
            print_perf_report()
            ser.close()
            conn.close()
            break
        except Exception as e:
            print(f"[ERR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()