"""Upload local data files to AutoDL server via SFTP with retry."""
import paramiko
import os
import time

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"

LOCAL_DIR = r"C:\Users\hlin2\FUTU-QUANT\data_store\market_data"
REMOTE_DIR = "/root/FUTU-QUANT/data_store/market_data"


def connect():
    for attempt in range(5):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(HOST, port=PORT, username=USER, password=PASS,
                           timeout=60, banner_timeout=60, auth_timeout=60)
            return client
        except Exception as e:
            print(f"  Connection attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Failed to connect after 5 attempts")


client = connect()
sftp = client.open_sftp()

for fname in os.listdir(LOCAL_DIR):
    if fname.endswith(".csv"):
        local_path = os.path.join(LOCAL_DIR, fname)
        remote_path = f"{REMOTE_DIR}/{fname}"
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  Uploading {fname} ({size_mb:.1f} MB)...", end=" ", flush=True)
        for attempt in range(3):
            try:
                sftp.put(local_path, remote_path)
                print("OK")
                break
            except Exception as e:
                print(f"retry {attempt+1}: {e}")
                try:
                    sftp.close()
                    client.close()
                except:
                    pass
                client = connect()
                sftp = client.open_sftp()

sftp.close()
client.close()
print("All uploads done!")
