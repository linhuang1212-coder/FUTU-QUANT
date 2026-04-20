"""Helper to execute commands on AutoDL server via SSH."""
import sys
import time
import paramiko

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"


def run(cmd: str, timeout: int = 600) -> str:
    for attempt in range(5):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(HOST, port=PORT, username=USER, password=PASS,
                           timeout=60, banner_timeout=60, auth_timeout=60)
            _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            client.close()
            return out + err
        except Exception as e:
            print(f"  SSH attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError("Failed to connect after 5 attempts")


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo hello && uname -a"
    print(run(cmd))
