import subprocess
import time
import urllib.request
import json

proc = subprocess.Popen(['uvicorn', 'main:app', '--port', '8002'])
time.sleep(6)

try:
    req = urllib.request.Request('http://127.0.0.1:8002/health')
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print('Success:', data)
except Exception as e:
    print('Failed:', e)
finally:
    proc.terminate()
