import json
import http.client

EMOTE_INDEX = 8
HOST = "127.0.0.1"
PORT = 80

conn = http.client.HTTPConnection(HOST, PORT, timeout=1)
payload = json.dumps({"emote_index": EMOTE_INDEX})
headers = {"Content-Type": "application/json"}

conn.request("POST", "/emote", payload, headers)
conn.getresponse()
conn.close()
