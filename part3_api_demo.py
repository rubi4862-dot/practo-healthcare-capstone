"""
part3_api_demo.py — Part 3: exercise every endpoint in-process with FastAPI's
TestClient (no network port needed) and print the JSON-Lines log it produced.

Run:  MOCK_LLM=1 python part3_api_demo.py
"""

import json

import config
from fastapi.testclient import TestClient

from api.app import LOG, app

LOG.path.unlink(missing_ok=True)          # fresh log for this transcript
client = TestClient(app)


def post(path, body):
    r = client.post(path, json=body)
    print(f"POST {path} {json.dumps(body)[:110]}\n  -> {r.status_code} {json.dumps(r.json())[:400]}\n")
    return r


print("=== POST /ask (normal) ===")
post("/ask", {"question": "How long do thyroid test reports take?"})

print("=== POST /ask (contains a phone number -> masked in the log) ===")
post("/ask", {"question": "Call me on 9876543210 - can I cancel for free 5 hours before?"})

print("=== POST /ask (Pydantic validation error: empty question) ===")
post("/ask", {"question": ""})

print("=== POST /ask (budget cap: oversized request rejected) ===")
post("/ask", {"question": "Please explain the refund rules. " * 60})

print("=== /add-document: new knowledge becomes answerable ===")
q = {"question": "Is free parking available at partner clinics?"}
post("/ask", q)
post("/add-document", {
    "doc_id": "13_clinic_parking_policy.md",
    "title": "Clinic Parking Policy",
    "content": "Partner clinics offer free parking for patients for up to 2 hours during their "
               "appointment. Patients should show their appointment ID at the parking desk.",
})
post("/ask", q)

print("=== WebSocket /ws/chat (two messages, then the client disconnects) ===")
with client.websocket_connect("/ws/chat") as ws:
    for question in ["What is the status of APT-1037?", "Who won the football match yesterday?"]:
        ws.send_json({"question": question})
        reply = ws.receive_json()
        print(f"WS -> {question}\nWS <- {reply['response']['answer']}\n")
client.close()

print("=== logs/requests.jsonl ===")
for line in LOG.path.read_text().splitlines():
    print(line)
