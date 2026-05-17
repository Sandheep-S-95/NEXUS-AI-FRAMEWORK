# ambulance_service.py
from flask import Flask, request, jsonify

app = Flask(__name__)

REGIONAL_REGISTRY = {
    "REG-01": "[DISPATCH OUTPOST NORTH] Ambulance Squad Alpha deployed.",
    "REG-02": "[DISPATCH TRAUMA CENTER] Medic Squad Bravo deployed."
}

@app.route("/alert", methods=["POST"])
def receive_alert():
    data = request.json
    cctv = data.get("cctv_id")
    loc = data.get("location")
    reg = data.get("region_code")
    conf = data.get("confidence", 0.0)
    
    print("\n============================================================")
    print("EMERGENCY INGESTED WITHIN CLUSTER NETWORK BOUNDARY")
    print(f"   |- Source Node: {cctv} ({loc})")
    print(f"   |- ML Model Confidence: {conf:.2f}")
    print(f"   |- Action: {REGIONAL_REGISTRY.get(reg, 'Dispatched General Backup')}")
    print("============================================================")
    return jsonify({
        "status": "acknowledged",
        "action": REGIONAL_REGISTRY.get(reg, 'Dispatched General Backup'),
        "cctv_id": cctv,
        "location": loc
    }), 200

if __name__ == "__main__":
    print("[AMBULANCE] Ambulance Dispatch Service starting on http://127.0.0.1:6000 ...")
    app.run(host="0.0.0.0", port=6000)
