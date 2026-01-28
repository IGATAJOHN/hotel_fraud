from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_room_assignment():
    # Mock data:
    # 101: Floor 1, Clean, Occ, Dist 2
    # 102: Floor 1, Dirty, Vac, Dist 5
    # 103: Floor 1, Clean, Vac, Dist 10 (Winner if 201 is Floor 2)
    # 201: Floor 2, Clean, Vac, Dist 1
    # 104: Floor 1, Clean, Vac, Dist 2 (Absolute Winner: Lowest Floor + Closest Elevator)
    
    rooms = [
        {"room_number": "101", "floor": 1, "is_clean": True, "is_occupied": True, "distance_to_elevator": 2.0},
        {"room_number": "102", "floor": 1, "is_clean": False, "is_occupied": False, "distance_to_elevator": 5.0},
        {"room_number": "103", "floor": 1, "is_clean": True, "is_occupied": False, "distance_to_elevator": 10.0},
        {"room_number": "201", "floor": 2, "is_clean": True, "is_occupied": False, "distance_to_elevator": 1.0},
        {"room_number": "104", "floor": 1, "is_clean": True, "is_occupied": False, "distance_to_elevator": 2.0},
    ]

    payload = {"rooms": rooms}
    
    print("--- Running Room Assignment Test (via TestClient) ---")
    response = client.post("/assign-room", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Assigned Room: {result['assigned_room']['room_number']}")
        print(f"Floor: {result['assigned_room']['floor']}")
        print(f"Dist to elevator: {result['assigned_room']['distance_to_elevator']}")
        
        # Expected: 104
        if result['assigned_room']['room_number'] == "104":
            print("SUCCESS: Correct room assigned.")
        else:
            print(f"FAILURE: Expected 104, got {result['assigned_room']['room_number']}")
            exit(1)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        exit(1)

if __name__ == "__main__":
    test_room_assignment()

