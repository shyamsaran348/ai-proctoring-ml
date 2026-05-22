import requests
import json

# Test the start_exam endpoint
url = "http://localhost:8000/api/problems/4/start_exam/"
headers = {
    'Content-Type': 'application/json',
}

try:
    payload = {"student_id": "sentinel_student"}
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 201:
        data = response.json()
        print(f"Session created successfully!")
        print(f"Session ID: {data.get('session_id')}")
        print(f"Problem: {data.get('problem', {}).get('title')}")
    else:
        print(f"Error: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("Could not connect to server. Make sure the Django server is running on localhost:8000")
except Exception as e:
    print(f"Error: {e}")
