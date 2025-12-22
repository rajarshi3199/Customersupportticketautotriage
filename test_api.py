"""
Test script for the Customer Support Ticket Auto-Triage API
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_categories():
    """Test categories endpoint"""
    print("\n" + "=" * 60)
    print("Testing Categories Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/categories")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Single Prediction Endpoint")
    print("=" * 60)
    
    test_cases = [
        {
            "subject": "Application crashes when opening reports",
            "description": "Every time I try to open the monthly report, the application crashes. This started happening after the last update."
        },
        {
            "subject": "Add dark mode option",
            "description": "Please add a dark mode theme option. It would be easier on the eyes during night work."
        },
        {
            "subject": "Cannot connect to server",
            "description": "I'm unable to connect to the server. Getting connection refused errors."
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Subject: {test_case['subject']}")
        print(f"  Description: {test_case['description'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            elapsed_time = (time.time() - start_time) * 1000
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                print(f"  Predicted Category: {prediction['predicted_category']}")
                print(f"  Confidence: {prediction['confidence']:.2%}")
                print(f"  Response Time: {elapsed_time:.2f}ms")
                success_count += 1
            else:
                print(f"  Error: {response.json()}")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    print(f"\nSuccess Rate: {success_count}/{len(test_cases)}")
    return success_count == len(test_cases)

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Batch Prediction Endpoint")
    print("=" * 60)
    
    tickets = [
        {
            "subject": "Application crashes",
            "description": "The app crashes when opening reports"
        },
        {
            "subject": "Add dark mode",
            "description": "Please add dark mode theme option"
        },
        {
            "subject": "Billing question",
            "description": "I have a question about my invoice charges"
        },
        {
            "subject": "Password reset",
            "description": "I forgot my password and need to reset it"
        },
        {
            "subject": "Server connection issue",
            "description": "Cannot connect to the server"
        }
    ]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/predict/batch",
            json={"tickets": tickets},
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total Tickets: {result['total_tickets']}")
            print(f"Response Time: {elapsed_time:.2f}ms")
            print(f"Average Time per Ticket: {elapsed_time/len(tickets):.2f}ms")
            
            print("\nPredictions:")
            for pred in result['predictions']:
                print(f"  Ticket {pred['index']}: {pred['predicted_category']} ({pred['confidence']:.2%})")
            
            return True
        else:
            print(f"Error: {response.json()}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test missing fields
    print("\n1. Testing missing fields:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json={"subject": "Test"},
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test empty request
    print("\n2. Testing empty request:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test invalid endpoint
    print("\n3. Testing invalid endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/api/invalid")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {str(e)}")

def main():
    """Run all API tests"""
    print("=" * 60)
    print("Customer Support Ticket Auto-Triage API - Test Suite")
    print("=" * 60)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API server is running: python api.py\n")
    
    results = {
        "Health Check": test_health_check(),
        "Categories": test_categories(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction()
    }
    
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check the output above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

