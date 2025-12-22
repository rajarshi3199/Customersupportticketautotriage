"""
RESTful API for Customer Support Ticket Auto-Triage System

An advanced machine learning project focused on revolutionizing customer support 
through intelligent ticket classification and automated routing systems.

Provides real-time ticket classification endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from predict import predict_ticket
from models import TicketClassifier
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global classifier instance
classifier = None

def load_model():
    """Load the trained model on startup"""
    global classifier
    try:
        classifier = TicketClassifier(model_type=config.MODEL_TYPE)
        classifier.load()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Customer Support Ticket Auto-Triage API',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict',
            'predict_batch': '/api/predict/batch',
            'health': '/api/health',
            'categories': '/api/categories'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = classifier is not None and classifier.is_trained
    return jsonify({
        'status': 'healthy' if model_status else 'unhealthy',
        'model_loaded': model_status,
        'model_type': config.MODEL_TYPE if model_status else None
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of available ticket categories"""
    return jsonify({
        'status': 'success',
        'categories': config.CATEGORIES,
        'priorities': config.PRIORITIES
    })

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """
    Predict category for a single ticket
    
    Request Body (JSON):
    {
        "subject": "Application crashes when opening reports",
        "description": "Every time I try to open the monthly report, the application crashes."
    }
    
    Response:
    {
        "status": "success",
        "prediction": {
            "predicted_category": "Bug Report",
            "confidence": 0.95,
            "probabilities": {
                "Bug Report": 0.95,
                "Feature Request": 0.02,
                ...
            }
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'subject' not in data or 'description' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: subject and description'
            }), 400
        
        subject = data.get('subject', '')
        description = data.get('description', '')
        
        # Validate input
        if not subject.strip() and not description.strip():
            return jsonify({
                'status': 'error',
                'message': 'At least one of subject or description must be provided'
            }), 400
        
        # Make prediction
        result = predict_ticket(subject, description)
        
        if result is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed. Model may not be loaded.'
            }), 500
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'predicted_category': result['predicted_category'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            },
            'input': {
                'subject': subject,
                'description': description
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict categories for multiple tickets
    
    Request Body (JSON):
    {
        "tickets": [
            {
                "subject": "Application crashes",
                "description": "The app crashes when opening reports"
            },
            {
                "subject": "Add dark mode",
                "description": "Please add dark mode theme"
            }
        ]
    }
    
    Response:
    {
        "status": "success",
        "predictions": [
            {
                "predicted_category": "Bug Report",
                "confidence": 0.95,
                ...
            },
            ...
        ]
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if 'tickets' not in data or not isinstance(data['tickets'], list):
            return jsonify({
                'status': 'error',
                'message': 'Missing or invalid "tickets" array in request body'
            }), 400
        
        tickets = data['tickets']
        
        if len(tickets) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Tickets array cannot be empty'
            }), 400
        
        # Process each ticket
        predictions = []
        for i, ticket in enumerate(tickets):
            if 'subject' not in ticket or 'description' not in ticket:
                predictions.append({
                    'index': i,
                    'status': 'error',
                    'message': 'Missing subject or description'
                })
                continue
            
            result = predict_ticket(
                ticket.get('subject', ''),
                ticket.get('description', '')
            )
            
            if result:
                predictions.append({
                    'index': i,
                    'status': 'success',
                    'predicted_category': result['predicted_category'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
            else:
                predictions.append({
                    'index': i,
                    'status': 'error',
                    'message': 'Prediction failed'
                })
        
        return jsonify({
            'status': 'success',
            'total_tickets': len(tickets),
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction error: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': ['/api/predict', '/api/predict/batch', '/api/health', '/api/categories']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

def main():
    """Run the API server"""
    # Load model before starting server
    if not load_model():
        print("ERROR: Failed to load model. Please train the model first.")
        print("Run: python train.py")
        sys.exit(1)
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 60)
    print("Customer Support Ticket Auto-Triage API")
    print("=" * 60)
    print(f"Server running on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    main()

