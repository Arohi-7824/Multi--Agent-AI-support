"""
# Multi-Agent AI Customer Support (Backend-Only)

This system classifies user messages, detects sentiment, generates responses using historical data, and flags messages for escalation.

## 🗂 Directory Structure
- `agents/`: All intelligent agents
- `data/`: CSV of historical ticket data
- `models/`: Trained model and vectorizer
- `logs/`: Chat logs
- `main.py`: FastAPI app
- `train.py`: Model training script
- `requirements.txt`: Dependencies

## 🚀 Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m textblob.download_corpora

# 2. Add your ticket data to data/Historical_ticket_data.csv

# 3. Train the model
python train.py

# 4. Run the backend
uvicorn main:app --reload
```

## 🔄 Sample API Usage
**POST** `/chat`
```json
{
  "user_id": "u123",
  "message": "I want a refund for my broken item"
}
```

**Response:**
```json
{
  "intent": "refund_request",
  "sentiment": "negative",
  "response": "We apologize. A refund will be initiated immediately.",
  "escalation_required": true
}
```

"""
