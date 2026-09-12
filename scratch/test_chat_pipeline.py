import sys
from pathlib import Path
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from app.database import SessionLocal
from app.services.session_service import SessionService
from app.services.chat_service import ChatService
from ml.inference.predict_bert import BERTIncidentPredictor
from ml.inference.predict_ner import NEREntityPredictor

db = SessionLocal()

print("=== 1. Direct Predictor Tests ===")
bert = BERTIncidentPredictor()
ner = NEREntityPredictor()

t1 = "my husband beats me everyday"
t2 = "he beats me everyday after coming home...he also drinks alcohol"

print("BERT t1:", bert.predict(t1))
print("BERT t2:", bert.predict(t2))
print("NER t1:", ner.predict(t1))
print("NER t2:", ner.predict(t2))

print("\n=== 2. Live Chat Simulation ===")
session_data = SessionService.create_session(db)
session_id = session_data['session_id']
print('Created session:', session_id)

msg1 = 'my husband beats me everyday'
res1 = ChatService.process_message(db, session_id, msg1)
print('Step 1 predicted_labels:', res1['predicted_labels'])
print('Step 1 entities:', res1['entities'])
print('Step 1 status:', res1['conversation_status'])

if res1['current_question']:
    msg2 = 'yes'
    res2 = ChatService.process_message(db, session_id, msg2)
    print('Step 2 (yes) predicted_labels:', res2['predicted_labels'])
    print('Step 2 (yes) entities:', res2['entities'])
    print('Step 2 (yes) status:', res2['conversation_status'])
    
    if res2['current_question']:
        msg3 = 'partner'
        res3 = ChatService.process_message(db, session_id, msg3)
        print('Step 3 (partner) predicted_labels:', res3['predicted_labels'])
        print('Step 3 (partner) entities:', res3['entities'])
        print('Step 3 (partner) status:', res3['conversation_status'])

        if res3['current_question']:
            msg4 = 'he beats me everyday after coming home...he also drinks alcohol'
            res4 = ChatService.process_message(db, session_id, msg4)
            print('Step 4 predicted_labels:', res4['predicted_labels'])
            print('Step 4 entities:', res4['entities'])
            print('Step 4 status:', res4['conversation_status'])
            print('Step 4 report:', res4['report'])
