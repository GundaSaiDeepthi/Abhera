# PostgreSQL Database Schema Architecture (Design Document)

## Tables Architecture

```sql
-- 1. Sessions Table
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Laws Table (Ingested from data/laws.csv)
CREATE TABLE laws (
    id SERIAL PRIMARY KEY,
    act_name VARCHAR(100) NOT NULL,
    section_number VARCHAR(50) NOT NULL,
    section_text TEXT NOT NULL,
    applicable_label VARCHAR(20) NOT NULL
);

-- 3. Support Services Table (Ingested from data/support_services.csv)
CREATE TABLE support_services (
    id SERIAL PRIMARY KEY,
    service_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    contact_number VARCHAR(255) NOT NULL,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    applicable_label VARCHAR(100) NOT NULL
);

-- 4. Incidents Table
CREATE TABLE incidents (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) REFERENCES sessions(submission_id) ON DELETE CASCADE,
    raw_narrative TEXT NOT NULL,
    combined_narrative TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Conversation Messages Table
CREATE TABLE conversation_messages (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) REFERENCES sessions(submission_id) ON DELETE CASCADE,
    sender VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    message_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Predictions Table (Multi-label classification results)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) REFERENCES sessions(submission_id) ON DELETE CASCADE,
    label VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL,
    is_positive BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Entities Table (NER extracted entities)
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) REFERENCES sessions(submission_id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL,
    entity_value TEXT NOT NULL,
    start_char INT,
    end_char INT
);

-- 8. Questions Table (Dynamic follow-up questions)
CREATE TABLE questions (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) REFERENCES sessions(submission_id) ON DELETE CASCADE,
    field_name VARCHAR(50) NOT NULL,
    question_text TEXT NOT NULL,
    user_response TEXT,
    is_answered BOOLEAN DEFAULT FALSE
);

-- 9. Reports Table
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(64) UNIQUE REFERENCES sessions(submission_id) ON DELETE CASCADE,
    report_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```
