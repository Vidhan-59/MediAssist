# MediAssist Chatbot - Enhanced with ChromaDB RAG

## Requirements
# pip install streamlit supabase openai python-dotenv requests chromadb reportlab

import streamlit as st
import json
import uuid
from datetime import datetime
import os
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass, asdict
import time
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# For demo purposes - you'll need to install these packages:
# pip install supabase openai python-dotenv chromadb reportlab

try:
    from supabase import create_client, Client
    import openai
    from dotenv import load_dotenv
    import chromadb
    from chromadb.config import Settings
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st.warning("⚠️ Some dependencies not installed. This is a demo version.")

# Load environment variables
if DEPENDENCIES_AVAILABLE:
    load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "your-supabase-key")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-openrouter-key")

@dataclass
class PatientData:
    name: str = ""
    age: int = 0
    gender: str = ""
    weight: float = 0.0
    height: float = 0.0
    main_symptom: str = ""
    additional_symptoms: str = ""
    symptom_duration: str = ""
    symptom_severity: str = ""
    pain_location: str = ""
    symptom_triggers: str = ""
    medical_history: str = ""
    current_medications: str = ""
    allergies: str = ""
    family_history: str = ""
    lifestyle_factors: str = ""
    recent_travel: str = ""
    vaccination_status: str = ""
    mental_health: str = ""
    sleep_patterns: str = ""
    dietary_habits: str = ""

class ChromaDBManager:
    """Manage ChromaDB for medical knowledge storage and retrieval"""
    
    def __init__(self):
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Get or create collection for medical knowledge
            self.collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize with medical knowledge if empty
            if self.collection.count() == 0:
                self._initialize_medical_knowledge()
                
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            self.client = None
            self.collection = None

    def _initialize_medical_knowledge(self):
        """Initialize ChromaDB with comprehensive medical knowledge"""
        medical_documents = [
            {
                "id": "fever_treatment",
                "text": "Fever management includes rest, hydration with electrolyte solutions, acetaminophen 500-1000mg every 6 hours, ibuprofen 400-600mg every 8 hours. Home remedies: cool compress, lukewarm baths, light clothing. Seek medical attention if fever >101.5°F for >3 days.",
                "category": "symptom_management",
                "conditions": ["fever", "flu", "infection"]
            },
            {
                "id": "headache_treatment",
                "text": "Headache relief: hydration 8-10 glasses water daily, rest in dark quiet room, acetaminophen 1000mg every 6 hours, ibuprofen 600mg every 8 hours. Home remedies: cold/warm compress, peppermint oil massage, ginger tea. Red flags: sudden severe headache, neck stiffness, vision changes.",
                "category": "symptom_management", 
                "conditions": ["headache", "migraine", "tension headache"]
            },
            {
                "id": "cough_treatment",
                "text": "Cough management: honey 1-2 tsp before bed, warm liquids, humidifier, dextromethorphan 15-30mg every 4 hours. Home remedies: ginger honey tea, steam inhalation, salt water gargle. Seek help for: blood in cough, difficulty breathing, persistent >3 weeks.",
                "category": "symptom_management",
                "conditions": ["cough", "bronchitis", "respiratory infection"]
            },
            {
                "id": "nausea_treatment", 
                "text": "Nausea relief: small frequent meals, ginger 1g daily, ondansetron 4-8mg every 8 hours, metoclopramide 10mg before meals. Home remedies: ginger tea, peppermint, BRAT diet, acupressure P6 point. Red flags: severe dehydration, blood in vomit, abdominal pain.",
                "category": "symptom_management",
                "conditions": ["nausea", "vomiting", "gastroenteritis"]
            },
            {
                "id": "stomach_pain_treatment",
                "text": "Abdominal pain management: identify location and type, antacids for heartburn, simethicone 40-80mg for gas, avoid NSAIDs on empty stomach. Home remedies: heating pad, chamomile tea, probiotics. Emergency signs: severe pain, rigidity, fever with pain.",
                "category": "symptom_management",
                "conditions": ["stomach pain", "abdominal pain", "indigestion"]
            },
            {
                "id": "respiratory_infections",
                "text": "Upper respiratory infection treatment: supportive care, rest 7-9 hours, fluids 8-10 glasses daily, throat lozenges, nasal saline irrigation. Medications: guaifenesin for mucus, pseudoephedrine for congestion. Home remedies: honey lemon tea, steam inhalation, zinc supplements.",
                "category": "condition_treatment",
                "conditions": ["cold", "flu", "respiratory infection", "sinusitis"]
            },
            {
                "id": "allergic_reactions",
                "text": "Allergic reaction management: identify and avoid triggers, antihistamines (cetirizine 10mg, loratadine 10mg daily), topical corticosteroids for skin. Home remedies: cool compresses, oatmeal baths, aloe vera. Emergency: difficulty breathing, swelling face/throat, severe reaction.",
                "category": "condition_treatment", 
                "conditions": ["allergies", "allergic reaction", "hives", "eczema"]
            },
            {
                "id": "digestive_issues",
                "text": "Digestive problem management: identify food triggers, probiotics, fiber 25-35g daily, adequate hydration. Medications: loperamide for diarrhea, polyethylene glycol for constipation. Home remedies: BRAT diet, peppermint tea, fennel seeds, avoid trigger foods.",
                "category": "condition_treatment",
                "conditions": ["diarrhea", "constipation", "IBS", "digestive issues"]
            },
            {
                "id": "sleep_disorders",
                "text": "Sleep improvement: consistent sleep schedule, avoid screens 1 hour before bed, cool dark room, meditation. Natural remedies: chamomile tea, valerian root, melatonin 1-3mg 30min before bed. Avoid: caffeine after 2PM, large meals before bed, alcohol.",
                "category": "condition_treatment",
                "conditions": ["insomnia", "sleep problems", "anxiety", "stress"]
            },
            {
                "id": "pain_management",
                "text": "General pain management: identify pain type and location, acetaminophen 1000mg every 6 hours, ibuprofen 600mg every 8 hours, topical analgesics. Home remedies: ice for acute injury, heat for muscle tension, gentle stretching, relaxation techniques.",
                "category": "symptom_management",
                "conditions": ["pain", "muscle pain", "joint pain", "back pain"]
            }
        ]
        
        # Add documents to ChromaDB
        ids = [doc["id"] for doc in medical_documents]
        texts = [doc["text"] for doc in medical_documents]
        metadatas = [
            {
                "category": doc["category"],
                "conditions": ", ".join(doc["conditions"])  # Convert list to string
            }
            for doc in medical_documents
        ]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        st.success("✅ Medical knowledge base initialized with ChromaDB")

    def query_medical_knowledge(self, symptoms: str, n_results: int = 5) -> str:
        """Query ChromaDB for relevant medical information"""
        if not self.collection:
            return "Medical knowledge base not available."
        
        try:
            # Query ChromaDB for relevant documents
            results = self.collection.query(
                query_texts=[symptoms],
                n_results=n_results
            )
            
            if results['documents'][0]:
                # Combine retrieved documents
                relevant_docs = results['documents'][0]
                return " | ".join(relevant_docs)
            else:
                return "No specific medical information found for these symptoms."
                
        except Exception as e:
            st.error(f"Error querying medical knowledge: {str(e)}")
            return "Error retrieving medical information."

class MediAssistChatbot:
    def __init__(self):
        if DEPENDENCIES_AVAILABLE:
            try:
                self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                self.chroma_manager = ChromaDBManager()
            except Exception as e:
                st.error(f"Failed to connect to databases: {str(e)}")
                self.supabase = None
                self.chroma_manager = None
        else:
            self.supabase = None
            self.chroma_manager = None
        
        # Enhanced medical questions for better context
        self.questions = [
            {"key": "name", "question": "What's your full name?", "type": "text", "required": True},
            {"key": "age", "question": "What's your age?", "type": "number", "required": True},
            {"key": "gender", "question": "What's your gender?", "type": "select", 
             "options": ["Male", "Female", "Other", "Prefer not to say"], "required": True},
            {"key": "weight", "question": "What's your weight (in kg)?", "type": "number", "required": False},
            {"key": "height", "question": "What's your height (in cm)?", "type": "number", "required": False},
            {"key": "main_symptom", "question": "What is your main symptom or primary concern? Please describe in detail.", 
             "type": "text_area", "required": True},
            {"key": "additional_symptoms", "question": "Are you experiencing any additional symptoms? (e.g., fever, fatigue, nausea)", 
             "type": "text_area", "required": False},
            {"key": "symptom_duration", "question": "How long have you been experiencing these symptoms?", "type": "select", 
             "options": ["Less than 6 hours", "6-24 hours", "1-3 days", "4-7 days", "1-2 weeks", "2-4 weeks", "1-3 months", "More than 3 months"], "required": True},
            {"key": "symptom_severity", "question": "How would you rate the severity of your symptoms? (1=mild, 10=severe)", "type": "select",
             "options": ["1 - Very mild", "2 - Mild", "3 - Mild-moderate", "4 - Moderate", "5 - Moderate", "6 - Moderate-severe", "7 - Severe", "8 - Very severe", "9 - Extremely severe", "10 - Unbearable"], "required": True},
            {"key": "pain_location", "question": "If you're experiencing pain, please specify the exact location and type (sharp, dull, throbbing, etc.)", 
             "type": "text_area", "required": False},
            {"key": "symptom_triggers", "question": "Have you noticed any triggers that make your symptoms worse or better? (food, activity, stress, weather, etc.)", 
             "type": "text_area", "required": False},
            {"key": "medical_history", "question": "Please list any chronic medical conditions, past surgeries, or significant medical history", 
             "type": "text_area", "required": False},
            {"key": "current_medications", "question": "List all current medications, supplements, and vitamins you're taking (include dosages if known)", 
             "type": "text_area", "required": False},
            {"key": "allergies", "question": "Do you have any known allergies to medications, foods, or environmental factors?", 
             "type": "text_area", "required": False},
            {"key": "family_history", "question": "Any relevant family medical history? (diabetes, heart disease, cancer, mental health, etc.)", 
             "type": "text_area", "required": False},
            {"key": "lifestyle_factors", "question": "Lifestyle information: smoking status, alcohol consumption, exercise habits, occupation", 
             "type": "text_area", "required": False},
            {"key": "recent_travel", "question": "Recent travel history or exposure to sick individuals in the past 2 weeks?", 
             "type": "text_area", "required": False},
            {"key": "vaccination_status", "question": "Are you up to date with vaccinations? (COVID-19, flu, others relevant to your symptoms)", 
             "type": "select", "options": ["Fully up to date", "Partially up to date", "Not up to date", "Unsure"], "required": False},
            {"key": "mental_health", "question": "How would you describe your current stress levels and mental health?", "type": "select",
             "options": ["Excellent", "Good", "Fair", "Poor", "Very stressed", "Anxious/Depressed"], "required": False},
            {"key": "sleep_patterns", "question": "How would you describe your recent sleep patterns?", "type": "select",
             "options": ["Sleeping well (7-9 hours)", "Some difficulty sleeping", "Frequent sleep disruption", "Severe insomnia"], "required": False},
            {"key": "dietary_habits", "question": "Any recent changes in diet, appetite, or eating patterns?", 
             "type": "text_area", "required": False}
        ]

    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'patient_data' not in st.session_state:
            st.session_state.patient_data = PatientData()
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'diagnosis_complete' not in st.session_state:
            st.session_state.diagnosis_complete = False
        if 'diagnosis_result' not in st.session_state:
            st.session_state.diagnosis_result = None

    def render_question(self, question: Dict[str, Any]) -> Any:
        """Render a question based on its type"""
        key = question["key"]
        q_text = question["question"]
        q_type = question["type"]
        required = question.get("required", False)
        
        # Add required indicator
        if required:
            q_text += " *"
        
        if q_type == "text":
            return st.text_input(q_text, key=f"input_{key}")
        elif q_type == "text_area":
            return st.text_area(q_text, key=f"input_{key}", height=100)
        elif q_type == "number":
            if key in ["weight", "height"]:
                return st.number_input(q_text, min_value=0.0, max_value=300.0, step=0.1, key=f"input_{key}")
            else:
                return st.number_input(q_text, min_value=1, max_value=120, key=f"input_{key}")
        elif q_type == "select":
            return st.selectbox(q_text, [""] + question["options"], key=f"input_{key}")
        
        return None

    def validate_answer(self, question: Dict[str, Any], answer: Any) -> bool:
        """Validate if answer is provided for required questions"""
        if question.get("required", False):
            if answer is None or answer == "" or answer == 0:
                return False
        return True

    def save_answer(self, key: str, value: Any):
        """Save answer to patient data"""
        setattr(st.session_state.patient_data, key, value)

    def get_medical_context_from_chroma(self, patient_data: PatientData) -> str:
        """Get medical context using ChromaDB RAG"""
        if not self.chroma_manager:
            return self._get_fallback_medical_context(patient_data)
        
        # Create comprehensive symptom query
        symptom_query = f"{patient_data.main_symptom} {patient_data.additional_symptoms} {patient_data.pain_location}".strip()
        
        # Get relevant medical information
        medical_context = self.chroma_manager.query_medical_knowledge(symptom_query, n_results=3)
        
        return medical_context

    def _get_fallback_medical_context(self, patient_data: PatientData) -> str:
        """Fallback medical context when ChromaDB is not available"""
        medical_context = {
            "fever": "Fever management: rest, hydration, acetaminophen/ibuprofen, monitor temperature",
            "headache": "Headache relief: rest, hydration, pain relievers, avoid triggers",
            "cough": "Cough treatment: honey, warm liquids, humidifier, avoid irritants",
            "nausea": "Nausea management: small meals, ginger, stay hydrated, avoid strong odors",
            "pain": "Pain relief: appropriate analgesics, rest, ice/heat therapy, gentle movement"
        }
        
        symptoms = f"{patient_data.main_symptom} {patient_data.additional_symptoms}".lower()
        context_parts = []
        
        for keyword, info in medical_context.items():
            if keyword in symptoms:
                context_parts.append(info)
        
        return " | ".join(context_parts) if context_parts else "General symptom evaluation and supportive care recommended."

    def call_openrouter_api(self, patient_data: PatientData, medical_context: str) -> Dict[str, Any]:
        """Call OpenRouter API for comprehensive diagnosis"""
        
        # Demo response with comprehensive information
        demo_response = {
            "collected_data": asdict(patient_data),
            "possible_diagnosis": [
                {"condition": "Viral Upper Respiratory Infection", "probability": "65%", "description": "Common cold-like symptoms"},
                {"condition": "Seasonal Influenza", "probability": "25%", "description": "Flu with systemic symptoms"},
                {"condition": "Bacterial Respiratory Infection", "probability": "10%", "description": "Less common but possible"}
            ],
            "prescribed_medications": [
                {"name": "Acetaminophen", "dosage": "500-1000mg every 6 hours", "purpose": "Fever and pain relief", "precautions": "Max 4g/day, avoid with liver disease"},
                {"name": "Ibuprofen", "dosage": "400-600mg every 8 hours", "purpose": "Anti-inflammatory and pain relief", "precautions": "Take with food, avoid if kidney disease"},
                {"name": "Guaifenesin", "dosage": "200-400mg every 4 hours", "purpose": "Expectorant for cough", "precautions": "Increase fluid intake"}
            ],
            "home_remedies": [
                {"remedy": "Honey and Lemon Tea", "preparation": "1 tbsp honey + lemon juice in warm water", "benefits": "Soothes throat, antimicrobial properties"},
                {"remedy": "Steam Inhalation", "preparation": "Inhale steam from hot water bowl for 10-15 minutes", "benefits": "Opens airways, loosens mucus"},
                {"remedy": "Salt Water Gargle", "preparation": "1/2 tsp salt in warm water", "benefits": "Reduces throat inflammation"},
                {"remedy": "Ginger Tea", "preparation": "Fresh ginger slices in hot water", "benefits": "Anti-inflammatory, nausea relief"},
                {"remedy": "Rest and Hydration", "preparation": "8-10 glasses water daily, 7-9 hours sleep", "benefits": "Supports immune system recovery"}
            ],
            "treatment_recommendations": [
                "Complete rest for 48-72 hours after fever subsides",
                "Maintain hydration with clear fluids and electrolytes",
                "Use humidifier or steam inhalation for respiratory symptoms", 
                "Monitor temperature regularly",
                "Gradual return to normal activities as symptoms improve",
                "Follow up with healthcare provider if symptoms worsen"
            ],
            "red_flags": [
                "High fever (>101.5°F/38.6°C) persisting more than 3 days",
                "Difficulty breathing or shortness of breath",
                "Severe headache with neck stiffness",
                "Persistent vomiting leading to dehydration",
                "Chest pain or rapid heartbeat",
                "Confusion or altered mental state",
                "Signs of severe dehydration (dizziness, dry mouth, decreased urination)"
            ],
            "lifestyle_recommendations": [
                "Avoid smoking and secondhand smoke",
                "Wash hands frequently with soap and water",
                "Get adequate sleep (7-9 hours nightly)",
                "Eat nutritious foods rich in vitamins C and D",
                "Stay physically active when feeling better",
                "Manage stress through relaxation techniques"
            ],
            "follow_up_care": [
                "Schedule follow-up if symptoms persist beyond 10 days",
                "Return for evaluation if any red flag symptoms develop",
                "Consider preventive measures for future respiratory infections",
                "Discuss vaccination status with healthcare provider"
            ],
            "disclaimer": "This is not a medical diagnosis and should not replace professional medical advice. Please consult a qualified healthcare provider for proper diagnosis and treatment."
        }
        
        if not DEPENDENCIES_AVAILABLE or not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-key":
            return demo_response
        
        try:
            system_prompt = """
You are MediAssist, an advanced AI medical assistant. Analyze patient data and medical context to provide comprehensive health assessment.

Provide detailed response in this JSON format:
{
  "collected_data": { ... },
  "possible_diagnosis": [
     {"condition": "condition_name", "probability": "percentage", "description": "brief description"}
  ],
  "prescribed_medications": [
     {"name": "medication", "dosage": "amount and frequency", "purpose": "why prescribed", "precautions": "important warnings"}
  ],
  "home_remedies": [
     {"remedy": "remedy name", "preparation": "how to prepare", "benefits": "why it helps"}
  ],
  "treatment_recommendations": ["detailed recommendations"],
  "red_flags": ["emergency symptoms"],
  "lifestyle_recommendations": ["preventive measures"],
  "follow_up_care": ["when to seek further care"],
  "disclaimer": "medical disclaimer"
}

Guidelines:
- Be medically accurate and evidence-based
- Include appropriate medications with proper dosages
- Suggest safe home remedies
- Always include medical disclaimer
- Focus on common conditions unless clear indicators suggest otherwise
"""
            
            user_message = f"""
Patient Information:
{json.dumps(asdict(patient_data), indent=2)}

Medical Knowledge Context:
{medical_context}

Please provide a comprehensive medical assessment including prescribed medications, home remedies, and detailed care recommendations.
"""
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                try:
                    return json.loads(content)
                except:
                    return demo_response
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return demo_response
                
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
            return demo_response

    def save_to_supabase(self, patient_data: PatientData, diagnosis_result: Dict[str, Any]):
        """Save patient data and diagnosis to Supabase with better error handling"""
        if not self.supabase:
            st.warning("Supabase not connected - data not saved to database")
            return False
        
        try:
            # First, check if tables exist by trying to select from them
            try:
                test_query = self.supabase.table('patients').select("id").limit(1).execute()
            except Exception as table_error:
                st.error(f"Database tables not properly configured: {str(table_error)}")
                st.info("Please ensure you've run the database setup SQL in your Supabase project.")
                return False
            
            # Insert patient data
            patient_insert_data = {
                'name': patient_data.name,
                'age': patient_data.age,
                'gender': patient_data.gender,
                'created_at': datetime.now().isoformat()
            }
            
            patient_result = self.supabase.table('patients').insert(patient_insert_data).execute()
            
            if not patient_result.data:
                st.error("Failed to insert patient data")
                return False
            
            patient_id = patient_result.data[0]['id']
            
            # Insert symptom session
            session_insert_data = {
                'patient_id': patient_id,
                'answers': asdict(patient_data),
                'diagnosis': json.dumps(diagnosis_result.get('possible_diagnosis', [])),
                'treatment': json.dumps(diagnosis_result.get('treatment_recommendations', [])),
                'created_at': datetime.now().isoformat()
            }
            
            session_result = self.supabase.table('symptom_sessions').insert(session_insert_data).execute()
            
            if session_result.data:
                st.success("✅ Data saved to database successfully")
                return True
            else:
                st.error("Failed to insert symptom session data")
                return False
            
        except Exception as e:
            st.error(f"Error saving to database: {str(e)}")
            st.info("This might be due to RLS policies or missing database setup. Check your Supabase configuration.")
            return False

    def generate_pdf_report(self, patient_data: PatientData, diagnosis_result: Dict[str, Any]) -> bytes:
        """Generate comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, textColor=colors.darkblue)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=12, textColor=colors.darkblue)
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("🏥 MediAssist - Health Assessment Report", title_style))
        story.append(Spacer(1, 12))
        
        # Patient Information
        story.append(Paragraph("📋 Patient Information", heading_style))
        patient_info = [
            ['Name:', patient_data.name],
            ['Age:', f"{patient_data.age} years"],
            ['Gender:', patient_data.gender],
            ['Assessment Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        if patient_data.weight > 0:
            patient_info.append(['Weight:', f"{patient_data.weight} kg"])
        if patient_data.height > 0:
            patient_info.append(['Height:', f"{patient_data.height} cm"])
            
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Symptoms
        story.append(Paragraph("🩺 Symptoms & Assessment", heading_style))
        story.append(Paragraph(f"<b>Primary Symptom:</b> {patient_data.main_symptom}", styles['Normal']))
        if patient_data.additional_symptoms:
            story.append(Paragraph(f"<b>Additional Symptoms:</b> {patient_data.additional_symptoms}", styles['Normal']))
        story.append(Paragraph(f"<b>Duration:</b> {patient_data.symptom_duration}", styles['Normal']))
        story.append(Paragraph(f"<b>Severity:</b> {patient_data.symptom_severity}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Possible Diagnoses
        story.append(Paragraph("🔍 Possible Conditions", heading_style))
        for diag in diagnosis_result.get('possible_diagnosis', []):
            story.append(Paragraph(f"• <b>{diag['condition']}</b> - {diag['probability']} likelihood", styles['Normal']))
            if 'description' in diag:
                story.append(Paragraph(f"  {diag['description']}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Prescribed Medications
        if 'prescribed_medications' in diagnosis_result and diagnosis_result['prescribed_medications']:
            story.append(Paragraph("💊 Prescribed Medications", heading_style))
            for med in diagnosis_result['prescribed_medications']:
                story.append(Paragraph(f"<b>{med['name']}</b>", styles['Normal']))
                story.append(Paragraph(f"Dosage: {med['dosage']}", styles['Normal']))
                story.append(Paragraph(f"Purpose: {med['purpose']}", styles['Normal']))
                story.append(Paragraph(f"Precautions: {med['precautions']}", styles['Normal']))
                story.append(Spacer(1, 8))
            story.append(Spacer(1, 12))
        
        # Home Remedies
        if 'home_remedies' in diagnosis_result and diagnosis_result['home_remedies']:
            story.append(Paragraph("🏠 Home Remedies", heading_style))
            for remedy in diagnosis_result['home_remedies']:
                story.append(Paragraph(f"<b>{remedy['remedy']}</b>", styles['Normal']))
                story.append(Paragraph(f"Preparation: {remedy['preparation']}", styles['Normal']))
                story.append(Paragraph(f"Benefits: {remedy['benefits']}", styles['Normal']))
                story.append(Spacer(1, 8))
            story.append(Spacer(1, 12))
        
        # Treatment Recommendations
        story.append(Paragraph("💡 Treatment Recommendations", heading_style))
        for i, rec in enumerate(diagnosis_result.get('treatment_recommendations', []), 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Red Flags
        story.append(Paragraph("🚨 Emergency Warning Signs", heading_style))
        story.append(Paragraph("<b>Seek immediate medical attention if you experience:</b>", styles['Normal']))
        for flag in diagnosis_result.get('red_flags', []):
            story.append(Paragraph(f"• {flag}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Lifestyle Recommendations
        if 'lifestyle_recommendations' in diagnosis_result:
            story.append(Paragraph("🌱 Lifestyle Recommendations", heading_style))
            for rec in diagnosis_result['lifestyle_recommendations']:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Follow-up Care
        if 'follow_up_care' in diagnosis_result:
            story.append(Paragraph("📅 Follow-up Care", heading_style))
            for care in diagnosis_result['follow_up_care']:
                story.append(Paragraph(f"• {care}", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("⚖️ Important Medical Disclaimer", heading_style))
        disclaimer_text = diagnosis_result.get('disclaimer', 
            "This assessment is for informational purposes only and does not constitute medical advice. "
            "Please consult with a qualified healthcare provider for proper diagnosis and treatment.")
        story.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def run(self):
        """Main application logic"""
        st.set_page_config(
            page_title="MediAssist - AI Health Assistant",
            page_icon="🏥",
            layout="wide"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1e40af;
            margin-bottom: 2rem;
        }
        .question-container {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        .diagnosis-container {
            background-color: #f0f9ff;
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #0ea5e9;
        }
        .red-flag {
            background-color: #fef2f2;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
            margin: 1rem 0;
        }
        .medication-card {
            background-color: #f0fdf4;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #22c55e;
            margin: 0.5rem 0;
        }
        .remedy-card {
            background-color: #fefce8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #eab308;
            margin: 0.5rem 0;
        }
        .required {
            color: #ef4444;
        }
        </style>
        """, unsafe_allow_html=True)

        self.init_session_state()
        
        # Header
        st.markdown('<h1 class="main-header">🏥 MediAssist - Enhanced AI Health Assistant</h1>', unsafe_allow_html=True)
        st.markdown("*Comprehensive health guidance with AI-powered diagnosis, medication recommendations, and home remedies*")
        
        # Disclaimer
        st.warning("⚠️ **Important**: This tool provides general health information only and is not a substitute for professional medical advice, diagnosis, or treatment.")
        
        # Show ChromaDB status
        if self.chroma_manager and self.chroma_manager.collection:
            st.success("✅ Enhanced medical knowledge base loaded (ChromaDB)")
        else:
            st.info("ℹ️ Using fallback medical knowledge base")
        
        # Progress bar
        if not st.session_state.diagnosis_complete:
            progress = st.session_state.current_question / len(self.questions)
            st.progress(progress)
            st.write(f"Progress: {st.session_state.current_question}/{len(self.questions)} questions completed")
        
        # Main application logic
        if not st.session_state.diagnosis_complete:
            if st.session_state.current_question < len(self.questions):
                # Show current question
                current_q = self.questions[st.session_state.current_question]
                
                with st.container():
                    st.markdown('<div class="question-container">', unsafe_allow_html=True)
                    st.subheader(f"Question {st.session_state.current_question + 1} of {len(self.questions)}")
                    
                    # Render question
                    answer = self.render_question(current_q)
                    
                    # Show required field indicator
                    if current_q.get("required", False):
                        st.markdown('<p class="required">* Required field</p>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col2:
                        if st.button("Next ➡️", type="primary", use_container_width=True):
                            if self.validate_answer(current_q, answer):
                                self.save_answer(current_q["key"], answer)
                                st.session_state.current_question += 1
                                st.rerun()
                            else:
                                st.error("Please provide an answer for this required field.")
                    
                    # Back button
                    with col1:
                        if st.session_state.current_question > 0:
                            if st.button("⬅️ Back", use_container_width=True):
                                st.session_state.current_question -= 1
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # All questions completed - show summary and get diagnosis
                st.success("✅ All questions completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Your Responses Summary")
                    patient_data = st.session_state.patient_data
                    
                    st.write(f"**Name:** {patient_data.name}")
                    st.write(f"**Age:** {patient_data.age} years")
                    st.write(f"**Gender:** {patient_data.gender}")
                    if patient_data.weight > 0:
                        st.write(f"**Weight:** {patient_data.weight} kg")
                    if patient_data.height > 0:
                        st.write(f"**Height:** {patient_data.height} cm")
                    st.write(f"**Primary Symptom:** {patient_data.main_symptom}")
                    if patient_data.additional_symptoms:
                        st.write(f"**Additional Symptoms:** {patient_data.additional_symptoms}")
                    st.write(f"**Duration:** {patient_data.symptom_duration}")
                    st.write(f"**Severity:** {patient_data.symptom_severity}")
                
                with col2:
                    st.subheader("🔍 Ready for Enhanced Analysis")
                    st.write("Our AI will analyze your symptoms using:")
                    st.write("• Advanced medical knowledge base")
                    st.write("• Evidence-based treatment protocols")
                    st.write("• Personalized medication recommendations")
                    st.write("• Safe home remedy suggestions")
                    
                    if st.button("🩺 Get My Comprehensive Health Assessment", type="primary", use_container_width=True):
                        with st.spinner("Analyzing your symptoms with advanced medical AI..."):
                            # Get medical context from ChromaDB
                            medical_context = self.get_medical_context_from_chroma(patient_data)
                            
                            # Get comprehensive diagnosis from AI
                            diagnosis_result = self.call_openrouter_api(patient_data, medical_context)
                            
                            # Save to session state
                            st.session_state.diagnosis_result = diagnosis_result
                            st.session_state.diagnosis_complete = True
                            
                            # Save to database
                            self.save_to_supabase(patient_data, diagnosis_result)
                            
                            time.sleep(2)  # Allow user to see the analysis process
                            st.rerun()
        
        else:
            # Show comprehensive diagnosis results
            self.display_comprehensive_results()

    def display_comprehensive_results(self):
        """Display comprehensive diagnosis results with medications and remedies"""
        result = st.session_state.diagnosis_result
        patient_data = st.session_state.patient_data
        
        st.markdown('<div class="diagnosis-container">', unsafe_allow_html=True)
        st.header("🩺 Your Comprehensive Health Assessment")
        
        # Patient summary
        st.subheader("👤 Patient Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{patient_data.age} years")
        with col2:
            st.metric("Gender", patient_data.gender)
        with col3:
            if patient_data.weight > 0:
                bmi = patient_data.weight / ((patient_data.height/100) ** 2) if patient_data.height > 0 else 0
                if bmi > 0:
                    st.metric("BMI", f"{bmi:.1f}")
        
        # Possible Diagnoses
        st.subheader("🔍 Possible Conditions")
        for diagnosis in result.get('possible_diagnosis', []):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{diagnosis['condition']}**")
                if 'description' in diagnosis:
                    st.write(f"*{diagnosis['description']}*")
            with col2:
                st.write(f"**{diagnosis['probability']}**")
        
        # Prescribed Medications
        if 'prescribed_medications' in result and result['prescribed_medications']:
            st.subheader("💊 Recommended Medications")
            for med in result['prescribed_medications']:
                st.markdown('<div class="medication-card">', unsafe_allow_html=True)
                st.write(f"**{med['name']}**")
                st.write(f"**Dosage:** {med['dosage']}")
                st.write(f"**Purpose:** {med['purpose']}")
                st.write(f"**⚠️ Precautions:** {med['precautions']}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Home Remedies
        if 'home_remedies' in result and result['home_remedies']:
            st.subheader("🏠 Natural Home Remedies")
            for remedy in result['home_remedies']:
                st.markdown('<div class="remedy-card">', unsafe_allow_html=True)
                st.write(f"**{remedy['remedy']}**")
                st.write(f"**How to prepare:** {remedy['preparation']}")
                st.write(f"**Benefits:** {remedy['benefits']}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Treatment Recommendations
        st.subheader("💡 Treatment Plan")
        for i, recommendation in enumerate(result.get('treatment_recommendations', []), 1):
            st.write(f"{i}. {recommendation}")
        
        # Lifestyle Recommendations
        if 'lifestyle_recommendations' in result:
            st.subheader("🌱 Lifestyle & Prevention")
            for rec in result['lifestyle_recommendations']:
                st.write(f"• {rec}")
        
        # Follow-up Care
        if 'follow_up_care' in result:
            st.subheader("📅 Follow-up Care")
            for care in result['follow_up_care']:
                st.write(f"• {care}")
        
        # Red Flags
        st.subheader("🚨 Emergency Warning Signs")
        st.markdown('<div class="red-flag">', unsafe_allow_html=True)
        st.write("**⚠️ Seek immediate medical attention if you experience:**")
        for flag in result.get('red_flags', []):
            st.write(f"• {flag}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Disclaimer
        st.subheader("⚖️ Important Medical Disclaimer")
        st.info(result.get('disclaimer', 'This assessment is for informational purposes only.'))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 New Assessment", use_container_width=True):
                # Reset session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            # Generate and download PDF
            try:
                pdf_data = self.generate_pdf_report(patient_data, result)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"health_report_{patient_data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
                # Fallback JSON download
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🏥 Find Providers", use_container_width=True):
                st.info("💡 We recommend consulting with a healthcare provider in your area for proper medical care and prescription validation.")
        
        with col4:
            if st.button("📞 Emergency Help", use_container_width=True):
                st.error("🚨 **Emergency Numbers:**\n- Emergency: 911 (US) / 102 (India)\n- Poison Control: 1-800-222-1222 (US)\n- Crisis Helpline: 988 (US)")


# Database setup SQL with enhanced schema
database_setup_sql = """
-- Run this in your Supabase SQL editor to set up the enhanced database

-- Enable required extensions
create extension if not exists "uuid-ossp";
create extension if not exists vector;

-- Drop existing tables if they exist (be careful in production!)
drop table if exists symptom_sessions;
drop table if exists patients;
drop table if exists medical_knowledge;

-- Patients table with enhanced fields
create table patients (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  age int not null,
  gender text not null,
  weight float,
  height float,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enhanced symptom sessions table
create table symptom_sessions (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid references patients(id) on delete cascade,
  answers jsonb not null,
  diagnosis jsonb,
  treatment jsonb,
  medications jsonb,
  home_remedies jsonb,
  session_id text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Medical knowledge base for RAG (enhanced)
create table medical_knowledge (
  id bigserial primary key,
  title text not null,
  content text not null,
  category text,
  conditions text[],
  embedding vector(1536),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for better performance
create index patients_created_at_idx on patients(created_at);
create index symptom_sessions_patient_id_idx on symptom_sessions(patient_id);
create index symptom_sessions_created_at_idx on symptom_sessions(created_at);
create index medical_knowledge_category_idx on medical_knowledge(category);

-- Enable Row Level Security (RLS) - Disable for testing, enable in production
-- alter table patients enable row level security;
-- alter table symptom_sessions enable row level security;
-- alter table medical_knowledge enable row level security;

-- Create policies for authenticated users (uncomment for production)
-- create policy "Enable all operations for authenticated users" on patients
--   for all using (auth.role() = 'authenticated');

-- create policy "Enable all operations for authenticated users" on symptom_sessions
--   for all using (auth.role() = 'authenticated');

-- create policy "Enable read access for authenticated users" on medical_knowledge
--   for select using (auth.role() = 'authenticated');

-- Insert sample medical knowledge
insert into medical_knowledge (title, content, category, conditions) values
('Fever Management', 'Fever treatment includes rest, hydration, acetaminophen 500-1000mg every 6 hours, ibuprofen 400-600mg every 8 hours. Monitor temperature regularly.', 'symptom_management', '{"fever","flu","infection"}'),
('Headache Treatment', 'Headache relief through hydration, rest in dark room, acetaminophen or ibuprofen, identify triggers. Seek help for sudden severe headaches.', 'symptom_management', '{"headache","migraine"}'),
('Respiratory Infections', 'Upper respiratory infections require supportive care, rest, fluids, throat lozenges, steam inhalation, honey for cough.', 'condition_treatment', '{"cold","flu","cough","respiratory"}');
"""

# Main execution
if __name__ == "__main__":
    chatbot = MediAssistChatbot()
    chatbot.run()