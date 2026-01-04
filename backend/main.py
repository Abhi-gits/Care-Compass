from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical Assistant API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
    
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model (use a supported model from list_models)
model = genai.GenerativeModel('models/gemini-2.5-flash', generation_config={"temperature": 0.4})

# In-memory conversation storage
conversations: Dict[str, List[Dict]] = {}

class MessageHistory(BaseModel):
    content: str
    isUser: bool

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[MessageHistory]] = []
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str

def _wrap_plain_text_as_rtf_with_bold_subheadings(text: str) -> str:
    """Wrap plain text into simple RTF, bolding likely subheadings (lines ending with ':')."""
    escaped_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip('\r')
        escaped_line = (
            line.replace('\\', r'\\')
                .replace('{', r'\{')
                .replace('}', r'\}')
        )
        if escaped_line.strip().endswith(':') and len(escaped_line.strip()) <= 80:
            escaped_lines.append(f"\\b {escaped_line}\\b0")
        else:
            escaped_lines.append(escaped_line)
    body = "\\par\n".join(escaped_lines)
    rtf = (
        "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Arial;}}\\fs22\n"
        "\\b Medical Assistant Response\\b0\\par\n"
        f"{body}\n"
        "}"
    )
    return rtf

def create_medical_prompt(user_message: str, conversation_history: List[MessageHistory]) -> str:
    """Create a comprehensive prompt for the medical assistant"""
    
    system_prompt = """You are a professional medical assistant AI. Provide clear, compassionate, and accurate general medical information and first-aid guidance while prioritizing user safety.

CORE RESPONSIBILITIES:
1. Provide step-by-step first aid instructions and precautions when asked
2. Suggest immediate relief actions for common ailments
3. Explain medical conditions, symptoms, and common causes
4. Ask relevant follow-up questions for clarity
5. Guide users appropriately when professional medical care is needed

Understand the situation by asking clarifying questions if necessary. Always prioritize user safety and well-being.

SAFETY GUARDRAILS (CRITICAL):
- NEVER provide specific diagnoses, prescriptions, or personalized medical advice
- ALWAYS recommend calling emergency services (112/911) for medical emergencies
- Advise seeking professional care immediately for serious conditions
- State limitations clearly: you provide general information only
- Include medical disclaimer in every response

EMERGENCY INDICATORS - Immediately recommend emergency services for:
- Chest pain, difficulty breathing, severe bleeding
- Loss of consciousness, severe head injury
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe allergic reaction, severe burns
- Suicidal thoughts or severe mental health crisis

RESPONSE FORMAT:
Use clear section headers with colons (e.g., "Key Details:", "What You Can Do Now:", "When to Get Help:") followed by bullet points for actions and guidance. Keep response under 150 words.

COMMUNICATION STYLE:
- Be warm, empathetic, and professional
- Use clear, simple language
- Focus on actionable, concrete steps
- Ask one thoughtful follow-up question if needed
"""
    
    # Build conversation context
    context = ""
    if conversation_history:
        context = "\n\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "User" if msg.isUser else "Assistant"
            context += f"{role}: {msg.content}\n"
    
    full_prompt = f"{system_prompt}{context}\n\nUser's current message: {user_message}\n\nPlease respond as a caring medical assistant, following all guidelines. Use clear section headers with colons and bullet points."
    
    return full_prompt

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Medical Assistant API is running", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for medical assistance"""
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error. Please contact administrator."
        )
    
    try:
        # Create the prompt with conversation context
        prompt = create_medical_prompt(request.message, request.conversation_history)
        
        logger.info(f"Processing chat request for session: {request.session_id}")
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="Unable to generate response. Please try again."
            )
        
        generated_text = response.text.strip()
        # Fallback: if model did not return RTF, wrap it into valid RTF with bold subheadings
        if not generated_text.startswith("{\\rtf"):
            logger.info("Model output was not RTF. Wrapping as RTF with bold subheadings.")
            generated_text = _wrap_plain_text_as_rtf_with_bold_subheadings(generated_text)
        
        # Store conversation in memory (optional, for session management)
        if request.session_id not in conversations:
            conversations[request.session_id] = []
        
        conversations[request.session_id].extend([
            {"content": request.message, "isUser": True},
            {"content": generated_text, "isUser": False}
        ])
        # Keep only last 20 messages per session to manage memory
        if len(conversations[request.session_id]) > 20:
            conversations[request.session_id] = conversations[request.session_id][-20:]
        
        return ChatResponse(response=generated_text)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        
        # Temporarily return the actual error for debugging
        error_message = f"DEBUG: {str(e)} (Type: {type(e).__name__})"
        
        return ChatResponse(response=error_message)

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_configured": bool(GEMINI_API_KEY),
        "active_sessions": len(conversations)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
