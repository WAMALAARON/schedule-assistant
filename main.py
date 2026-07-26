import os
import json
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, Response, Cookie, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from google import genai
from google.genai import types
import uvicorn

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

# Set insecure transport for local development (HTTP instead of HTTPS)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = FastAPI(title="Schedule Assistant API", version="0.1.0")

# Define Google Calendar Scope
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CLIENT_SECRETS_FILE = "credentials.json"
REDIRECT_URI = "http://localhost:8000/callback"

# In-memory storage for session credentials during testing
user_credentials = {}

# --- Pydantic Models for AI Parser ---
class NLPRequest(BaseModel):
    prompt: str = Field(..., description="Natural language request, e.g., 'Schedule a study session tomorrow at 3pm'")

class ParsedEvent(BaseModel):
    summary: str
    start_time: str = Field(..., description="ISO 8601 formatted datetime string in UTC (Z)")
    end_time: str = Field(..., description="ISO 8601 formatted datetime string in UTC (Z)")

@app.get("/")
def read_root():
    return {"status": "Schedule Assistant API is running"}

@app.get("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    response = RedirectResponse(authorization_url)
    response.set_cookie(key="oauth_state", value=state, httponly=True)
    if hasattr(flow, 'code_verifier') and flow.code_verifier:
        response.set_cookie(key="code_verifier", value=flow.code_verifier, httponly=True)
    return response

@app.get("/callback")
def callback(
    request: Request,
    state: str = None,
    code: str = None,
    oauth_state: str = Cookie(None),
    code_verifier: str = Cookie(None)
):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    if code_verifier:
        flow.code_verifier = code_verifier

    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Store credentials in memory
    user_credentials["token"] = credentials.token
    user_credentials["refresh_token"] = credentials.refresh_token
    user_credentials["token_uri"] = credentials.token_uri
    user_credentials["client_id"] = credentials.client_id
    user_credentials["client_secret"] = credentials.client_secret
    user_credentials["scopes"] = credentials.scopes

    response = Response(content=json.dumps({
        "message": "Authentication successful!",
        "token": credentials.token,
        "refresh_token": credentials.refresh_token
    }), media_type="application/json")
    
    response.set_cookie(key="auth_token", value=credentials.token, httponly=True)
    return response

def get_calendar_service():
    if not user_credentials.get("token"):
        raise HTTPException(
            status_code=401, 
            detail="User not authenticated. Please visit http://localhost:8000/login first."
        )
    
    creds = Credentials(
        token=user_credentials.get("token"),
        refresh_token=user_credentials.get("refresh_token"),
        token_uri=user_credentials.get("token_uri"),
        client_id=user_credentials.get("client_id"),
        client_secret=user_credentials.get("client_secret"),
        scopes=user_credentials.get("scopes")
    )
    return build("calendar", "v3", credentials=creds)

@app.get("/events")
def get_events():
    try:
        service = get_calendar_service()
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])
        return {"events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calendar events: {str(e)}")

@app.post("/create-event")
def create_event(summary: str = "Assistant Test Event", start_time: str = None, end_time: str = None):
    try:
        service = get_calendar_service()
        
        # Default start/end time if not specified (1 hour from now)
        if not start_time:
            start_dt = datetime.utcnow() + timedelta(hours=1)
            start_time = start_dt.isoformat() + "Z"
        if not end_time:
            end_dt = datetime.utcnow() + timedelta(hours=2)
            end_time = end_dt.isoformat() + "Z"

        event_body = {
            "summary": summary,
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }

        created_event = service.events().insert(
            calendarId="primary",
            body=event_body
        ).execute()

        return {
            "status": "Event created successfully",
            "event_link": created_event.get("htmlLink"),
            "event_id": created_event.get("id")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create calendar event: {str(e)}")

@app.post("/schedule")
def schedule_event_nlp(request: NLPRequest):
    # Ensure Google Calendar is authenticated before consuming AI quota
    service = get_calendar_service()
    
    # Load Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set. Please create a .env file.")
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Provide timezone and time context to the AI
        now_utc = datetime.now(timezone.utc)
        current_time_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        system_instruction = f"""
        You are an expert scheduling assistant.
        The current date and time is {current_time_str}.
        The user is located in East Africa Time (EAT, UTC+3).
        Extract the event summary, start time, and end time from the user's natural language request.
        Assume all relative dates (e.g., 'tomorrow', 'next Monday') are relative to the current date provided.
        If no end time is specified, assume the event lasts for exactly 1 hour.
        Output the start_time and end_time strictly in ISO 8601 format with a 'Z' at the end to indicate UTC. 
        You must convert the user's intended local time to UTC for the final output.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ParsedEvent,
                temperature=0.1
            )
        )
        
        # Parse the structured JSON returned by Gemini
        parsed_data = json.loads(response.text)
        
        # Create the event using the extracted details
        event_body = {
            "summary": parsed_data.get("summary"),
            "start": {"dateTime": parsed_data.get("start_time")},
            "end": {"dateTime": parsed_data.get("end_time")},
        }

        created_event = service.events().insert(
            calendarId="primary",
            body=event_body
        ).execute()

        return {
            "status": "Event automatically scheduled via AI",
            "ai_extracted_data": parsed_data,
            "event_link": created_event.get("htmlLink"),
            "event_id": created_event.get("id")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process natural language request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)