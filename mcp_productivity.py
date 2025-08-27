from fastapi import FastAPI
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os, pickle, base64, openai, asyncio
import logging
from email.mime.text import MIMEText
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Productivity")

#setting scopes that also needs to be added in Google Developer API OAuth Screen

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events"
]

TOKEN_FILE = "token.pickle"

def get_credentials():
    """Get Google API credentials (synchronous setup function)"""
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                logger.error("credentials.json not found. Please download from Google Cloud Console.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=8080)  # browser login once

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return creds

# Initialize services
creds = get_credentials()
gmail_service = None
calendar_service = None

if creds:
    try:
        gmail_service = build("gmail", "v1", credentials=creds)
        calendar_service = build("calendar", "v3", credentials=creds)
        logger.info("Google services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google services: {e}")
else:
    logger.warning("Google credentials not available - email/calendar features disabled")

# Pydantic Use
class EmailRequest(BaseModel):
    to: str
    subject: str
    message: str

class SummarizeEmailRequest(BaseModel):
    raw_text: str
    to: str
    subject: str

# eMail Functions
@app.post("/email/send")
async def send_email(data: EmailRequest):
    """Send email using Gmail API (async)"""
    if not gmail_service:
        return {"status": "Error", "error": "Gmail service not configured. Check credentials.json"}
    
    try:
        logger.info(f"Attempting to send email to {data.to}")
        
        message = MIMEText(data.message)
        message["to"] = data.to
        message["subject"] = data.subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Make Gmail API call async to avoid blocking
        result = await asyncio.to_thread(
            lambda: gmail_service.users().messages().send(
                userId="me", 
                body={"raw": raw}
            ).execute()
        )
        
        logger.info(f"Email sent successfully to {data.to}")
        return {"status": "Email sent", "to": data.to, "subject": data.subject, "id": result.get("id")}
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {"status": "Error", "error": str(e)}

@app.get("/email/read")
async def read_emails(max_results: int = 5):
    """Fetch last N Unread emails (async)"""
    if not gmail_service:
        return {"error": "Gmail service not configured", "emails": []}
    
    try:
        logger.info(f"Reading {max_results} emails")
        
        # Make Gmail API calls async
        results = await asyncio.to_thread(
            lambda: gmail_service.users().messages().list(
                userId="me",
                maxResults=max_results
            ).execute()
        )
        
        messages = results.get("messages", [])
        emails = []
        
        # Process each message asynchronously
        for msg in messages:
            msg_data = await asyncio.to_thread(
                lambda: gmail_service.users().messages().get(
                    userId="me",
                    id=msg["id"]
                ).execute()
            )
            
            snippet = msg_data.get("snippet", "")
            # Extract subject from headers
            headers = msg_data.get("payload", {}).get("headers", [])
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
            
            emails.append({
                "id": msg["id"], 
                "snippet": snippet,
                "subject": subject,
                "from": sender
            })
        
        logger.info(f"Successfully read {len(emails)} emails")
        return {"emails": emails, "count": len(emails)}
        
    except Exception as e:
        logger.error(f"Failed to read emails: {e}")
        return {"error": str(e), "emails": []}


@app.get("/calendar/events")
async def get_calendar_events(max_results: int = 5):
    """Fetch upcoming calendar events (async)"""
    if not calendar_service:
        return {"error": "Calendar service not configured", "events": []}
    
    try:
        logger.info(f"Fetching {max_results} calendar events")
        
        events_result = await asyncio.to_thread(
            lambda: calendar_service.events().list(
                calendarId="primary",
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
        )
        
        events = events_result.get("items", [])
        formatted = []
        
        for event in events:
            event_data = {
                "id": event.get("id"),
                "summary": event.get("summary", "No Title"),
                "start": event.get("start", {}),
                "end": event.get("end", {}),
                "description": event.get("description", ""),
                "location": event.get("location", "")
            }
            formatted.append(event_data)
        
        logger.info(f"Successfully fetched {len(formatted)} calendar events")
        return {"events": formatted, "count": len(formatted)}
        
    except Exception as e:
        logger.error(f"Failed to fetch calendar events: {e}")
        return {"error": str(e), "events": []}


@app.post("/email/summarize_and_send")
async def summarize_and_email(data: SummarizeEmailRequest):
    """Summarize text and send email (async)"""
    if not gmail_service:
        return {"status": "Error", "error": "Gmail service not configured"}
    
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        
        if openai.api_key == "YOUR_OPENAI_API_KEY":
            return {"status": "Error", "error": "OpenAI API key not configured"}
        
        logger.info(f"Summarizing text and sending email to {data.to}")
        
        prompt = f"Summarize this into a professional email:\n\n{data.raw_text}"
        
        # Make OpenAI API call async
        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
        )
        
        summary = response["choices"][0]["message"]["content"]
        
        # Create and send email
        message = MIMEText(summary)
        message["to"] = data.to
        message["subject"] = data.subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Make Gmail API call async
        result = await asyncio.to_thread(
            lambda: gmail_service.users().messages().send(
                userId="me",
                body={"raw": raw}
            ).execute()
        )
        
        logger.info(f"Summarized email sent successfully to {data.to}")
        return {
            "status": "Email sent", 
            "summary": summary, 
            "to": data.to,
            "subject": data.subject,
            "email_id": result.get("id")
        }
        
    except Exception as e:
        logger.error(f"Failed to summarize and send email: {e}")
        return {"status": "Error", "error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MCP Productivity Server",
        "features": ["Gmail", "Calendar", "AI Summarization"],
        "gmail_configured": gmail_service is not None,
        "calendar_configured": calendar_service is not None,
        "async": True,
        "version": "2.0.0"
    }


@app.get("/test")
async def test_connection():
    """Test if the productivity server is working."""
    return {
        "status": "healthy",
        "service": "MCP Productivity Server",
        "gmail_configured": gmail_service is not None,
        "calendar_configured": calendar_service is not None,
        "openai_configured": os.getenv("OPENAI_API_KEY") != "YOUR_OPENAI_API_KEY"
    }


@app.get("/info")
async def get_info():
    """Get server information"""
    return {
        "title": "MCP Productivity Server",
        "description": "Async productivity server for Gmail and Calendar operations",
        "version": "2.0.0",
        "endpoints": {
            "/email/send": "Send emails",
            "/email/read": "Read recent emails",
            "/calendar/events": "Get calendar events",
            "/email/summarize_and_send": "AI-powered email composition",
            "/health": "Health check",
            "/test": "Connection test",
            "/info": "Server information"
        },
        "async_compatible": True
    }