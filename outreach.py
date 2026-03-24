"""
Donor Prospector — Email Outreach Module
Select prospects and send personalized outreach emails.
Uses SMTP for sending. Stores outreach history.
"""

import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

OUTREACH_DIR = Path(__file__).parent / "data"
OUTREACH_LOG = OUTREACH_DIR / "outreach_log.json"
SMTP_CONFIG_PATH = Path(__file__).parent / "smtp_config.json"

# Default email templates
TEMPLATES = {
    "introduction": {
        "subject": "Partnership Opportunity — {charity_name}",
        "body": """Dear {contact_name},

I hope this message finds you well. My name is {sender_name} and I represent {charity_name}.

I'm reaching out because your organization, {org_name}, is doing remarkable work in {sector}. We believe there's a strong alignment between our missions, and I'd love to explore how we might collaborate.

{charity_name} focuses on {charity_mission}, and we see an opportunity to create meaningful impact together, particularly in the {state} community.

Would you be open to a brief conversation to explore potential synergies? I'm available at your convenience.

Best regards,
{sender_name}
{sender_title}
{charity_name}
{sender_email}
{sender_phone}""",
    },
    "donation_request": {
        "subject": "Supporting {sector} in {state} — {charity_name}",
        "body": """Dear {contact_name},

I'm writing on behalf of {charity_name} to share how your support could make a direct impact in {sector} across {state}.

Your organization, {org_name}, has demonstrated a strong commitment to community well-being, and we believe a partnership could amplify both our efforts.

Here's what your support would enable:
- {impact_point_1}
- {impact_point_2}
- {impact_point_3}

We would welcome the opportunity to discuss how we can work together. Could we schedule a brief call this week?

With gratitude,
{sender_name}
{sender_title}
{charity_name}
{sender_email}""",
    },
    "follow_up": {
        "subject": "Following Up — {charity_name} & {org_name}",
        "body": """Dear {contact_name},

I wanted to follow up on my previous message about a potential collaboration between {charity_name} and {org_name}.

We remain excited about the possibility of working together to advance {sector} initiatives in {state}. If now isn't the right time, I completely understand — I'd be happy to reconnect at a later date.

Please don't hesitate to reach out if you have any questions.

Best,
{sender_name}
{charity_name}
{sender_email}""",
    },
}


def load_smtp_config() -> dict:
    if SMTP_CONFIG_PATH.exists():
        with open(SMTP_CONFIG_PATH) as f:
            return json.load(f)
    return {
        "smtp_server": "",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "sender_email": "",
        "sender_name": "",
        "sender_title": "",
        "sender_phone": "",
        "charity_name": "",
        "charity_mission": "",
    }


def save_smtp_config(config: dict):
    SMTP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SMTP_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def fill_template(template_name: str, variables: dict) -> tuple[str, str]:
    """Fill a template with variables. Returns (subject, body)."""
    template = TEMPLATES.get(template_name, TEMPLATES["introduction"])
    subject = template["subject"]
    body = template["body"]
    for key, val in variables.items():
        subject = subject.replace(f"{{{key}}}", str(val))
        body = body.replace(f"{{{key}}}", str(val))
    return subject, body


def send_email(to_email: str, subject: str, body: str, smtp_config: dict = None) -> dict:
    """Send an email via SMTP. Returns status dict."""
    config = smtp_config or load_smtp_config()

    if not config.get("smtp_server") or not config.get("username"):
        return {"status": "error", "message": "SMTP not configured. Go to Outreach tab → Settings."}

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{config.get('sender_name', '')} <{config['sender_email']}>"
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config["smtp_server"], config.get("smtp_port", 587)) as server:
            server.starttls(context=context)
            server.login(config["username"], config["password"])
            server.sendmail(config["sender_email"], to_email, msg.as_string())

        log_entry = {
            "to": to_email,
            "subject": subject,
            "sent_at": datetime.now().isoformat(),
            "status": "sent",
        }
        _log_outreach(log_entry)
        return {"status": "sent", "message": f"Email sent to {to_email}"}

    except Exception as e:
        log_entry = {
            "to": to_email,
            "subject": subject,
            "sent_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }
        _log_outreach(log_entry)
        return {"status": "error", "message": str(e)}


def preview_email(template_name: str, variables: dict) -> str:
    """Preview a filled template without sending."""
    subject, body = fill_template(template_name, variables)
    return f"Subject: {subject}\n\n{body}"


def _log_outreach(entry: dict):
    OUTREACH_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    if OUTREACH_LOG.exists():
        with open(OUTREACH_LOG) as f:
            log = json.load(f)
    log.append(entry)
    with open(OUTREACH_LOG, "w") as f:
        json.dump(log, f, indent=2)


def get_outreach_history() -> list[dict]:
    if OUTREACH_LOG.exists():
        with open(OUTREACH_LOG) as f:
            return json.load(f)
    return []


def get_template_names() -> list[str]:
    return list(TEMPLATES.keys())
