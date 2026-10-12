import os
import asyncio
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Pydantic-Struktur für das Llama-Output
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt.")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter).")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    recommended_action: str = Field(description="Empfehlung: 'Bewerben', 'Vielleicht' oder 'Ignorieren'.")

# 2. Ihr persönliches Profil
USER_PROFILE = """
- Rolle: Python Backend Entwickler
- Programmiersprachen: Python (Experte), SQL (Sehr gut), FastAPI, Docker
- Berufserfahrung: 3 Jahre
- Arbeitsweise: Hybrid / Remote (Sitz in Berlin)
- Sprache: Deutsch (Muttersprache), Englisch (Fließend)
"""

# 3. Job-Abfrage
async def fetch_jobs():
    print("🔎 Rufe aktuelle Jobangebote ab...")
    try:
        fallback_url = "https://www.arbeitnow.com/api/job-board-api"
        response = requests.get(fallback_url)
        if response.status_code == 200:
            data = response.json()
            jobs = []
            for item in data.get("data", [])[:5]:
                jobs.append({
                    "title": item.get("title"),
                    "company": item.get("company_name"),
                    "description": item.get("description", "")[:1000],
                    "location": item.get("location")
                })
            return jobs
        return []
    except Exception as e:
        print(f"❌ Fehler bei Job-Abfrage: {e}")
        return []

# 4. LangChain-Chain definieren
def create_matching_chain():
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.3-70b-specdec",
        temperature=0.0,
        groq_api_key=api_key
    )
    parser = JsonOutputParser(pydantic_object=JobMatch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein präziser HR-Analyst. Gleiche den Job mit dem Kandidatenprofil ab.\n{format_instructions}"),
        ("user", "Kandidatenprofil:\n{profile}\n\nJobangebot:\nTitel: {title}\nFirma: {company}\nBeschreibung: {description}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

# 5. E-Mail Versendungsfunktion (HTML)
def send_email_report(jobs_list):
    sender_email = os.getenv("EMAIL_SENDER")      # Ihre Gmail-Adresse
    sender_password = os.getenv("EMAIL_PASSWORD")  # Ihr 16-stelliges App-Passwort
    receiver_email = os.getenv("EMAIL_RECEIVER")  # Empfänger (kann dieselbe Adresse sein)
    
    if not sender_email or not sender_password or not receiver_email:
        print("⚠️ E-Mail-Konfiguration unvollständig. Überspringe Versand.")
        return

    # E-Mail Struktur aufbauen
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎯 Dein täglicher Job-Report - {len(jobs_list)} neue Matches"
    msg["From"] = f"Dein AI-Job-Agent <{sender_email}>"
    msg["To"] = receiver_email

    # HTML-Design für eine schöne Darstellung im Postfach
    html_template = """
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }
            .container { max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
            h2 { color: #1e3a8a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
            .job-card { background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin-bottom: 15px; }
            .badge { display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
            .badge-high { background-color: #d1fae5; color: #065f46; }
            .badge-medium { background-color: #fef3c7; color: #92400e; }
            .badge-low { background-color: #fee2e2; color: #991b1b; }
            .score { font-size: 18px; font-weight: bold; float: right; }
            .reason { font-style: italic; color: #555; margin-top: 10px; }
            .footer { font-size: 11px; color: #9ca3af; text-align: center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎯 Dein täglicher Job-Agent Report</h2>
            <p>Hallo! Ich habe heute die neuesten Jobs analysiert. Hier sind deine besten Treffer:</p>
    """

    for job in jobs_list:
        score = job['score']
        # Farbliches Badge basierend auf der Empfehlung bestimmen
        if score >= 70:
            badge_class = "badge-high"
            action = "Bewerben!"
        elif score >= 40:
            badge_class = "badge-medium"
            action = "Vielleicht"
        else:
            badge_class = "badge-low"
            action = "Ignorieren"

        html_template += f"""
            <div class="job-card">
                <span class="score" style="color: {'#059669' if score >= 70 else '#d97706' if score >= 40 else '#dc2626'}">{score}% Match</span>
                <strong style="font-size: 16px;">{job['job_title']}</strong><br>
                <span style="color: #6b7280; font-size: 14px;">{job['company']}</span>
                <div style="margin-top: 10px;">
                    <span class="badge {badge_class}">{action}</span>
                </div>
                <p class="reason"><strong>Kompakt-Match:</strong> {job['reason']}</p>
            </div>
        """

    html_template += """
            <div class="footer">
                Diese E-Mail wurde automatisch von deinem persönlichen AI-Agenten generiert.<br>
                Host: GitHub Actions | Model: Llama 3.1 (70B) via Groq Cloud.
            </div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_template, "html"))

    # Über Gmail SMTP-Server versenden
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("📨 E-Mail-Bericht erfolgreich versendet!")
    except Exception as e:
        print(f"❌ Fehler beim E-Mail-Versand: {e}")

# 6. Haupt-Orchestrator
async def main():
    raw_jobs = await fetch_jobs()
    if not raw_jobs:
        print("Keine Jobs gefunden.")
        return

    match_chain = create_matching_chain()
    analyzed_jobs = []

    print("\n🧠 Starte KI-Abgleich mit Llama 3.1 (70B) via Groq...")
    for job in raw_jobs:
        try:
            result = match_chain.invoke({
                "profile": USER_PROFILE,
                "title": job["title"],
                "company": job["company"],
                "description": job["description"]
            })
            analyzed_jobs.append(result)
        except Exception as e:
            print(f"⚠️ Fehler bei der Analyse von {job['title']}: {e}")

    # Nach Score sortieren (absteigend)
    analyzed_jobs.sort(key=lambda x: x['score'], reverse=True)

    # E-Mail Bericht senden
    send_email_report(analyzed_jobs)

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
