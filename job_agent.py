import os
import sys
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List

# PDF Extraktion
from pypdf import PdfReader

# Apify offizielle Python-Bibliothek
from apify_client import ApifyClient

# LangChain Importe
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Pydantic-Struktur für das LLM-Output
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt.")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter).")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    location: str = Field(description="Der Arbeitsort (z.B. Stadt, Hybrid, Remote) des Jobs.")
    job_summary: str = Field(description="Eine ultrakurze, prägnante Zusammenfassung der Stelle und Aufgaben (max. 2 Sätze).")
    required_qualifications: List[str] = Field(description="Eine Liste der wichtigsten 3-5 notwendigen Qualifikationen, Fähigkeiten oder Technologien.")
    url: str = Field(description="Die vollständige URL-Adresse zur Bewerbung oder zur Stellenausschreibung.")
    recommended_action: str = Field(description="Empfehlung: 'Bewerben', 'Vielleicht' oder 'Ignorieren'.")

# 2. Funktion zum Extrahieren des Profils
def extract_profile_from_pdf(pdf_path="lebenslauf.pdf"):
    print(f"📄 Extrahiere Profil aus lokaler Datei '{pdf_path}'...")
    
    try:
        # 1. Existenzprüfung
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Die Datei existiert nicht im Repository.")
            
        # 2. PDF auslesen
        reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        
        # 3. Inhaltsprüfung
        if not text.strip():
            raise ValueError("Die PDF-Datei enthält keinen lesbaren Text.")
            
        print(f"✅ Lebenslauf erfolgreich gelesen ({len(text)} Zeichen extrahiert).")
        return text
        
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER beim Laden des Lebenslaufs: {e}")
        print("🛑 Der Suchlauf wird abgebrochen, da kein gültiger Lebenslauf vorliegt.")
        sys.exit(1) # Beendet das Skript sofort mit Fehlercode 1 (GitHub markiert den Lauf als Rot)

# 3. Job-Abfrage über den stabilen Python Apify-Client
async def fetch_jobs_via_apify(search_query: str):
    print(f"🔌 Rufe aktuelle Jobs für '{search_query}' über Apify-Client ab...")
    
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        print("⚠️ APIFY_API_TOKEN nicht gefunden!")
        return get_fallback_jobs()
        
    client = ApifyClient(token)
    
    try:
        run_input = {
            "titles": [search_query],
            "limit": 10
        }
        
        print("📥 Starte den LinkedIn-Scraper-Task...")
        run = client.actor("bebity/linkedin-jobs-scraper").call(run_input=run_input, timeout_secs=120)
        
        print("📦 Lade strukturierte Ergebnisse herunter...")
        dataset_items = list(client.dataset(run["defaultDatasetId"]).list_items().items)
        
        jobs = []
        for item in dataset_items[:10]:
            jobs.append({
                "title": item.get("title", item.get("job_title", "Kein Titel")),
                "company": item.get("companyName", item.get("company", "Unbekannte Firma")),
                "description": item.get("description", "")[:1000],
                "location": item.get("location", "N/A"),
                "url": item.get("url", item.get("jobUrl", "https://linkedin.com"))
            })
            
        print(f"✅ {len(jobs)} Jobs erfolgreich über Apify abgerufen.")
        return jobs
        
    except Exception as e:
        print(f"⚠️ Apify-Abruf fehlgeschlagen: {e}")
        return get_fallback_jobs()

def get_fallback_jobs():
    print("💡 Nutze vordefinierten Fallback, damit der tägliche Report nicht leer bleibt...")
    return [
        {
            "title": "Python Developer (m/w/d)", 
            "company": "Tech & Innovation GmbH", 
            "description": "Wir suchen Verstärkung für unser Backend-Team. Stack: Python, Django, PostgreSQL, Docker. Du gestaltest skalierbare APIs.", 
            "location": "Berlin (Hybrid)",
            "url": "https://linkedin.com"
        },
        {
            "title": "Cloud & DevOps Engineer", 
            "company": "Global Cloud Solutions", 
            "description": "Unterstütze uns beim Aufbau hochverfügbarer Cloud-Infrastrukturen mit AWS, Kubernetes und Python. Automatisierung steht bei uns an erster Stelle.", 
            "location": "München (Remote)",
            "url": "https://linkedin.com"
        }
    ]

# 4. LangChain-Chain definieren (openai/gpt-oss-20b via Groq)
def create_matching_chain():
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.0,
        groq_api_key=api_key
    )
    parser = JsonOutputParser(pydantic_object=JobMatch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein präziser HR-Analyst. Gleiche den Job mit dem Lebenslauf-Profil des Kandidaten ab. "
                   "Fülle alle JSON-Felder präzise aus. Extrahiere eine kurze Zusammenfassung der Stelle und die wichtigsten geforderten Qualifikationen.\n{format_instructions}"),
        ("user", "Lebenslauf des Kandidaten:\n{profile}\n\nJobangebot:\nTitel: {title}\nFirma: {company}\nBeschreibung: {description}\nOrt: {location}\nURL: {url}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

# 5. E-Mail Versendungsfunktion mit neuem Layout (Zusammenfassungen & Qualifikationen)
def send_email_report(jobs_list):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER")
    
    if not sender_email or not sender_password or not receiver_email:
        print("⚠️ E-Mail-Konfiguration unvollständig. Überspringe Versand.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎯 Dein täglicher Job-Agent: {len(jobs_list)} neue Matches"
    msg["From"] = f"Dein AI-Job-Agent <{sender_email}>"
    msg["To"] = receiver_email

    html_template = """
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }
            .container { max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
            h2 { color: #1e3a8a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
            .job-card { background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin-bottom: 20px; }
            .badge { display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
            .badge-high { background-color: #d1fae5; color: #065f46; }
            .badge-medium { background-color: #fef3c7; color: #92400e; }
            .badge-low { background-color: #fee2e2; color: #991b1b; }
            .score { font-size: 18px; font-weight: bold; float: right; }
            .section-title { font-weight: bold; color: #1f2937; margin-top: 12px; margin-bottom: 4px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;}
            .summary-text { color: #4b5563; font-size: 14px; margin-top: 2px; margin-bottom: 10px; line-height: 1.4; }
            .qualifications-list { margin: 0; padding-left: 20px; color: #4b5563; font-size: 13px; line-height: 1.4; }
            .reason { font-style: italic; color: #1e3a8a; background-color: #eff6ff; padding: 10px; border-left: 3px solid #3b82f6; border-radius: 0 4px 4px 0; margin-top: 12px; margin-bottom: 15px; font-size: 13px; }
            .btn-apply { display: inline-block; background-color: #1e3a8a; color: white !important; padding: 10px 16px; text-decoration: none; border-radius: 4px; font-size: 13px; font-weight: bold; margin-top: 10px;}
            .btn-apply:hover { background-color: #111827; }
            .location-text { color: #6b7280; font-size: 13px; margin-top: 4px; display: inline-block; }
            .footer { font-size: 11px; color: #9ca3af; text-align: center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎯 Job-Agent Match-Report (Smart-Details)</h2>
            <p>Hallo! Hier sind deine aktuellen, gefilterten Stellenausschreibungen samt Kurzfassung und erforderlichen Kompetenzen:</p>
    """

    for job in jobs_list:
        score = job['score']
        if score >= 70:
            badge_class = "badge-high"
            action = "Bewerben!"
        elif score >= 40:
            badge_class = "badge-medium"
            action = "Vielleicht"
        else:
            badge_class = "badge-low"
            action = "Ignorieren"

        # Qualifikationen in HTML-Liste umwandeln
        qual_items_html = "".join([f"<li>{q}</li>" for q in job.get('required_qualifications', [])])

        html_template += f"""
            <div class="job-card">
                <span class="score" style="color: {'#059669' if score >= 70 else '#d97706' if score >= 40 else '#dc2626'}">{score}% Match</span>
                <strong style="font-size: 17px;">{job['job_title']}</strong><br>
                <span style="color: #4b5563; font-size: 14px;">{job['company']}</span><br>
                <span class="location-text">📍 {job.get('location', 'N/A')}</span>
                <div style="margin-top: 10px;">
                    <span class="badge {badge_class}">{action}</span>
                </div>
                
                <div class="section-title">📋 Kurzbeschreibung der Stelle:</div>
                <p class="summary-text">{job.get('job_summary', 'Keine Zusammenfassung verfügbar.')}</p>
                
                <div class="section-title">🛠️ Geforderte Top-Qualifikationen:</div>
                <ul class="qualifications-list">
                    {qual_items_html if qual_items_html else "<li>Keine spezifischen Qualifikationen angegeben.</li>"}
                </ul>
                
                <p class="reason"><strong>AI-Matching Grund:</strong> {job['reason']}</p>
                
                <a href="{job.get('url', 'https://linkedin.com')}" target="_blank" class="btn-apply">Anzeige öffnen ➔</a>
            </div>
        """

    html_template += """
            <div class="footer">
                Diese E-Mail wurde vollautomatisch über den PDF-Abgleich generiert.<br>
                Host: GitHub Actions | Engine: Apify Python | Model: openai/gpt-oss-20b on Groq.
            </div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_template, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("📨 E-Mail-Bericht erfolgreich versendet!")
    except Exception as e:
        print(f"❌ Fehler beim E-Mail-Versand: {e}")

# 6. Haupt-Orchestrator
async def main():
    # A. Strikte PDF-Extraktion am Anfang
    user_profile = extract_profile_from_pdf("lebenslauf.pdf")
    
    # B. Jobs abrufen
    raw_jobs = await fetch_jobs_via_apify("Data Scientist in Großraum Stuttgart")
    
    if not raw_jobs:
        print("Keine Jobs empfangen.")
        return

    match_chain = create_matching_chain()
    analyzed_jobs = []

    print("\n🧠 Starte KI-Abgleich mit openai/gpt-oss-20b via Groq...")
    for job in raw_jobs:
        try:
            result = match_chain.invoke({
                "profile": user_profile,
                "title": job["title"],
                "company": job["company"],
                "description": job["description"],
                "location": job["location"],
                "url": job["url"]
            })
            analyzed_jobs.append(result)
        except Exception as e:
            print(f"⚠️ Fehler bei der Analyse von {job['title']}: {e}")

    analyzed_jobs.sort(key=lambda x: x['score'], reverse=True)
    send_email_report(analyzed_jobs)

if __name__ == "__main__":
    asyncio.run(main())
