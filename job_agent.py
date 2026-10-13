import os
import sys
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any

# PDF Extraktion
from pypdf import PdfReader

# Apify offizielle Python-Bibliothek
from apify_client import ApifyClient

# LangChain Importe
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 📍 AKTUALISIERTER WOHNORT
MY_HOME_LOCATION = "Ditzingen, 71254"

# 1. Pydantic-Struktur (recommended_action vollständig entfernt)
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt. Berücksichtige die Fahrzeiten (Auto und ÖPNV) negativ, falls beide über 50 Minuten liegen!")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter). Erwähne kurz die Erreichbarkeit mit Auto vs. ÖPNV.")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    location: str = Field(description="Der Arbeitsort (z.B. Stadt, Hybrid, Remote) des Jobs.")
    route_distance: str = Field(description="Die geschätzte Entfernung von Ditzingen zum Arbeitsort in Kilometern (z.B. '25 km').")
    route_duration: str = Field(description="Die geschätzte Reisezeit mit dem Auto bei normalem Verkehr (z.B. '30 Min. Autofahrt').")
    route_transit: str = Field(description="Die geschätzte Reisezeit mit Bahn/ÖPNV vom Bahnhof Ditzingen zum Ziel (z.B. '45 Min. mit ÖPNV').")
    salary_source: str = Field(description="Das geschätzte oder verifizierte Gehalt / Gehaltsspanne der Stelle (z.B. '80.000 € - 95.000 €' oder 'N/A').")
    post_date: str = Field(description="Das genaue Veröffentlichungsdatum der Stelle (z.B. 'Vor 3 Tagen' oder das exakte Datum, falls vorhanden).")
    job_summary: str = Field(description="Eine ultrakurze, prägnante Zusammenfassung der Stelle und Aufgaben (max. 2 Sätze).")
    required_qualifications: List[str] = Field(description="Eine Liste der wichtigsten 3-5 notwendigen Qualifikationen, Fähigkeiten oder Technologien.")
    url: str = Field(description="Die vollständige URL-Adresse zur Bewerbung oder zur Stellenausschreibung.")

# 2. Funktion zum Extrahieren des Profils (Strikter Abbruch)
def extract_profile_from_pdf(pdf_path="lebenslauf.pdf"):
    print(f"📄 Extrahiere Profil aus lokaler Datei '{pdf_path}'...")
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Die Datei existiert nicht im Repository.")
        reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if not text.strip():
            raise ValueError("Die PDF-Datei enthält keinen lesbaren Text.")
        print(f"✅ Lebenslauf erfolgreich gelesen ({len(text)} Zeichen extrahiert).")
        return text
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER beim Laden des Lebenslaufs: {e}")
        print("🛑 Der Suchlauf wird abgebrochen, da kein gültiger Lebenslauf vorliegt.")
        sys.exit(1)

# 3. Job-Abfrage über den stabilen Python Apify-Client (Mit erweiterten Filtern)
async def fetch_jobs_via_apify(search_titles: List[str], location: str, distance_km: int = 30, test_mode: bool = False) -> Dict[str, Any]:
    distance_miles = int(distance_km * 0.62)
    limit = 4 if test_mode else 50
    
    print(f"🔌 [TEST_MODE={test_mode}] Rufe aktuelle Jobs (Limit: {limit}) für {search_titles} im Umkreis von {distance_km} km um '{location}' ab...")
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        print("❌ KRITISCHER FEHLER: APIFY_API_TOKEN nicht in Secrets hinterlegt!")
        sys.exit(1)
        
    client = ApifyClient(token)
    try:
        run_input = {
            "titles": search_titles,      
            "locations": [location],       
            "distance": distance_miles,    
            "limit": limit,                   
            "publishedAt": "r604800", # Letzte 7 Tage
            "enrichCompany": True,
            "jobType": [
                "full-time",
                "part-time",
                "contract"
            ],
            "experienceLevel": [
                "entry",
                "associate",
                "mid-senior"
            ]
        }
        
        print(f"📥 Starte den LinkedIn-Scraper-Task...")
        run = client.actor("bebity/linkedin-jobs-scraper").call(run_input=run_input, timeout_secs=180)
        
        print("📦 Lade strukturierte Ergebnisse herunter...")
        dataset = client.dataset(run["defaultDatasetId"])
        dataset_list = dataset.list_items()
        dataset_items = dataset_list.items
        
        linkedin_search_totals = "Unbekannt"
        try:
            kvs = client.key_value_store(run["defaultKeyValueStoreId"])

            # Wir versuchen bis zu 3 Mal im Abstand von 1,5 Sekunden,
            # den Wert zu laden, um dem Apify-Server Zeit zum Schreiben zu geben.
            for attempt in range(3):
                search_totals_record = kvs.get_record("SEARCH_TOTALS")
                if search_totals_record and "value" in search_totals_record:
                    linkedin_search_totals = str(search_totals_record["value"])
                    break # Erfolgreich ausgelesen!
                print("⏳ Warte auf die Bereitstellung der LinkedIn-Statistiken...")
                import time
                time.sleep(1.5)

        except Exception as e:
            print(f"⚠️ Statistiken konnten nicht geladen werden: {e}")
        
        jobs = []
        for item in dataset_items[:limit]:
            jobs.append({
                "title": item.get("title", item.get("job_title", "Kein Titel")),
                "company": item.get("companyName", item.get("company", "Unbekannte Firma")),
                "description": item.get("description", "")[:1000],
                "location": item.get("location", "N/A"),
                "url": item.get("url", item.get("jobUrl", "https://linkedin.com")),
                "salary": item.get("salary", item.get("salarySource", "N/A")),
                "postDate": item.get("postDate", item.get("postedAt", "N/A"))
            })
        scraped_total = len(jobs)

        print(f"📊 --- STATISTIK ---")
        print(f"🌍 LinkedIn Gesamt-Treffer: {linkedin_search_totals}")
        print(f"📥 Scraper Importiert: {scraped_total}")
        print(f"--------------------")
        return {
            "jobs": jobs,
            "linkedin_total": linkedin_search_totals,
            "scraped_total": scraped_total
        }
        
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER beim Job-Abruf über Apify: {e}")
        sys.exit(1)

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
        ("system", "Du bist ein präziser HR-Analyst und Geografie-Experte für den Großraum Stuttgart (inkl. Ditzingen). "
                   "Gleiche den Job mit dem Lebenslauf-Profil des Kandidaten ab. "
                   "Schätze basierend auf deinem geographischen Wissen die Entfernung von Ditzingen (71254) zum angegebenen Arbeitsort "
                   "sowie die ungefähre Fahrtzeit mit dem Auto und mit öffentlichen Verkehrsmitteln (ÖPNV) realistisch ein. "
                   "Berücksichtige diese Pendelzeit negativ im Match-Score, falls sie zu lang ist.\n{format_instructions}"),
        ("user", "Lebenslauf des Kandidaten:\n{profile}\n\n"
                 "Wohnort des Kandidaten: {home_location}\n\n"
                 "Jobangebot:\nTitel: {title}\nFirma: {company}\nBeschreibung: {description}\nOrt: {location}\n"
                 "Gehaltsdaten (Apify): {salary_info}\nVeröffentlicht am/vor (Apify): {post_info}\nURL: {url}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

# 5. E-Mail Versendungsfunktion (HTML - Badge & empfohlenen Handlungsteil entfernt)
def send_email_report(jobs_list: List[Dict[str, Any]], linkedin_total: str, scraped_total: int, test_mode: bool = False):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER")
    
    if not sender_email or not sender_password or not receiver_email:
        print("⚠️ E-Mail-Konfiguration unvollständig. Überspringe Versand.")
        return

    msg = MIMEMultipart("alternative")
    subject_prefix = "[TEST-RUN] " if test_mode else ""
    msg["Subject"] = f"🎯 {subject_prefix}Dein täglicher Job-Agent: {len(jobs_list)} neue Matches"
    msg["From"] = f"Dein AI-Job-Agent <{sender_email}>"
    msg["To"] = receiver_email

    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            h2 {{ color: #1e3a8a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
            .job-card {{ background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin-bottom: 20px; }}
            .score {{ font-size: 18px; font-weight: bold; float: right; }}
            .section-title {{ font-weight: bold; color: #1f2937; margin-top: 12px; margin-bottom: 4px; font-size: 13px; text-transform: uppercase; }}
            .summary-text {{ color: #4b5563; font-size: 14px; margin-top: 2px; margin-bottom: 10px; }}
            .qualifications-list {{ margin: 0; padding-left: 20px; color: #4b5563; font-size: 13px; }}
            .reason {{ font-style: italic; color: #1e3a8a; background-color: #eff6ff; padding: 10px; border-left: 3px solid #3b82f6; border-radius: 0 4px 4px 0; margin-top: 12px; margin-bottom: 15px; font-size: 13px; }}
            .btn-apply {{ display: inline-block; background-color: #1e3a8a; color: white !important; padding: 10px 16px; text-decoration: none; border-radius: 4px; font-size: 13px; font-weight: bold; margin-top: 10px; }}
            .location-text {{ color: #4b5563; font-size: 13px; margin-top: 4px; display: inline-block; }}
            .route-info {{ background-color: #f3f4f6; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #374151; margin-top: 8px; display: block; line-height: 1.5; }}
            .meta-info {{ margin-top: 10px; font-size: 13px; color: #4b5563; display: block; }}
            .footer {{ font-size: 11px; color: #9ca3af; text-align: center; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎯 Job-Agent Match-Report (Smart-Details)</h2>
            <p>Hallo! Hier sind deine aktuellen Stellenausschreibungen inklusive Gehaltsspanne, Veröffentlichungsdatum und Arbeitsweg:</p>
            
            <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #1e3a8a; margin-bottom: 20px;">
                📊 <b>Echtzeit-Stellenmarkt-Statistik für dich:</b><br>
                • Insgesamt passende Stellen auf LinkedIn vorhanden: <b>{linkedin_total}</b><br>
                • Davon für diesen Bericht heruntergeladen und analysiert: <b>{scraped_total}</b> (dein Limit: {'4 (Testmodus)' if test_mode else '50'})
            </div>
    """

    for job in jobs_list:
        score = job['score']
        qual_items_html = "".join([f"<li>{q}</li>" for q in job.get('required_qualifications', [])])

        # Badge-Generierung & Recommended Action sind hier vollständig aus dem HTML-Template entfernt!
        html_template += f"""
            <div class="job-card">
                <span class="score" style="color: {'#059669' if score >= 70 else '#d97706' if score >= 40 else '#dc2626'}">{score}% Match</span>
                <strong style="font-size: 17px;">{job['job_title']}</strong><br>
                <span style="color: #4b5563; font-size: 14px;">{job['company']}</span><br>
                <span class="location-text">📍 {job.get('location', 'N/A')}</span><br>
                
                <div class="route-info">
                    🚗 <b>Auto Arbeitsweg:</b> {job.get('route_distance', 'N/A')} (ca. {job.get('route_duration', 'N/A')})<br>
                    🚌 <b>ÖPNV Arbeitsweg:</b> {job.get('route_transit', 'N/A')}
                </div>
                
                <div class="meta-info">
                    💰 <b>Gehalt:</b> {job.get('salary_source', 'Nicht angegeben')}<br>
                    📅 <b>Veröffentlicht:</b> {job.get('post_date', 'N/A')}
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

    html_template += f"""
            <div class="footer">
                Diese E-Mail wurde vollautomatisch über den PDF-Abgleich generiert.<br>
                Heimat-Wohnort: {MY_HOME_LOCATION}<br>
                Modus: {'🔴 TEST MODUS AKTIV' if test_mode else '🟢 PRODUKTIONS-MODUS'}<br>
                Host: GitHub Actions | Model: openai/gpt-oss-20b on Groq.
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
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    # A. Strikte PDF-Extraktion am Anfang
    user_profile = extract_profile_from_pdf("lebenslauf.pdf")
    
    # B. Wunschtitel
    WUNSCH_TITEL = [
        "Data Scientist", 
        "AI Engineer", 
        "Entwickler Regelungsfunktionen",
        "Python Backend Developer"
    ]
    
    # C. Jobs abrufen (Schnittstellen-Daten werden als Dict geliefert)
    scraper_result = await fetch_jobs_via_apify(
        search_titles=WUNSCH_TITEL, 
        location=MY_HOME_LOCATION, 
        distance_km=30,
        test_mode=test_mode
    )
    
    raw_jobs = scraper_result["jobs"]
    linkedin_total = scraper_result["linkedin_total"]
    scraped_total = scraper_result["scraped_total"]
    
    if not raw_jobs:
        print("❌ Keine Jobs empfangen. Breche ab.")
        sys.exit(1)

    match_chain = create_matching_chain()
    analyzed_jobs = []

    print("\n🧠 Starte KI-Abgleich und geografische Pendel-Analyse...")
    for job in raw_jobs:
        if not test_mode:
            await asyncio.sleep(1.5)
        
        try:
            result = match_chain.invoke({
                "profile": user_profile,
                "home_location": MY_HOME_LOCATION,
                "title": job["title"],
                "company": job["company"],
                "description": job["description"],
                "location": job["location"],
                "salary_info": job.get("salary", "N/A"),
                "post_info": job.get("postDate", "N/A"),
                "url": job["url"]
            })
            analyzed_jobs.append(result)
            
        except Exception as e:
            print(f"⚠️ Fehler bei der Analyse von {job['title']}: {e}")

    analyzed_jobs.sort(key=lambda x: x['score'], reverse=True)
    
    send_email_report(
        jobs_list=analyzed_jobs, 
        linkedin_total=linkedin_total, 
        scraped_total=scraped_total, 
        test_mode=test_mode
    )

if __name__ == "__main__":
    asyncio.run(main())
