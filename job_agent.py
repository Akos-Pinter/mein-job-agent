import os
import sys
import asyncio
import requests
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

# 📍 TRAGEN SIE HIER IHREN WOHNORT EIN
MY_HOME_LOCATION = "Stuttgart"

# 1. Pydantic-Struktur für das LLM-Output
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt. Berücksichtige die Fahrzeiten (Auto und ÖPNV) negativ, falls beide über 50 Minuten liegen!")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter). Erwähne kurz die Erreichbarkeit mit Auto vs. ÖPNV.")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    location: str = Field(description="Der Arbeitsort (z.B. Stadt, Hybrid, Remote) des Jobs.")
    route_distance: str = Field(description="Die berechnete Entfernung zum Wohnort des Kandidaten in Kilometern (z.B. '42.1 km').")
    route_duration: str = Field(description="Die berechnete Reisezeit mit dem Auto (z.B. '35 Minuten Autofahrt').")
    route_transit: str = Field(description="Die berechnete Reisezeit mit Bus & Bahn (z.B. '48 Minuten mit ÖPNV').")
    job_summary: str = Field(description="Eine ultrakurze, prägnante Zusammenfassung der Stelle und Aufgaben (max. 2 Sätze).")
    required_qualifications: List[str] = Field(description="Eine Liste der wichtigsten 3-5 notwendigen Qualifikationen, Fähigkeiten oder Technologien.")
    url: str = Field(description="Die vollständige URL-Adresse zur Bewerbung oder zur Stellenausschreibung.")
    recommended_action: str = Field(description="Empfehlung: 'Bewerben', 'Vielleicht' oder 'Ignorieren'.")

# 2. Hilfsfunktion zur Geokodierung einer Adresse (Nominatim OSM)
def get_coordinates(address):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {
            "User-Agent": "UniqueJobAgentStuttgartUser_AkosPinter_v2" 
        }
        params = {"q": address, "format": "json", "limit": 1}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"⚠️ Geokodierung für '{address}' fehlgeschlagen: {e}")
    return None


# 3. Hilfsfunktion zur Berechnung von Distanz & Auto-Fahrzeit (OSRM)
def calculate_route_osrm(origin_coords, dest_coords):
    try:
        coords_str = f"{origin_coords},{origin_coords[0]};{dest_coords},{dest_coords[0]}"
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}"
        params = {"overview": "false"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and "routes" in data and len(data["routes"]) > 0:
            route = data["routes"][0]
            distance_km = round(route["distance"] / 1000.0, 1)
            duration_mins = round(route["duration"] / 60.0)
            return distance_km, duration_mins
    except Exception as e:
        print(f"⚠️ Auto-Routenberechnung fehlgeschlagen: {e}")
    return None

# 4. Hilfsfunktion zur echten ÖPNV-Fahrzeit-Berechnung (DB HAFAS API)
def calculate_transit_hafas(origin_name, dest_name):
    print(f"🚌 Berechne ÖPNV-Verbindung von '{origin_name}' nach '{dest_name}'...")
    try:
        url = "https://v5.db.transport.rest/journeys"
        params = {
            "from": origin_name,
            "to": dest_name,
            "results": 1,
            "language": "de"
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if data and "journeys" in data and len(data["journeys"]) > 0:
            journey = data["journeys"][0]
            legs = journey.get("legs", [])
            if legs:
                start_time = legs[0].get("departure")
                end_time = legs[-1].get("arrival")
                
                from datetime import datetime
                fmt = "%Y-%m-%dT%H:%M:%S%z"
                dt_start = datetime.strptime(start_time[:-3] + start_time[-2:], fmt)
                dt_end = datetime.strptime(end_time[:-3] + end_time[-2:], fmt)
                
                duration_mins = round((dt_end - dt_start).seconds / 60.0)
                changes = len(legs) - 1
                return f"{duration_mins} Minuten mit ÖPNV ({changes}x Umsteigen)"
    except Exception as e:
        print(f"⚠️ ÖPNV-Berechnung fehlgeschlagen: {e}")
    return "Unbekannt mit ÖPNV"

# 5. Funktion zum Extrahieren des Profils (Strikter Abbruch)
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

# 6. Job-Abfrage über den stabilen Python Apify-Client (Strikter Abbruch bei Fehlern!)
# Suchen und ersetzen Sie den bisherigen "fetch_jobs_via_apify" Block:

async def fetch_jobs_via_apify(search_titles: List[str], location: str, distance_km: int = 30):
    # Umrechnung von km in Meilen für die LinkedIn-API (1 km = 0.62 Meilen)
    distance_miles = int(distance_km * 0.62)

    print(f"🔌 Rufe aktuelle Jobs für {search_titles} im Umkreis von {distance_km} km um '{location}' ab...")

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        print("❌ KRITISCHER FEHLER: APIFY_API_TOKEN nicht in Secrets hinterlegt!")
        sys.exit(1)

    client = ApifyClient(token)
    try:
        # Konfiguration für die LinkedIn-API über Apify
        run_input = {
            "titles": search_titles,
            "locations": [location],
            "distance": distance_miles,
            "limit": 5,
            "publishedAt": "r604800" # Technischer Code für 'Letzte 7 Tage' (past 7 days)
        }

        print("📥 Starte den LinkedIn-Scraper-Task...")
        run = client.actor("bebity/linkedin-jobs-scraper").call(run_input=run_input, timeout_secs=180)

        print("📦 Lade strukturierte Ergebnisse herunter...")
        dataset_items = list(client.dataset(run["defaultDatasetId"]).list_items().items)

        jobs = []
        for item in dataset_items:
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
        print(f"❌ KRITISCHER FEHLER beim Job-Abruf über Apify: {e}")
        print("🛑 Der Suchlauf wird abgebrochen, da kein valides Job-Sourcing möglich ist.")
        sys.exit(1)

# 7. LangChain-Chain definieren (openai/gpt-oss-20b via Groq)
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
                   "Fülle alle JSON-Felder präzise aus. Berücksichtige die berechneten Fahrzeiten negativ im Match-Score, falls diese zu lang sind.\n{format_instructions}"),
        ("user", "Lebenslauf des Kandidaten:\n{profile}\n\n"
                 "Jobangebot:\nTitel: {title}\nFirma: {company}\nBeschreibung: {description}\nOrt: {location}\n"
                 "--- BERECHNETE ARBEITSWEGE ZUM KANDIDATEN ---\n"
                 "Entfernung: {distance_info}\nFahrzeit (Auto): {duration_info}\nReisezeit (ÖPNV): {transit_info}\nURL: {url}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

# 8. E-Mail Versendungsfunktion (HTML)
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
            .section-title { font-weight: bold; color: #1f2937; margin-top: 12px; margin-bottom: 4px; font-size: 13px; text-transform: uppercase; }
            .summary-text { color: #4b5563; font-size: 14px; margin-top: 2px; margin-bottom: 10px; }
            .qualifications-list { margin: 0; padding-left: 20px; color: #4b5563; font-size: 13px; }
            .reason { font-style: italic; color: #1e3a8a; background-color: #eff6ff; padding: 10px; border-left: 3px solid #3b82f6; border-radius: 0 4px 4px 0; margin-top: 12px; margin-bottom: 15px; font-size: 13px; }
            .btn-apply { display: inline-block; background-color: #1e3a8a; color: white !important; padding: 10px 16px; text-decoration: none; border-radius: 4px; font-size: 13px; font-weight: bold; }
            .location-text { color: #4b5563; font-size: 13px; margin-top: 4px; display: inline-block; }
            .route-info { background-color: #f3f4f6; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #374151; margin-top: 8px; display: block; line-height: 1.5; }
            .footer { font-size: 11px; color: #9ca3af; text-align: center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎯 Job-Agent Match-Report (Wohnort-Routenabgleich)</h2>
            <p>Hallo! Ich habe die Jobs direkt mit deinem Lebenslauf abgeglichen und die Reisedaten (Auto & ÖPNV) berechnet:</p>
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

        qual_items_html = "".join([f"<li>{q}</li>" for q in job.get('required_qualifications', [])])

        html_template += f"""
            <div class="job-card">
                <span class="score" style="color: {'#059669' if score >= 70 else '#d97706' if score >= 40 else '#dc2626'}">{score}% Match</span>
                <strong style="font-size: 17px;">{job['job_title']}</strong><br>
                <span style="color: #4b5563; font-size: 14px;">{job['company']}</span><br>
                <span class="location-text">📍 {job.get('location', 'N/A')}</span><br>
                
                <div class="route-info">
                    🚗 <b>Auto Arbeitsweg:</b> {job.get('route_distance', 'Unbekannt')} (ca. {job.get('route_duration', 'Unbekannt')})<br>
                    🚌 <b>ÖPNV Arbeitsweg:</b> {job.get('route_transit', 'Unbekannt')}
                </div>
                
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

    html_template += f"""
            <div class="footer">
                Diese E-Mail wurde vollautomatisch über den PDF-Abgleich generiert.<br>
                Berechneter Start-Wohnort: {MY_HOME_LOCATION}<br>
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

# 9. Haupt-Orchestrator
# Suchen und ersetzen Sie die gesamte "main" Funktion am Ende des Skripts:

async def main():
    # A. Strikte PDF-Extraktion am Anfang
    user_profile = extract_profile_from_pdf("lebenslauf.pdf")
    
    # B. Koordinaten des eigenen Wohnorts holen (Stuttgart)
    home_coords = get_coordinates(MY_HOME_LOCATION)
    if not home_coords:
        print(f"⚠️ Konnte keine Koordinaten für deinen Wohnort '{MY_HOME_LOCATION}' ermitteln!")
    
    # C. IHRE WUNSCH-JOBTITEL HIER DEFINIEREN
    WUNSCH_TITEL = [
        "Data Scientist", 
        "AI Engineer", 
        "Entwickler Regelungsfunktionen",
        "Python Backend Developer"
    ]
    
    # D. Jobs mit Umkreis 30 km um Stuttgart abrufen (Hartes Beenden bei Fehlern)
    raw_jobs = await fetch_jobs_via_apify(
        search_titles=WUNSCH_TITEL, 
        location=MY_HOME_LOCATION, 
        distance_km=30
    )
    
    if not raw_jobs:
        print("❌ Keine Jobs empfangen. Breche ab.")
        sys.exit(1)

    match_chain = create_matching_chain()
    analyzed_jobs = []
    
    print("\n🧠 Berechne Arbeitswege (Auto & ÖPNV) und starte KI-Abgleich...")
    for job in raw_jobs:
        await asyncio.sleep(1.5) # Wartet 1,5 Sek. vor jedem KI-Aufruf
        distance_str = "Unbekannt"
        duration_str = "Unbekannt"
        transit_str = "Unbekannt mit ÖPNV"
        
        # 1. Auto-Routenberechnung
        if home_coords:
            job_coords = get_coordinates(job["location"])
            if job_coords:
                route_data = calculate_route_osrm(home_coords, job_coords)
                if route_data:
                    distance_str = f"{route_data[0]} km"
                    duration_str = f"{route_data} Minuten"
        print(home_coords)
        print(job_coords)
        print(distance_str)
        print(duration_str)
        
        # 2. ÖPNV-Routenberechnung über DB HAFAS Schnittstelle
        if MY_HOME_LOCATION and job["location"] != "N/A":
            transit_str = calculate_transit_hafas(MY_HOME_LOCATION, job["location"])
            
    #     try:
    #         result = match_chain.invoke({
    #             "profile": user_profile,
    #             "title": job["title"],
    #             "company": job["company"],
    #             "description": job["description"],
    #             "location": job["location"],
    #             "distance_info": distance_str,
    #             "duration_info": duration_str,
    #             "transit_info": transit_str,
    #             "url": job["url"]
    #         })
            
    #         result["route_distance"] = distance_str
    #         result["route_duration"] = duration_str
    #         result["route_transit"] = transit_str
    #         analyzed_jobs.append(result)
            
    #     except Exception as e:
    #         print(f"⚠️ Fehler bei der Analyse von {job['title']}: {e}")

    # # Nach Score sortieren (Beste Matches zuerst)
    # analyzed_jobs.sort(key=lambda x: x['score'], reverse=True)
    # send_email_report(analyzed_jobs)

if __name__ == "__main__":
    asyncio.run(main())
