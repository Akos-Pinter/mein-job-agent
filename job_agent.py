import os
import sys
import json
import base64
import asyncio
import requests
import smtplib
from datetime import datetime, timedelta
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

# 📍 IHR WOHNORT
MY_HOME_LOCATION = "Ditzingen, Baden-Württemberg, Germany"
HISTORY_FILE = "history_jobs.json"
RAW_JOBS_FILE = "raw_jobs.json"
JOB_LIFETIME_DAYS = 14 


# 1. Pydantic-Struktur für das LLM-Output
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt. Berücksichtige die Fahrzeiten (Auto und ÖPNV) negativ, falls beide über 50 Minuten liegen!")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter). Erwähne kurz die Erreichbarkeit mit Auto vs. ÖPNV.")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    company_employee_count: str = Field(description="Die Mitarbeiteranzahl der Firma als formatiertes Textfeld (z.B. '240.000 Mitarbeiter' oder 'N/A' falls unbekannt).")
    location: str = Field(description="Der Arbeitsort (z.B. Stadt, Hybrid, Remote) des Jobs.")
    route_distance: str = Field(description="Die geschätzte Entfernung von Ditzingen zum Arbeitsort in Kilometern (z.B. '25 km').")
    route_duration: str = Field(description="Die geschätzte Reisezeit mit dem Auto bei normalem Verkehr (z.B. '30 Min. Autofahrt').")
    route_transit: str = Field(description="Die geschätzte Reisezeit mit Bahn/ÖPNV vom Bahnhof Ditzingen zum Ziel (z.B. '45 Min. mit ÖPNV').")
    contract_type: str = Field(description="Der Vertragstyp der Stelle (z.B. 'Full-time', 'Part-time', 'Contract' oder 'N/A').")
    experience_level: str = Field(description="Die geforderte Erfahrungsstufe der Stelle (z.B. 'Mid-Senior level', 'Entry level', 'Associate' oder 'N/A').")
    salary_source: str = Field(description="Das geschätzte oder verifizierte Gehalt / Gehaltsspanne der Stelle (z.B. '80.000 € - 95.000 €' oder 'N/A').")
    post_date: str = Field(description="Das genaue Veröffentlichungsdatum der Stelle (z.B. 'Vor 3 Tagen' oder das exakte Datum, falls vorhanden).")
    job_summary: str = Field(description="Eine ultrakurze, prägnante Zusammenfassung der Stelle und Aufgaben (max. 2 Sätze).")
    required_qualifications: List[str] = Field(description="Eine Liste der wichtigsten 3-5 notwendigen Qualifikationen, Fähigkeiten oder Technologien.")
    url: str = Field(description="Die vollständige URL-Adresse zur Bewerbung oder zur Stellenausschreibung.")


# 2. Hilfsfunktionen zum Laden und Speichern der Verlaufsdaten (JSON)
def load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"📁 Verlauf geladen: {len(data)} bereits analysierte Jobs gefunden.")
                return data
        except Exception as e:
            print(f"⚠️ Fehler beim Laden des Verlaufs: {e}. Starte mit leeren Daten.")
    return []


def save_json_file(data: List[Dict[str, Any]], filepath: str):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Datei erfolgreich aktualisiert: '{filepath}' ({len(data)} Einträge gesichert).")
    except Exception as e:
        print(f"❌ Fehler beim Schreiben der Datei '{filepath}': {e}")


# 3. Funktion zum Extrahieren des Profils (Strikter Abbruch)
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
        sys.exit(1)


# 4.1 Fallback-Sourcing über RapidAPI JSearch
def fetch_jobs_via_jsearch(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 25) -> Dict[str, Any]:
    print(f"🔌 ⚙️ [FALLBACK AKTIV] Starte Umkreis-Abfrage ({distance_km} km) über RapidAPI JSearch...")
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        print("❌ KRITISCHER FEHLER: RAPIDAPI_KEY nicht in den Secrets hinterlegt! Fallback unmöglich.")
        sys.exit(1)
        
    url = "https://jsearch.p.rapidapi.com/search-v2"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
	
    required_pages = min(max(1, -(-fetch_limit // 10)), 20)
    titles_query = " OR ".join(f'"{title}"' for title in search_titles)
    params = {
        "query": f"{titles_query} in {location}",
		"country": "de",
        "num_pages": str(required_pages),
		"radius": str(distance_km),
		#"employment_types": "FULLTIME,PARTTIME,CONTRACTOR",
		#"job_requirements": "under_3_years_experience,3_to_7_years_experience"
		}
	
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        response_data = response.json()
        
        raw_items = response_data.get("data", []).get("jobs", [])
        total_hits = response_data.get("estimated_results", response_data.get("total_results", "Unbekannt"))
        print(f"📥 JSearch lieferte {len(raw_items)} gefilterte Treffer im {distance_km}-km-Radius von insgesamt ca. {total_hits} Stellen.")
		
        jobs = []
        for item in raw_items:
            # Gehaltsspanne formatieren
            min_sal = item.get("job_min_salary")
            max_sal = item.get("job_max_salary")
            currency = item.get("job_salary_currency", "EUR")
            salary_str = f"{min_sal:,} - {max_sal:,} {currency}".replace(",", ".") if min_sal and max_sal else "N/A"
            
            # Mapping der Felder zu unserem einheitlichen Format
            jobs.append({
                "title": item.get("job_title", "Kein Titel"),
                "company": item.get("employer_name", "Unbekannte Firma"),
                "description": item.get("job_description", "")[:1000],
                "location": f"{item.get('job_city', '')}, {item.get('job_state', '')}".strip(", "),
                "url": item.get("job_apply_link", item.get("job_google_link", "https://linkedin.com")),
                "salary": salary_str,
                "postDate": item.get("job_posted_at_datetime_utc", "N/A")[:10] if item.get("job_posted_at_datetime_utc") else "N/A",
                "contractType": "Full-time" if item.get("job_employment_type") == "FULLTIME" else item.get("job_employment_type", "N/A"),
                "experienceLevel": "Mid-Senior level" if item.get("job_required_experience", {}).get("required_experience_in_months") else "Entry level",
                "companyEmployeeCount": "N/A"
            })
            
        return {
            "jobs": jobs,
            "linkedin_total": f"ca. {total_hits} (JSearch)",
            "scraped_total": len(jobs)
        }
    except Exception as e:
        print(f"❌ KRITISCH: Fallback über JSearch ebenfalls fehlgeschlagen: {e}")
        sys.exit(1)


# 4.2 Job-Abfrage über den stabilen Python Apify-Client (Mit automatischem JSearch-Fallback)
async def fetch_jobs_via_apify(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 50) -> Dict[str, Any]:
    distance_miles = int(distance_km * 0.62)
    
    print(f"🔌 Rufe aktuelle Jobs (Sourcing-Limit: {fetch_limit}) für {search_titles} im Umkreis von {distance_km} km um '{location}' ab...")
    token = os.getenv("APIFY_API_TOKEN")
    
    if not token:
        print("⚠️ APIFY_API_TOKEN nicht gefunden! Weiche direkt auf JSearch-Fallback aus.")
        return fetch_jobs_via_jsearch(search_titles=search_titles, location=location, distance_km=distance_km, fetch_limit=fetch_limit)
        
    client = ApifyClient(token)
    try:
        run_input = {
            "titles": search_titles,      
            "locations": [location],       
            "distance": distance_miles,    
            "limit": fetch_limit,             
            "publishedAt": "r604800", # Letzte 7 Tage
            "enrichCompany": True,
            "jobType": ["full-time", "part-time", "contract"],
            "experienceLevel": ["entry", "associate", "mid-senior"]
        }
        
        print("📥 Starte den LinkedIn-Scraper-Task...")
        run = client.actor("bebity/linkedin-jobs-scraper").call(run_input=run_input, timeout_secs=180)
        
        print("📦 Lade strukturierte Ergebnisse herunter...")
        dataset = client.dataset(run["defaultDatasetId"])
        dataset_list = dataset.list_items()
        dataset_items = dataset_list.items
        
        linkedin_search_totals = "Unbekannt"
        try:
            kvs = client.key_value_store(run["defaultKeyValueStoreId"])
            for _ in range(2):
                search_totals_record = kvs.get_record("SEARCH_TOTALS")
                if search_totals_record and "value" in search_totals_record:
                    linkedin_search_totals = str(search_totals_record["value"])
                    break
                import time
                time.sleep(1.5)
        except Exception:
            pass
        
        jobs = []
        for item in dataset_items:
            jobs.append({
                "title": item.get("title", item.get("job_title", "Kein Titel")),
                "company": item.get("companyName", item.get("company", "Unbekannte Firma")),
                "description": item.get("description", "")[:1000],
                "location": item.get("location", "N/A"),
                "url": item.get("url", item.get("jobUrl", "https://linkedin.com")),
                "salary": item.get("salary", item.get("salarySource", "N/A")),
                "postDate": item.get("postDate", item.get("postedAt", "N/A")),
                "contractType": item.get("contractType", "N/A"),
                "experienceLevel": item.get("experienceLevel", "N/A"),
                "companyEmployeeCount": item.get("companyEmployeeCount", "N/A")
            })
            
        scraped_total = len(jobs)
        
        return {
            "jobs": jobs,
            "linkedin_total": linkedin_search_totals,
            "scraped_total": scraped_total
        }
        
    except Exception as e:
        print(f"⚠️ Apify-Fehler aufgetreten: {e}. Weiche automatisch auf JSearch aus...")
        return fetch_jobs_via_jsearch(search_titles=search_titles, location=location, distance_km=distance_km, fetch_limit=fetch_limit)

# 4.3 Kostenlose Abfrage der offiziellen Jobbörse (Bundesagentur für Arbeit)
async def fetch_jobs_via_arbeitsagentur(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 25) -> Dict[str, Any]:
    print(f"🔌 🇩🇪 [FREE API ACTIVE] Starte synchrone V6-Abfrage ({distance_km} km) über Bundesagentur für Arbeit (Limit pro Suche: {fetch_limit})...")
    
    headers = {
        "X-API-Key": "jobboerse-jobsuche"
    }
    
    zip_code = "".join(filter(str.isdigit, location)) or '71254'
    jobs = []
    
    for title in search_titles:
        search_url = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs"
        params = {
            "was": title,
            "wo": zip_code,
            "umkreis": str(distance_km),
            "arbeitszeit": "vz",             # 'vz' (Vollzeit)
            #"veroeffentlichtseit": "14",     # Letzte 14 Tage
            "befristung": "2",               # Nur unbefristete Festanstellungen
            "zeitarbeit": "false",           # Keine Zeitarbeit
            "pav": "false",                  # Keine Vermittler
            "angebotsart": "1",              # Nur reguläre Jobs
            "page": "1",
            "size": str(fetch_limit)         # Limit pro Keyword-Abfrage
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=15)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            search_results = response.json()
            
            raw_items = search_results.get("ergebnisliste", [])
            print(f"   ↳ '{title}': {len(raw_items)} Treffer gefunden (API size: {fetch_limit}).")
            
            for item in raw_items:
                ref_nr = item.get("referenznummer") or item.get("refnr")
                if not ref_nr:
                    continue
                
                # Duplikate über die verschiedenen Suchen hinweg vermeiden
                if any(job["url"].endswith(ref_nr) for job in jobs):
                    continue
                
                # Detailabfrage
                encoded_ref = base64.b64encode(ref_nr.encode()).decode()
                detail_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{encoded_ref}"
                
                salary_val = item.get("gehalt", "N/A")
                description = ""
                contract_type = "Full-time"
                experience_level = "Mid-Senior level"
                
                try:
                    detail_resp = requests.get(detail_url, headers=headers, timeout=6)
                    if detail_resp.status_code == 200:
                        detail_data = detail_resp.json()
                        description = detail_data.get("stellenangebotsBeschreibung", "")
                        is_temporary = detail_data.get("befristung", 2) == 1
                        contract_type = "Contract" if is_temporary else "Full-time"
                except Exception:
                    pass
                
                if not description:
                    description = f"Stelle als {item.get('beruf', 'Fachkraft')} in {item.get('arbeitsort', {}).get('ort', 'N/A')}."

                jobs.append({
                    "title": item.get("beruf", "Kein Titel"),
                    "company": item.get("arbeitgeber", "Unbekannte Firma"),
                    "description": description[:1000],
                    "location": item.get("arbeitsort", {}).get("ort", "N/A"),
                    "url": item.get("externeUrl") or f"https://www.arbeitsagentur.de/jobsuche/job/{ref_nr}",
                    "salary": salary_val if salary_val else "N/A",
                    "postDate": item.get("modifikationsDatum", "N/A"),
                    "contractType": contract_type,
                    "experienceLevel": experience_level,
                    "companyEmployeeCount": "N/A"
                })
                
        except Exception as e:
            print(f"⚠️ Warnung bei Abfrage von '{title}': {e}")
            continue

    scraped_total = len(jobs)
    print(f"📥 Insgesamt {scraped_total} eindeutige Jobs von der Bundesagentur geladen.")
    
    return {
        "jobs": jobs,
        "linkedin_total": f"{scraped_total} (Arbeitsagentur)",
        "scraped_total": scraped_total
    }


# 5. LangChain-Chain definieren (openai/gpt-oss-20b via Groq)
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
                 "Jobangebot:\n"
                 "Titel: {title}\n"
                 "Firma: {company}\n"
                 "Mitarbeiterzahl: {employee_count}\n"
                 "Beschreibung: {description}\n"
                 "Ort: {location}\n"
                 "Vertragstyp: {contract_type}\n"
                 "Erfahrungsstufe: {exp_level}\n"
                 "Gehaltsdaten: {salary_info}\n"
                 "Veröffentlicht: {post_info}\n"
                 "URL: {url}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


# 6. E-Mail Versendungsfunktion (HTML)
def send_email_report(new_jobs: List[Dict[str, Any]], old_jobs: List[Dict[str, Any]], linkedin_total: str, scraped_total: int, test_mode: bool = False):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER")
    
    if not sender_email or not sender_password or not receiver_email:
        print("⚠️ E-Mail-Konfiguration unvollständig. Überspringe Versand.")
        return
    msg = MIMEMultipart("alternative")
    subject_prefix = "[TEST-RUN] " if test_mode else ""
    
    full_jobs_list = new_jobs + old_jobs
    full_jobs_list.sort(key=lambda x: x.get('score', 0), reverse=True)
    msg["Subject"] = f"🎯 {subject_prefix}Dein Job-Agent: {len(new_jobs)} neue | {len(full_jobs_list)} Gesamt-Matches"
    msg["From"] = f"Dein AI-Job-Agent <{sender_email}>"
    msg["To"] = receiver_email
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            h2 {{ color: #1e3a8a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
            .job-card {{ background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin-bottom: 20px; position: relative; }}
            .new-badge {{ position: absolute; top: 15px; left: 15px; background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; text-transform: uppercase; }}
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
                • Insgesamt passende Stellen auf LinkedIn vorhanden: <b>{linkedin_total if not test_mode else 'Inaktiv (im Testmodus)'}</b><br>
                • Davon für diesen Bericht live geladen: <b>{scraped_total}</b><br>
                • Neu hinzugekommene und analysierte Treffer: <b style="color: #ef4444;">{len(new_jobs)}</b><br>
                • Gesamtanzahl in deinem Portfolio: <b>{len(full_jobs_list)}</b> (Aufbewahrung: {JOB_LIFETIME_DAYS} Tage)
            </div>
    """
    for job in full_jobs_list:
        score = job['score']
        qual_items_html = "".join([f"<li>{q}</li>" for q in job.get('required_qualifications', [])])
        
        is_new = any(nj['url'] == job['url'] for nj in new_jobs)
        new_badge_html = '<span style="background-color: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; text-transform: uppercase; margin-right: 8px; display: inline-block; vertical-align: middle; white-space: nowrap;">NEU ✨</span>' if is_new else ''
        emp_count = job.get('company_employee_count', 'N/A')
        formatted_company_display = f"{job['company']} ({emp_count})" if emp_count != "N/A" else job['company']
        html_template += f"""
            <div class="job-card">
                
                <!-- 💡 Titelzeile mit linksbündigem Titel (inkl. Badge) und rechtsbündigem Match-Score -->
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 8px;">
                    
                    <!-- Linke Seite: NEU-Badge + Jobtitel fließen nahtlos ineinander -->
                    <div style="display: flex; align-items: center; text-align: left; padding-right: 15px;">
                        {new_badge_html}
                        <strong style="font-size: 17px; color: #1e3a8a; vertical-align: middle;">
                            {job['job_title']}
                        </strong>
                    </div>
                    
                    <span style="font-size: 18px; font-weight: bold; color: {'#059669' if score >= 70 else '#d97706' if score >= 40 else '#dc2626'}; text-align: right; white-space: nowrap; margin-left: auto;">
                        {score}% Match
                    </span>
                </div>
                
                <span style="color: #4b5563; font-size: 14px;">{formatted_company_display}</span><br>
                <span class="location-text">📍 {job.get('location', 'N/A')}</span><br>
                
                <div class="route-info">
                    🚗 <b>Auto Arbeitsweg:</b> {job.get('route_distance', 'N/A')} (ca. {job.get('route_duration', 'N/A')})<br>
                    🚌 <b>ÖPNV Arbeitsweg:</b> {job.get('route_transit', 'N/A')}
                </div>
                
                <div class="meta-info">
                    💼 <b>Vertragstyp & Ebene:</b> {job.get('contract_type', 'N/A')} | {job.get('experience_level', 'N/A')}<br>
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


# 7. Haupt-Orchestrator
async def main():
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    FETCH_LIMIT = 25
    ANALYZE_LIMIT = 1 if test_mode else 25
    
    # A. Strikte PDF-Extraktion am Anfang
    user_profile = extract_profile_from_pdf("lebenslauf.pdf")
    
    # B. Verlauf laden
    history_data = load_history()
    
    # C. Wunschtitel
    WUNSCH_TITEL = [
        "Data Scientist", 
        "AI Engineer"
		]
    
    # D. Aktuelle Live-Stellen abrufen (Fällt bei Apify-Fehlern automatisch auf JSearch zurück)
    scraper_result = await fetch_jobs_via_arbeitsagentur(
        search_titles=WUNSCH_TITEL, 
        location=MY_HOME_LOCATION, 
        distance_km=30,
        fetch_limit=FETCH_LIMIT
    )
    
    raw_jobs = scraper_result["jobs"]
    linkedin_total = scraper_result["linkedin_total"]
    scraped_total = scraper_result["scraped_total"]
    
    if not raw_jobs:
        print("❌ Keine Jobs empfangen. Breche ab.")
        sys.exit(1)
        
    # E. Die rohen Sourcing-Ergebnisse abspeichern
    save_json_file(raw_jobs, RAW_JOBS_FILE)
    
    # F. SÄUBERUNGSWEICHE: "Time-to-Live" (TTL) statt Live-Abgleich
    cleaned_history = []
    current_date = datetime.now()
    for job in history_data:
        analyzed_at_str = job.get("analyzed_at")
        if analyzed_at_str:
            analyzed_date = datetime.strptime(analyzed_at_str, "%Y-%m-%d")
            if current_date - analyzed_date <= timedelta(days=JOB_LIFETIME_DAYS):
                cleaned_history.append(job)
        else:
            # Fallback für ganz alte Jobs ohne Zeitstempel (Erhalten einmalig den heutigen Stempel)
            job["analyzed_at"] = current_date.strftime("%Y-%m-%d")
            cleaned_history.append(job)
		
    removed_count = len(history_data) - len(cleaned_history)
    if removed_count > 0:
        print(f"🧹 TTL-Säuberung: {removed_count} veraltete Jobs (älter als {JOB_LIFETIME_DAYS} Tage) wurden gelöscht.")
        
    # G. DUPLIKATE FILTERN: Nur neue URLs erfassen
    known_urls = {job['url'] for job in cleaned_history}
    new_unseen_jobs = [job for job in raw_jobs if job['url'] not in known_urls]
    
    print(f"🔍 Filter-Ergebnis: Von {len(raw_jobs)} live gefundenen Jobs sind {len(new_unseen_jobs)} komplett neu.")
    
    # H. KI-LIMITIERUNG ANWENDEN
    jobs_to_analyze = new_unseen_jobs[:ANALYZE_LIMIT]
    
    if new_unseen_jobs and len(new_unseen_jobs) > ANALYZE_LIMIT:
        print(f"⚠️ Limit erreicht: Es gibt {len(new_unseen_jobs)} neue Jobs. Es werden jedoch nur die ersten {ANALYZE_LIMIT} davon heute analysiert, um Ressourcen zu sparen.")
        
    match_chain = create_matching_chain()
    new_analyzed_jobs = []
    
    # I. Nur die begrenzten neuen Jobs durch die KI jagen
    if jobs_to_analyze:
        print(f"🧠 Starte KI-Abgleich für {len(jobs_to_analyze)} neue Jobs (Limit: {ANALYZE_LIMIT})...")
        for job in jobs_to_analyze:
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
                    "employee_count": job.get("companyEmployeeCount", "N/A"),
                    "contract_type": job.get("contractType", "N/A"),
                    "exp_level": job.get("experienceLevel", "N/A"),
                    "salary_info": job.get("salary", "N/A"),
                    "post_info": job.get("postDate", "N/A"),
                    "url": job["url"]
                })
                result["analyzed_at"] = current_date.strftime("%Y-%m-%d")
                new_analyzed_jobs.append(result)
                
            except Exception as e:
                print(f"⚠️ Fehler bei der Analyse von {job['title']}: {e}")
    else:
        print("☕ Keine neuen Jobs zur Analyse ausstehend.")
        
    # J. Bereinigten Verlauf mit neuen Analysen zusammenführen
    updated_history = cleaned_history + new_analyzed_jobs
    
    # K. Verlauf abspeichern
    save_json_file(updated_history, HISTORY_FILE)
    
    # L. E-Mail versenden (Zusammenführung alte + neue Jobs)
    send_email_report(
        new_jobs=new_analyzed_jobs,
        old_jobs=cleaned_history,
        linkedin_total=linkedin_total,
        scraped_total=scraped_total,
        test_mode=test_mode
    )

if __name__ == "__main__":
    asyncio.run(main())
