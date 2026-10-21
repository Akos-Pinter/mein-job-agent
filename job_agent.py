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
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt.")
    reason: str = Field(description="Begründung für die Bewertung (max. 40 Wörter).")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    company_employee_count: str = Field(description="Die Mitarbeiteranzahl der Firma als formatiertes Textfeld (z.B. '240.000 Mitarbeiter' oder 'N/A' falls unbekannt).")
    location: str = Field(description="Der Arbeitsort (z.B. Stadt, Hybrid, Remote) des Jobs.")
    route_distance: str = Field(description="Die geschätzte Entfernung vom Wohnort des Kandidats zum Arbeitsort in Kilometern (z.B. '25 km').")
    route_duration: str = Field(description="Die geschätzte Reisezeit mit dem Auto bei normalem Verkehr (z.B. '30 Min. Autofahrt').")
    route_transit: str = Field(description="Die geschätzte Reisezeit mit Bahn/ÖPNV vom Wohnort des Kandidats zum Arbeitsort (z.B. '45 Min. mit ÖPNV').")
    contract_type: str = Field(description="Der Vertragstyp der Stelle (z.B. 'Full-time', 'Part-time', 'Contract' oder 'N/A').")
    experience_level: str = Field(description="Die geforderte Erfahrungsstufe der Stelle (z.B. 'Mid-Senior level', 'Entry level', 'Associate' oder 'N/A').")
    salary_source: str = Field(description="Das geschätzte oder verifizierte Gehalt / Gehaltsspanne der Stelle (z.B. '80.000 € - 95.000 €' oder 'N/A').")
    post_date: str = Field(description="Das genaue Veröffentlichungsdatum der Stelle (z.B. 'Vor 3 Tagen' oder das exakte Datum, falls vorhanden).")
    job_summary: str = Field(description="Eine kurze, prägnante Zusammenfassung der Stelle und Aufgaben (max. 5 Sätze).")
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


# 4.1 Job-Abfrage über RapidAPI JSearch
async def fetch_jobs_via_jsearch(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 25) -> Dict[str, Any]:
    print("Starte Job Abfrage über RapidAPI JSearch...")
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
         print("⚠️ Warnung: RAPIDAPI_KEY fehlt.")
         return {"jobs": [], "scraped_total": 0}
        
    url = "https://jsearch.p.rapidapi.com/search-v2"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
	
    titles_query = " OR ".join(f'"{title}"' for title in search_titles)
    params = {
        "query": f"{titles_query} in {location}",
        "location": location,
        "num_pages": str(min(max(1, -(-fetch_limit // 10)), 20)),
        "country": "de",
        "work_from_home": True,
        "date_posted": "3days", # Options: today, 3days, week, month, all
        "employment_types": "FULLTIME",
        "job_requirements": "more_than_3_years_experience",
        "radius": str(distance_km)
    }
	
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        response_data = response.json()
        
        raw_items = response_data.get("data", []).get("jobs", [])
        print(f"📥 JSearch lieferte {len(raw_items)} Treffer")
		
        jobs = []
        for item in raw_items:
            min_sal = item.get("job_min_salary")
            max_sal = item.get("job_max_salary")
            currency = item.get("job_salary_currency", "EUR")
            salary_str = f"{min_sal:,} - {max_sal:,} {currency}".replace(",", ".") if min_sal and max_sal else item.get("job_salary", "N/A")
            
            jobs.append({
                "title": item.get("job_title", "Kein Titel"),
                "company": item.get("employer_name", "Unbekannte Firma"),
                "description": item.get("job_description", "")[:5000],
                "location": item.get('job_city', ''),
                "url": item.get("job_apply_link", item.get("job_google_link", "N/A")),
                "salary": salary_str,
                "postDate": item.get("job_posted_at_datetime_utc", "N/A")[:10],
                "contractType": item.get("job_employment_type", "N/A"),
                "experienceLevel": "N/A",
                "companyEmployeeCount": "N/A",
                "api": "JSearch"
            })
            
        return {
            "jobs": jobs,
            "scraped_total": len(jobs)
        }
    except Exception as e:
        print(f"⚠️ Warnung: JSearch Suche fehlgeschlagen: {e}")
        return {"jobs": [], "scraped_total": 0}


# 4.2 Job-Abfrage über Python Apify-Client (Indeed)
async def fetch_jobs_via_apify_indeed(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 50) -> Dict[str, Any]:  
    print("Starte Job Abfrage über Apify Indeed Search...")
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
         print("⚠️ Warnung: APIFY_API_TOKEN fehlt.")
         return {"jobs": [], "scraped_total": 0}
        
    client = ApifyClient(token)
    try:
        titles_query = " OR ".join(f'"{title}"' for title in search_titles)
        run_input = {
            "keyword": titles_query,
            "location": location, 
            "country": "DE",
            "maxItems": fetch_limit,
            "radius": str(distance_km),
            "radiusUnit": "km", 
            "fromDays": "14", # Options: 1, 3, 7, 14
            "jobType": "fulltime",
            "remote": "hybrid",
            "searchMode": "rich"   
        }
        
        run = client.actor("kaix/indeed-scraper").call(run_input=run_input)
        raw_items = client.dataset(run["defaultDatasetId"]).list_items().items
        print(f"📥 Apify Indeed lieferte {len(raw_items)} Treffer")
  
        jobs = []
        for item in raw_items:
            salary_data = item.get("salary") or {}
            min_sal = salary_data.get("min")
            max_sal = salary_data.get("max")
            currency = salary_data.get("currency", "EUR")
            salary_str = f"{min_sal:,} - {max_sal:,} {currency}".replace(",", ".") if min_sal and max_sal else salary_data.get("text", "N/A")
            
            jobs.append({
                "title": item.get("title", {}).get("text", "N/A") if isinstance(item.get("title"), dict) else item.get("title", "N/A"),
                "company": item.get("company", {}).get("name", "N/A") if isinstance(item.get("company"), dict) else item.get("company", "N/A"),
                "description": item.get("description", {}).get("text", "N/A")[:5000] if isinstance(item.get("description"), dict) else item.get("description", "N/A")[:5000],
                "location": item.get("location", {}).get("city", "N/A") if isinstance(item.get("location"), dict) else item.get("location", "N/A"),
                "url": item.get("urls", {}).get("indeed", "N/A") if isinstance(item.get("urls"), dict) else "N/A",
                "salary": salary_str,
                "postDate": item.get("dates", {}).get("posted", "N/A") if isinstance(item.get("dates"), dict) else "N/A",
                "contractType": item.get("classification", {}).get("jobType", "N/A") if isinstance(item.get("classification"), dict) else "N/A",
                "experienceLevel": "N/A",
                "companyEmployeeCount": "N/A",
                "api": "Apify Indeed"
            })
        
        return {
            "jobs": jobs,
            "scraped_total": len(jobs)
        }
    except Exception as e:
        print(f"⚠️ Warnung: Apify Indeed Suche fehlgeschlagen: {e}")
        return {"jobs": [], "scraped_total": 0}


# 4.2 Job-Abfrage über Python Apify-Client (LinkedIn)
async def fetch_jobs_via_apify_linkedin(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 50) -> Dict[str, Any]: 
    print("Starte Job Abfrage über Apify LinkedIn Search...")
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
         print("⚠️ Warnung: APIFY_API_TOKEN fehlt.")
         return {"jobs": [], "scraped_total": 0}
        
    client = ApifyClient(token)
    try:
        titles_query = " OR ".join(f'"{title}"' for title in search_titles)
        run_input = {
            "keywords": titles_query,
            "location": location,       
            "datePosted": "past24Hours",
            "limitPerSource": fetch_limit,
			"urlParam": [
				{"key": "distance", "value": str(int(distance_km * 0.62))},      # Radius in Meilen
				{"key": "f_JT", "value": "F,P,T"},       # Optionen: Vollzeit (F) / Teilzeit (P) / Freie Mitarbeit (C) / Befristete Anstellung (T), Praktikum (I), Ehrenamtlich(V)
				{"key": "f_E", "value": "3,4"},          # Optionen: Ohne Berufserfahrung / Berufseinstieg / Junior / Professional / Senior (1-5)
				{"key": "f_WT", "value": "3"},         # Optionen: Vor Ort / Nur Remote / Hybrid (1-3)
				]
        }
        
        run = client.actor("curious_coder/linkedin-jobs-scraper").call(run_input=run_input)
        raw_items = client.dataset(run["defaultDatasetId"]).list_items().items
  
        jobs = []
        for item in raw_items:
            jobs.append({
                "title": item.get("title", "N/A"),
                "company": item.get("companyName", "N/A"),
                "description": item.get("descriptionText", "")[:5000],
                "location": "N/A" if item.get("companyAddress") is None else item.get("companyAddress").get("addressLocality", "N/A"),
                "url": item.get("link", "https://linkedin.com"),
                "salary": item.get("salary", "N/A"),
                "postDate": item.get("postedAt", "N/A"),
                "contractType": item.get("employmentType", "N/A"),
                "experienceLevel": item.get("seniorityLevel", "N/A"),
                "companyEmployeeCount": item.get("companyEmployeesCount", "N/A"),
                "api": "Apify LinkedIn"
            })
        print(f"📥 Apify LinkedIn lieferte {len(jobs)} Treffer")
		
        return {
            "jobs": jobs,
            "scraped_total": len(jobs)
        }
    except Exception as e:
        print(f"⚠️ Warnung: Apify LinkedIn Suche fehlgeschlagen: {e}")
        return {"jobs": [], "scraped_total": 0}


# 4.3 Job-Abfrage über die offizielle Jobbörse (Bundesagentur für Arbeit)
async def fetch_jobs_via_arbeitsagentur(search_titles: List[str], location: str, distance_km: int = 30, fetch_limit: int = 25) -> Dict[str, Any]:
    print("Starte Job Abfrage über Bundesagentur für Arbeit...")
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
            "arbeitszeit": "vz",
			"veroeffentlichtseit": "14",
            "befristung": "2",
            "zeitarbeit": "false",
            "pav": "false",
            "angebotsart": "1",
            "page": "1",
            "size": str(fetch_limit)
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=15)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            search_results = response.json()
            
            raw_items = search_results.get("ergebnisliste", [])
            print(f"📥 BfA lieferte {len(raw_items)} Treffer")
            
            for item in raw_items:
                ref_nr = item.get("referenznummer", "")	
                if any(job["url"].endswith(ref_nr) for job in jobs):
                    continue
                
                base64_ref = base64.b64encode(ref_nr.encode("utf-8")).decode("utf-8")
                detail_url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{base64_ref}"
                
                try:
                    detail_resp = requests.get(detail_url, headers=headers, timeout=6)
                    if detail_resp.status_code == 200:
                        detail_data = detail_resp.json()
                        description = detail_data.get("stellenangebotsBeschreibung", "")
                except Exception:
                    description = "N/A"
                
                lokationen = item.get("stellenlokationen", [])
                ort_name = lokationen[0].get("adresse", {}).get("ort", "N/A") if lokationen else "N/A"
                verguetung = item.get("verguetungsangabe")
                salary_val = "N/A" if verguetung == "KEINE_ANGABEN" else verguetung
                
                if item.get("arbeitszeitVollzeit"):
                    arbeitszeit = "Vollzeit"
                elif any([item.get("arbeitszeitTeilzeitFlexibel"), item.get("arbeitszeitTeilzeitVormittag"), 
                          item.get("arbeitszeitTeilzeitNachmittag"), item.get("arbeitszeitTeilzeitAbend")]):
                    arbeitszeit = "Teilzeit"
                else:
                    arbeitszeit = "N/A"
				
                jobs.append({
				    "title": item.get("stellenangebotsTitel", "N/A"),
				    "company": item.get("firma", "N/A"),
				    "description": description[:5000],
				    "location": ort_name,
				    "url": item.get("externeUrl") or f"https://www.arbeitsagentur.de/jobsuche/job/{ref_nr}",
				    "salary": salary_val,
				    "postDate": item.get("datumErsteVeroeffentlichung", "N/A"),
				    "contractType": arbeitszeit,
				    "experienceLevel": "N/A",
				    "companyEmployeeCount": "N/A",
				    "api": "BfA"
				})
        except Exception as e:
            print(f"⚠️ Warnung: Suche für {title} über Bundesagentur für Arbeit failed: {e}")
            continue
    
    return {
        "jobs": jobs,
        "scraped_total": len(jobs)
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
        ("system", "Du bist ein präziser HR-Analyst und Karriere-Coach, der sich auf den Quereinstieg von "
                   "Ingenieuren in Data-Science-, KI- und Software-bezogene Stellen spezialisiert hat. "
                   "Gleiche den Job mit dem Lebenslauf-Profil des Kandidaten ab. "
                   "Berücksichtige hierbei die ingenieurwissenschaftlichen Stärken (z.B. analytisches Denken, "
                   "Problemlösungskompetenz, technische Grundlagen) als starken Pluspunkt für den Quereinstieg.\n\n"
                   "WICHTIGE BEWERTUNGSREGELN:\n"
                   "1. Schätze basierend auf deinem geographischen Wissen die Entfernung von {home_location} zum "
                   "Arbeitsort sowie die Pendelzeiten (Auto und ÖPNV) realistisch ein. Liegt die Pendelzeit sowohl "
				   "mit dem Auto als auch mit ÖPNV über eine Stunde, dann setze den Match-Score zwingend und ausnahmslos auf 0%.\n"
                   "2. Wenn es sich bei der Stelle um ein 'Internship', 'Praktikum', 'Duales Studium', 'Werkstudentenstelle' "
                   "oder eine reine 'Abschlussarbeit' handelt, setze den Match-Score zwingend und ausnahmslos auf 0%.\n{format_instructions}"),
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
def send_email_report(new_jobs: List[Dict[str, Any]], old_jobs: List[Dict[str, Any]], scraped_total: int, api_stats: Dict[str, int], test_mode: bool = False):
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

    # Formatierung des API-Aufteilungstextes (z.B. "JSearch: 12 | BfA: 5")
    stats_formatted_list = [f"{api_name}: {count}" for api_name, count in api_stats.items() if count > 0]
    api_stats_str = " | ".join(stats_formatted_list) if stats_formatted_list else "Keine"

    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            h2 {{ color: #1e3a8a; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
            .job-card {{ background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin-bottom: 20px; position: relative; }}
            .new-badge {{ position: absolute; top: 15px; left: 15px; background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; text-transform: uppercase; }}
            .api-badge {{ display: inline-block; background-color: #3b82f6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-left: 8px; vertical-align: middle; }}
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
                • Davon für diesen Bericht live geladen: <b>{scraped_total}</b> ({api_stats_str})<br>
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

        # Einlesen der genutzten API für die jeweilige Job-Karte
        job_api_source = job.get("api", "Unbekannt")

        html_template += f"""
            <div class="job-card">
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; text-align: left; padding-right: 15px;">
                        {new_badge_html}
                        <strong style="font-size: 17px; color: #1e3a8a; vertical-align: middle;">
                            {job['job_title']}
                        </strong>
                        <span class="api-badge">{job_api_source}</span> <!-- NEU: API-Herkunfts-Badge -->
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

    # D. Aktuelle Live-Stellen abrufen
    raw_jobs = []

    # Dictionary zum Zählen der Treffer pro API (z.B. {"JSearch": 12, "BfA": 5, ...})
    api_stats = {
        "JSearch": 0,
        "Apify Indeed": 0,
        "Apify LinkedIn": 0,
        "BfA": 0
    }

    for suche in [fetch_jobs_via_jsearch, fetch_jobs_via_apify_indeed, fetch_jobs_via_apify_linkedin, fetch_jobs_via_arbeitsagentur]:
        try:
            scraper_result = await suche(
                search_titles=WUNSCH_TITEL,
                location=MY_HOME_LOCATION,
                distance_km=30,
                fetch_limit=FETCH_LIMIT
            )

            if scraper_result and "jobs" in scraper_result:
                jobs_list = scraper_result["jobs"]
                raw_jobs.extend(jobs_list)

                # Bestimmen, von welcher API der Scraper kam, und Trefferanzahl hochzählen
                scraped_count = scraper_result.get("scraped_total", 0)
                if jobs_list:
                    # Wir prüfen die API-Identifikation des ersten Elements in der Liste
                    api_name = jobs_list[0].get("api", "Unbekannt")
                    if api_name in api_stats:
                        api_stats[api_name] = scraped_count
                    else:
                        api_stats[api_name] = scraped_count

        except Exception as e:
            print(f"⚠️ Warnung bei Scraper-Ausführung: {e}")

    scraped_total = sum(api_stats.values())

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
                result["api"] = job.get("api", "Unbekannt")
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
        scraped_total=scraped_total,
        api_stats=api_stats,
        test_mode=test_mode
    )

if __name__ == "__main__":
    asyncio.run(main())
