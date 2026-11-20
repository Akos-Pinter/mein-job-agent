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

# ==========================================
# KONFIGURATION & GLOBALE VARIABLEN
# ==========================================
HISTORY_FILE = "history_jobs.json"

# Pydantic Schema für strukturierte Job-Bewertung
class JobMatch(BaseModel):
    score: int = Field(description="Match-Score von 0 bis 100")
    reasoning: str = Field(description="Detaillierte Begründung für den Score in Deutsch")
    required_qualifications: List[str] = Field(description="Liste von zwingend oder vorteilhaft geforderten Qualifikationen, Skills, Frameworks oder Tools")
    fitting_qualifications: List[str] = Field(description="Liste der Qualifikationen des Bewerbers, die gut zur Stelle passen")

# ==========================================
# HILFSFUNKTIONEN
# ==========================================

def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"📄 Extrahiere Text aus PDF: {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen der PDF-Datei {pdf_path}: {e}")
        return ""

def load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Historie: {e}. Starte neu.")
            return []
    return []

def save_history(history: List[Dict[str, Any]]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"💾 Historie erfolgreich aktualisiert ({len(history)} Einträge insgesamt).")
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern der Historie: {e}")

# ==========================================
# SCRAPER - APIFY INTEGRATION
# ==========================================

def run_apify_scraper(api_key: str, actor_id: str, run_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    print(f"🕵️ Starte Apify Scraper Actor: {actor_id}...")
    client = ApifyClient(api_key)
    try:
        # Starten des Actors und Warten auf das Ende der Ausführung
        run = client.actor(actor_id).call(run_input=run_input)
        
        # Abrufen der Ergebnisse aus dem Dataset der Ausführung
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        print(f"✅ Scraper beendet. {len(dataset_items)} Ergebnisse abgerufen.")
        return dataset_items
    except Exception as e:
        print(f"⚠️ Fehler beim Scraper-Lauf: {e}")
        return []

# ==========================================
# EXTRAKTION & NORMALISIERUNG DER SCRAPING DATA
# ==========================================

def normalize_jobs(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in raw_results:
        # Extraktion wichtiger Felder unabhängig vom genauen Scraper-Format
        job_title = item.get("title") or item.get("jobTitle") or item.get("positionName")
        company = item.get("company") or item.get("companyName") or item.get("employerName")
        location = item.get("location") or item.get("locationName") or "Remote / Nicht angegeben"
        url = item.get("url") or item.get("jobUrl") or item.get("link")
        description = item.get("description") or item.get("jobDescription") or item.get("text") or ""
        scraped_at = datetime.now().isoformat()
        
        if job_title and company:
            normalized.append({
                "job_title": job_title.strip(),
                "company": company.strip(),
                "location": location.strip(),
                "url": url.strip() if url else "",
                "description": description.strip(),
                "scraped_at": scraped_at
            })
    return normalized

def filter_unseen_jobs(scraped_jobs: List[Dict[str, Any]], history_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Wir identifizieren Duplikate anhand der Kombination von Job-Titel und Firma
    seen_keys = { (job.get("job_title", "").lower().strip(), job.get("company", "").lower().strip()) for job in history_jobs }
    
    unseen = []
    for job in scraped_jobs:
        key = (job.get("job_title", "").lower().strip(), job.get("company", "").lower().strip())
        if key not in seen_keys:
            unseen.append(job)
    
    print(f"🔍 Filtern abgeschlossen: {len(unseen)} von {len(scraped_jobs)} Scraper-Ergebnissen sind neu.")
    return unseen

# ==========================================
# Lücken-Analyse & Weiterbildungsempfehlungen
# ==========================================
def generate_qualification_recommendations(profile: str, jobs: List[Dict[str, Any]]) -> str:
    print("🧠 Analysiere Qualifikationslücken für Jobs mit einem Match-Score > 66%...")
    high_score_jobs = [job for job in jobs if job.get("score", 0) > 66]
    
    if not high_score_jobs:
        return "Keine Jobs mit einem Match-Score von über 66% für eine Lückenanalyse gefunden."
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY fehlt. Qualifikationsanalyse kann nicht durchgeführt werden."
        
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        groq_api_key=api_key
    )
    
    # Vorbereitung der Job-Details (Titel, Firma, benötigte Qualifikationen)
    jobs_summary_list = []
    for i, job in enumerate(high_score_jobs, 1):
        quals = ", ".join(job.get("required_qualifications", []))
        jobs_summary_list.append(
            f"Job {i}: {job.get('job_title')} bei {job.get('company')} (Match-Score: {job.get('score')}%)\n"
            f"  - Geforderte Qualifikationen: {quals}"
        )
    jobs_summary_str = "\n".join(jobs_summary_list)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein erfahrener Karriere-Coach und HR-Stratege, der sich auf den Quereinstieg von "
                   "Ingenieuren in Data-Science-, KI- und Software-bezogene Stellen spezialisiert hat. "
                   "Deine Aufgabe ist es, die geforderten Qualifikationen von vielversprechenden Stellen (Match-Score > 66%) "
                   "mit dem Profil des Kandidaten abzugleichen. Identifiziere konkrete Qualifikationslücken und "
                   "empfehle präzise, umsetzbare Maßnahmen (z.B. Online-Kurse, Zertifizierungen, Portfolio-Projekte), "
                   "um diese Lücken schnellstmöglich zu schließen."),
        ("user", "=== PROFIL DES KANDIDATEN ===\n{profile}\n\n"
                 "=== RELEVANTE STELLEN (SCORE > 66%) ===\n{jobs_summary}\n\n"
                 "Bitte erstelle einen detaillierten, professionellen Bericht auf Deutsch mit:\n"
                 "1. Zusammenfassung der häufigsten Qualifikationslücken (Gaps) basierend auf dem required_qualifications Feld\n"
                 "2. Konkreten Empfehlungen für Weiterbildungen, Kurse oder Zertifizierungen (z.B. Coursera, AWS, Python-Zertifikate, etc.)\n"
                 "3. Praktischen Maßnahmen und Projekten zur Schließung der Lücken, um das Profil attraktiver zu gestalten.")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "profile": profile,
            "jobs_summary": jobs_summary_str
        })
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"⚠️ Fehler bei der Generierung der Empfehlungen: {e}")
        return f"Fehler bei der Qualifikationsanalyse: {e}"

# ==========================================
# LangChain-Chain definieren
# ==========================================

def create_matching_chain() -> Any:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY ist nicht in den Umgebungsvariablen definiert.")
        sys.exit(1)
        
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
        groq_api_key=api_key
    )
    
    parser = JsonOutputParser(pydantic_object=JobMatch)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein präzises HR-Matching-Modell. Vergleiche das Bewerberprofil mit der Jobbeschreibung.\n"
                   "Führe eine objektive Bewertung durch. Erstelle ein Match-Ergebnis im vorgegebenen JSON-Format.\n\n"
                   "{format_instructions}"),
        ("user", "=== BEWERBERPROFIL ===\n{profile}\n\n"
                 "=== JOBBESCHREIBUNG ===\n{description}\n\n"
                 "Bitte analysiere diese Stelle und gib das JSON-Objekt zurück.")
    ])
    
    # Injizieren der Pydantic Parser Instruktionen in das System Prompt
    prompt_with_instructions = prompt.partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt_with_instructions | llm | parser
    return chain

# ==========================================
# E-MAIL BERICHTS-GENERATOR (SMTP)
# ==========================================

def send_email_report(new_jobs: List[Dict[str, Any]], old_jobs: List[Dict[str, Any]], scraped_total: int, api_stats: Dict[str, int], test_mode: bool = False, recommendations: str = None):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("REPORT_RECIPIENT")
    
    if not test_mode and not all([smtp_server, smtp_user, smtp_password, recipient]):
        print("⚠️ E-Mail Konfiguration unvollständig. Überspringe E-Mail-Versand.")
        return
        
    print("📧 Generiere E-Mail-Bericht...")
    
    # Sortieren der neuen Jobs nach Score (höchster zuerst)
    new_jobs_sorted = sorted(new_jobs, key=lambda x: x.get("score", 0), reverse=True)
    
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1f2937; background-color: #f3f4f6; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 2px solid #e5e7eb; padding-bottom: 20px; margin-bottom: 25px; }}
            .header h1 {{ color: #1e3a8a; margin: 0; font-size: 24px; }}
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 25px; }}
            .stat-card {{ background-color: #f9fafb; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #e5e7eb; }}
            .stat-card .num {{ font-size: 20px; font-weight: bold; color: #10b981; }}
            .stat-card .label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; }}
            .job-card {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 20px; margin-bottom: 15px; background-color: #ffffff; }}
            .badge-high {{ background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .badge-medium {{ background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .badge-low {{ background-color: #f3f4f6; color: #374151; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .section-title {{ color: #1f2937; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 30px; }}
            .quals-list {{ margin: 10px 0 0 0; padding-left: 20px; font-size: 12px; color: #4b5563; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 JobAgent Matching-Bericht</h1>
                <p style="color: #6b7280; margin: 5px 0 0 0;">{datetime.now().strftime('%d.%m.%Y - %H:%M')} Uhr</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="num" style="color: #2563eb;">{scraped_total}</div>
                    <div class="label">Gescrapt</div>
                </div>
                <div class="stat-card">
                    <div class="num" style="color: #10b981;">{len(new_jobs)}</div>
                    <div class="label">Neu & Bewertet</div>
                </div>
                <div class="stat-card">
                    <div class="num" style="color: #4b5563;">{api_stats.get('successful_calls', 0)} / {api_stats.get('total_calls', 0)}</div>
                    <div class="label">AI API-Lauf</div>
                </div>
            </div>
    """

    # Empfehlungen einfügen falls vorhanden (Am Anfang des Berichts)
    if recommendations:
        # Markdown to simple HTML linebreaks / bullet points conversion
        rec_html = recommendations.replace('\n', '<br>')
        html_template += f'''
            <div class="job-card" style="border-left: 6px solid #10b981; background-color: #f0fdf4; margin-bottom: 25px;">
                <h3 style="color: #065f46; margin-top: 0; font-size: 18px; display: flex; align-items: center; gap: 8px;">💡 Empfehlungen zur Weiterbildung & Qualifikation</h3>
                <p style="font-size: 13px; color: #065f46; margin-bottom: 12px; font-weight: bold;">
                    Basierend auf deinen Top-Matches (> 66%) wurden folgende Qualifikationslücken (Gaps) identifiziert:
                </p>
                <div style="font-size: 13px; color: #374151; line-height: 1.6;">
                    {rec_html}
                </div>
            </div>
        '''

    html_template += """
            <h2 class="section-title">Neue Stellenangebote ({num_new})</h2>
    """.replace("{num_new}", str(len(new_jobs)))
    
    if not new_jobs:
        html_template += "<p style='color: #6b7280; font-style: italic;'>Keine neuen Stellen in diesem Durchlauf gefunden.</p>"
    else:
        for job in new_jobs_sorted:
            score = job.get("score", 0)
            if score >= 75:
                badge_class = "badge-high"
            elif score >= 50:
                badge_class = "badge-medium"
            else:
                badge_class = "badge-low"
                
            quals_fitting = "".join([f"<li>{q}</li>" for q in job.get("fitting_qualifications", [])])
            quals_req = "".join([f"<li>{q}</li>" for q in job.get("required_qualifications", [])])
            
            html_template += f"""
            <div class="job-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                    <div>
                        <h3 style="margin: 0; color: #1f2937; font-size: 16px;">{job.get('job_title')}</h3>
                        <p style="margin: 2px 0 0 0; color: #4b5563; font-size: 13px;">{job.get('company')} - {job.get('location')}</p>
                    </div>
                    <span class="{badge_class}">{score}% Match</span>
                </div>
                
                <p style="font-size: 13px; color: #374151; line-height: 1.5; margin: 10px 0 15px 0;">
                    <strong>KI-Einschätzung:</strong> {job.get('reasoning')}
                </p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <strong style="font-size: 12px; color: #065f46;">Passt gut:</strong>
                        <ul class="quals-list" style="color: #065f46; list-style-type: '✓ ';">
                            {quals_fitting or '<li>Keine expliziten Matches identifiziert</li>'}
                        </ul>
                    </div>
                    <div>
                        <strong style="font-size: 12px; color: #92400e;">Gefordert / Offene Punkte:</strong>
                        <ul class="quals-list" style="color: #92400e; list-style-type: '• ';">
                            {quals_req or '<li>Keine Angabe</li>'}
                        </ul>
                    </div>
                </div>
                
                {f'<div style="margin-top: 15px;"><a href="{job.get("url")}" target="_blank" style="background-color: #2563eb; color: #ffffff; padding: 6px 12px; font-size: 12px; font-weight: bold; text-decoration: none; border-radius: 4px; display: inline-block;">Stelle anzeigen ↗</a></div>' if job.get("url") else ""}
            </div>
            """
            
    # Wir fügen auch eine kompakte Übersicht alter Übereinstimmungen an
    html_template += """
            <h2 class="section-title">Bisherige Top-Matches (Historie)</h2>
    """
    
    top_old_jobs = [job for job in old_jobs if job.get("score", 0) >= 60]
    top_old_sorted = sorted(top_old_jobs, key=lambda x: x.get("score", 0), reverse=True)[:5]
    
    if not top_old_sorted:
        html_template += "<p style='color: #6b7280; font-style: italic;'>Keine bisherigen Top-Matches in der Historie.</p>"
    else:
        for job in top_old_sorted:
            html_template += f"""
            <div style="padding: 10px 0; border-bottom: 1px dashed #e5e7eb; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 13px; font-weight: 500; color: #374151;">{job.get('job_title')}</span>
                    <span style="font-size: 11px; color: #6b7280; margin-left: 5px;">bei {job.get('company')}</span>
                </div>
                <span style="font-size: 12px; font-weight: bold; color: #2563eb;">{job.get('score')}%</span>
            </div>
            """
            
    html_template += """
            <div style="margin-top: 30px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 15px; font-size: 11px; color: #9ca3af;">
                JobAgent läuft vollautomatisch. Berichte werden nur bei neuen Funden generiert.
            </div>
        </div>
    </body>
    </html>
    """
    
    if test_mode:
        print("🧪 [TESTMODE] E-Mail Vorschau erstellt. HTML-Länge:", len(html_template))
        with open("email_preview_test.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        print("💾 [TESTMODE] HTML-Vorschau in 'email_preview_test.html' gespeichert.")
        return
        
    # E-Mail versenden via SMTP
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"JobAgent Report: {len(new_jobs)} neue Matches für dich!"
        msg["From"] = smtp_user
        msg["To"] = recipient
        
        part_html = MIMEText(html_template, "html", "utf-8")
        msg.attach(part_html)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
            
        print("✉️ E-Mail-Bericht wurde erfolgreich versendet!")
    except Exception as e:
        print(f"⚠️ E-Mail-Versand fehlgeschlagen: {e}")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main():
    print("🚀 Starte Job-Search Agent mit automatischem Qualifikationslücken-Report...")
    
    # 0. Testmodus checken
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    if test_mode:
        print("🧪 Agent läuft im TESTMODUS (kein SMTP-Versand, HTML wird lokal abgelegt).")
        
    # A. Bewerberprofil laden (Erwartet PDF im selben Ordner)
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ Kein Bewerberprofil (.pdf) im aktuellen Verzeichnis gefunden.")
        sys.exit(1)
        
    user_profile = extract_text_from_pdf(pdf_files[0])
    if not user_profile:
        print("❌ Das Bewerberprofil konnte nicht ausgelesen werden.")
        sys.exit(1)
        
    # B. Historie bereits bewerteter Jobs laden
    cleaned_history = load_history()
    
    # C. Apify Scraper Parameter laden
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        print("❌ APIFY_API_TOKEN fehlt in den Umgebungsvariablen.")
        sys.exit(1)
        
    # Standard Scraper: Indeed Scraper (actor "apify/indeed-scraper")
    actor_id = "apify/indeed-scraper"
    run_input = {
        "position": "Data Scientist OR Machine Learning Engineer",
        "location": "München",
        "maxItems": 15,
        "startUrls": []
    }
    
    # D. Scraper starten
    raw_jobs = run_apify_scraper(apify_token, actor_id, run_input)
    scraped_total = len(raw_jobs)
    
    # E. Jobs normalisieren
    normalized_jobs = normalize_jobs(raw_jobs)
    
    # F. Nur ungesehene Jobs filtern
    new_unseen_jobs = filter_unseen_jobs(normalized_jobs, cleaned_history)
    
    # G. Matching-Chain initialisieren
    matching_chain = create_matching_chain()
    
    # H. Jobs durch das LLM bewerten
    new_analyzed_jobs = []
    api_stats = {"total_calls": 0, "successful_calls": 0}
    
    for i, job in enumerate(new_unseen_jobs, 1):
        print(f"🤖 Analysiere Job {i}/{len(new_unseen_jobs)}: {job.get('job_title')} bei {job.get('company')}...")
        api_stats["total_calls"] += 1
        
        try:
            # LLM Chain synchron ausführen
            result = matching_chain.invoke({
                "profile": user_profile,
                "description": job.get("description", "")
            })
            
            # Matchergebnisse in das Job-Dictionary mergen
            job["score"] = result.get("score", 0)
            job["reasoning"] = result.get("reasoning", "")
            job["required_qualifications"] = result.get("required_qualifications", [])
            job["fitting_qualifications"] = result.get("fitting_qualifications", [])
            
            new_analyzed_jobs.append(job)
            api_stats["successful_calls"] += 1
            print(f"   └─ Score: {job['score']}%")
            
        except Exception as e:
            print(f"   ⚠️ Fehler bei der LLM-Bewertung: {e}")
            
    # I. Wenn keine neuen Jobs gefunden/analysiert wurden und wir nicht im Testmodus sind, beenden
    if not new_analyzed_jobs and not test_mode:
        print("😴 Keine neuen Jobs zu melden. Beende Programmlauf.")
        return
        
    # K. Historie zusammenführen und speichern
    updated_history = cleaned_history + new_analyzed_jobs
    save_history(updated_history)
    
    # L. Qualifikationslücken & Weiterbildungsempfehlungen generieren (für Jobs > 66% Score)
    recommendations = generate_qualification_recommendations(user_profile, updated_history)
    print("\n=== QUALIFIKATIONSEMPFEHLUNGEN ===\n", recommendations, "\n==================================\n")
    
    # M. E-Mail versenden (Zusammenführung alte + neue Jobs & Empfehlungen)
    send_email_report(
        new_jobs=new_analyzed_jobs,
        old_jobs=cleaned_history,
        scraped_total=scraped_total,
        api_stats=api_stats,
        test_mode=test_mode,
        recommendations=recommendations
    )

if __name__ == "__main__":
    main()
