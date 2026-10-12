import os
import asyncio
import json
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Pydantic-Struktur für strukturiertes JSON-Output von Llama 3.1
class JobMatch(BaseModel):
    score: int = Field(description="Ein Score von 0 bis 100, wie gut der Job zum Profil passt.")
    reason: str = Field(description="Begründung für die Bewertung (max. 20 Wörter).")
    job_title: str = Field(description="Der Titel des analysierten Jobs.")
    company: str = Field(description="Die Firma des Jobs.")
    recommended_action: str = Field(description="Empfehlung: 'Bewerben', 'Vielleicht' oder 'Ignorieren'.")

# 2. Ihr persönliches Profil (Hier einfach Ihre Daten anpassen)
USER_PROFILE = """
- Rolle: Python Backend Entwickler
- Programmiersprachen: Python (Experte), SQL (Sehr gut), FastAPI, Docker
- Berufserfahrung: 3 Jahre
- Arbeitsweise: Hybrid / Remote (Sitz in Berlin)
- Sprache: Deutsch (Muttersprache), Englisch (Fließend)
"""

# 3. Job-Abfrage über MCP-Anbindung (mit Fallback)
async def fetch_jobs():
    print("🔎 Rufe aktuelle Jobangebote ab...")
    
    # Cloud-Alternative für MCP: Da GitHub Actions isoliert läuft, ist der einfachste 
    # Weg, standardisierte APIs (wie SerpAPI oder freie RSS-Feeds) via MCP/Request zu laden.
    try:
        # HIER KÖNNEN SIE IHREN MCP-SERVER ANSTEUERN
        # Falls Sie z.B. einen gehosteten MCP-Server auf Apify/Hugging Face nutzen:
        # server_params = StdioServerParameters(command="npx", args=["-y", "@apify/cli", "mcp", ...])
        raise NotImplementedError("Nutze stabilen Fallback-Feed für die Cloud")
        
    except Exception:
        # Robuster, kostenloser Cloud-Fallback: Job-Feed parsen
        # Wir nutzen ein freies API-Beispiel für Tech-Jobs in Deutschland
        fallback_url = "https://www.arbeitnow.com/api/job-board-api"
        response = requests.get(fallback_url)
        if response.status_code == 200:
            data = response.json()
            jobs = []
            for item in data.get("data", [])[:5]: # Die neuesten 5 Jobs analysieren
                jobs.append({
                    "title": item.get("title"),
                    "company": item.get("company_name"),
                    "description": item.get("description", "")[:1000], # Text kürzen für Token-Limit
                    "location": item.get("location")
                })
            return jobs
        return []

# 4. LangChain-Chain für den Abgleich definieren
def create_matching_chain():
    # Wir holen uns den API Key aus den GitHub Secrets / Umgebungsvariablen
    api_key = os.getenv("GROQ_API_KEY")
    
    # Llama 3.1 70B über Groq initialisieren (Kostenlos und extrem schlau)
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
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

# 5. Telegram-Benachrichtigung (Optional, aber genial für die Cloud)
def send_telegram_message(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
        print("📨 Telegram-Benachrichtigung gesendet!")

# 6. Hauptprogramm
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

    # Output generieren
    output_lines = ["🏆 *DEINE TÄGLICHEN TOP-JOBS* 🏆\n"]
    for match in analyzed_jobs:
        score_icon = "🟢" if match['score'] >= 70 else "🟡" if match['score'] >= 40 else "🔴"
        line = f"{score_icon} *{match['score']}%* - {match['job_title']} ({match['company']})\n👉 _Empfehlung:_ *{match['recommended_action']}*\n💬 _Grund:_ {match['reason']}\n"
        output_lines.append(line)
        print(line)

    # Nachricht versenden
    full_message = "\n".join(output_lines)
    send_telegram_message(full_message)

if __name__ == "__main__":
    asyncio.run(main())
