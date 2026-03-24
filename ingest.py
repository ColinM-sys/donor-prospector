"""
Donor Prospector — Data Ingestion Pipeline
Scrapes public healthcare & donation data, chunks it, and stores in ChromaDB.
Sources: IRS 990 (ProPublica API), CMS Open Payments, web scraping by URL.
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# ── helpers ──────────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _get_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="donor_prospects",
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)

# ── IRS 990 via ProPublica ──────────────────────────────────────────────────

def fetch_irs990_orgs(query: str, state: str = "", ntee_major: str = "", page: int = 0) -> list[dict]:
    """Search ProPublica Nonprofit Explorer for orgs by name/keyword.
    Note: ProPublica API errors when combining state + ntee filters,
    so we only pass state to the API and filter ntee locally."""
    url = "https://projects.propublica.org/nonprofits/api/v2/search.json"
    params = {"q": query, "page": page}
    if state:
        params["state[id]"] = state.upper()
    # Don't pass ntee to API — it 500s when combined with state. Filter locally instead.
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    orgs = data.get("organizations", [])
    # Local NTEE filter
    if ntee_major:
        ntee_upper = ntee_major.upper()
        orgs = [o for o in orgs if (o.get("ntee_code") or "").upper().startswith(ntee_upper)]
    return orgs


def fetch_irs990_filings(ein: str) -> list[dict]:
    """Get filing history for an org by EIN."""
    url = f"https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("filings_with_data", []) or data.get("filings_without_data", [])


def fetch_org_details(ein: str) -> dict:
    """Fetch full org details including financials from ProPublica."""
    url = f"https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        org = data.get("organization", {})
        return {
            "total_revenue": org.get("income_amount", 0) or 0,
            "total_assets": org.get("asset_amount", 0) or 0,
            "tax_period": org.get("tax_period", 0) or 0,
        }
    except Exception:
        return {}


def ingest_irs990(query: str, state: str = "", sector: str = "", max_pages: int = 3) -> list[dict]:
    """Search IRS 990 data and return structured records."""
    all_records = []
    for page in range(max_pages):
        orgs = fetch_irs990_orgs(query, state=state, ntee_major=sector, page=page)
        if not orgs:
            break
        for org in orgs:
            # Determine activity status from tax period
            tax_period = org.get("tax_period", 0) or 0
            # tax_period is YYYYMM format (e.g. 202312)
            filing_year = tax_period // 100 if tax_period > 0 else 0
            current_year = datetime.now().year
            if filing_year >= current_year - 2:
                activity_status = "ACTIVE"
            elif filing_year >= current_year - 5:
                activity_status = "POSSIBLY ACTIVE"
            elif filing_year > 0:
                activity_status = "INACTIVE"
            else:
                activity_status = "UNKNOWN"

            revenue = org.get("income_amount", 0) or 0
            assets = org.get("asset_amount", 0) or 0

            # If search API returned $0, fetch full org details
            ein = str(org.get("ein", ""))
            if revenue == 0 and assets == 0 and ein:
                try:
                    details = fetch_org_details(ein)
                    if details:
                        revenue = details.get("total_revenue", 0) or 0
                        assets = details.get("total_assets", 0) or 0
                        # Update tax_period/activity if we got better data
                        if details.get("tax_period", 0):
                            tax_period = details["tax_period"]
                            filing_year = tax_period // 100 if tax_period > 0 else 0
                            if filing_year >= current_year - 2:
                                activity_status = "ACTIVE"
                            elif filing_year >= current_year - 5:
                                activity_status = "POSSIBLY ACTIVE"
                            elif filing_year > 0:
                                activity_status = "INACTIVE"
                    time.sleep(0.2)  # be polite to API
                except Exception:
                    pass

            record = {
                "source": "irs990",
                "ein": ein,
                "name": org.get("name", ""),
                "city": org.get("city", ""),
                "state": org.get("state", ""),
                "ntee_code": org.get("ntee_code", ""),
                "subsection_code": str(org.get("subsection_code", "")),
                "total_revenue": revenue,
                "total_assets": assets,
                "score": org.get("score", 0),
                "tax_period": tax_period,
                "filing_year": filing_year,
                "activity_status": activity_status,
            }
            all_records.append(record)
        time.sleep(0.3)  # be polite
    return all_records


# ── CMS Open Payments ───────────────────────────────────────────────────────

def fetch_cms_payments(physician_name: str = "", company: str = "", state: str = "", limit: int = 100) -> list[dict]:
    """Query CMS Open Payments (general payments dataset) via SODA API."""
    # Dataset ID for General Payments 2023
    url = "https://openpaymentsdata.cms.gov/api/1/datastore/query/16ccc945-3a0e-33c8-bfc0-37ffe582e984"
    conditions = []
    if physician_name:
        parts = physician_name.strip().split()
        if len(parts) >= 2:
            conditions.append({"resource": "t", "property": "covered_recipient_last_name", "value": parts[-1].upper(), "operator": "="})
            conditions.append({"resource": "t", "property": "covered_recipient_first_name", "value": parts[0].upper(), "operator": "="})
    if company:
        conditions.append({"resource": "t", "property": "applicable_manufacturer_or_applicable_gpo_making_payment_name", "value": f"%{company.upper()}%", "operator": "LIKE"})
    if state:
        conditions.append({"resource": "t", "property": "covered_recipient_primary_business_street_line1_state", "value": state.upper(), "operator": "="})

    payload = {"limit": limit, "offset": 0}
    if conditions:
        payload["conditions"] = conditions

    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = data.get("results", [])
    records = []
    for r in results:
        records.append({
            "source": "cms_open_payments",
            "physician": f"{r.get('covered_recipient_first_name', '')} {r.get('covered_recipient_last_name', '')}".strip(),
            "company": r.get("applicable_manufacturer_or_applicable_gpo_making_payment_name", ""),
            "amount": float(r.get("total_amount_of_payment_usdollars", 0)),
            "payment_nature": r.get("nature_of_payment_or_transfer_of_value", ""),
            "date": r.get("date_of_payment", ""),
            "state": r.get("covered_recipient_primary_business_street_line1_state", ""),
            "specialty": r.get("covered_recipient_specialty_1", ""),
        })
    return records


# ── Web scraper ─────────────────────────────────────────────────────────────

def scrape_url(url: str, terms: list[str] | None = None) -> list[dict]:
    """Scrape a webpage and optionally filter paragraphs by key terms."""
    headers = {"User-Agent": "DonorProspector/1.0 (research)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    paragraphs = [p.get_text(strip=True) for p in soup.find_all(["p", "li", "td", "h1", "h2", "h3"]) if p.get_text(strip=True)]
    if terms:
        term_lower = [t.lower() for t in terms]
        paragraphs = [p for p in paragraphs if any(t in p.lower() for t in term_lower)]
    records = []
    for p in paragraphs:
        records.append({
            "source": "web_scrape",
            "url": url,
            "text": p,
        })
    return records


# ── Chunking & storage ──────────────────────────────────────────────────────

NTEE_LABELS = {
    "A": "Arts, Culture & Humanities", "B": "Education",
    "C": "Environment", "D": "Animal-Related",
    "E": "Health Care", "F": "Mental Health & Crisis",
    "G": "Disease, Disorder & Medical Disciplines", "H": "Medical Research",
    "I": "Crime & Legal-Related", "J": "Employment",
    "K": "Food, Agriculture & Nutrition", "L": "Housing & Shelter",
    "M": "Public Safety & Disaster Relief", "N": "Recreation & Sports",
    "O": "Youth Development", "P": "Human Services",
    "Q": "International Affairs", "R": "Civil Rights & Social Action",
    "S": "Community Improvement", "T": "Philanthropy & Voluntarism",
    "U": "Science & Technology", "V": "Social Science Research",
    "W": "Public & Society Benefit", "X": "Religion-Related",
    "Y": "Mutual & Membership Benefit", "Z": "Unknown",
}


def _ntee_description(code: str) -> str:
    if not code:
        return "Unknown"
    major = code[0].upper()
    return NTEE_LABELS.get(major, f"Category {major}")


def _record_to_text(rec: dict) -> str:
    """Convert a structured record to a readable text block for embedding."""
    src = rec.get("source", "")
    if src == "irs990":
        ntee = rec.get("ntee_code", "")
        sector = _ntee_description(ntee)
        filing_year = rec.get("filing_year", 0)
        activity = rec.get("activity_status", "UNKNOWN")
        last_filing = f"{filing_year}" if filing_year else "Unknown"
        return (
            f"Organization: {rec['name']}\n"
            f"EIN: {rec['ein']}\n"
            f"Location: {rec['city']}, {rec['state']}\n"
            f"NTEE Code: {ntee}\n"
            f"Sector: {sector}\n"
            f"Total Revenue: ${rec['total_revenue']:,.0f}\n"
            f"Total Assets: ${rec['total_assets']:,.0f}\n"
            f"Last Filing: {last_filing}\n"
            f"Status: {activity}\n"
            f"Source: IRS 990 / ProPublica Nonprofit Explorer"
        )
    elif src == "cms_open_payments":
        return (
            f"Physician: {rec['physician']}\n"
            f"Specialty: {rec['specialty']}\n"
            f"Company: {rec['company']}\n"
            f"Payment Amount: ${rec['amount']:,.2f}\n"
            f"Nature: {rec['payment_nature']}\n"
            f"Date: {rec['date']}\n"
            f"State: {rec['state']}\n"
            f"Source: CMS Open Payments"
        )
    elif src == "web_scrape":
        return f"[Web: {rec['url']}]\n{rec['text']}"
    else:
        return json.dumps(rec, indent=2)


def store_records(records: list[dict]) -> int:
    """Chunk records and upsert into ChromaDB. Returns count of chunks added."""
    _, collection = _get_chroma()
    docs, ids, metadatas = [], [], []

    for rec in records:
        text = _record_to_text(rec)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = _hash(chunk)
            docs.append(chunk)
            ids.append(doc_id)
            meta = {
                "source": rec.get("source", "unknown"),
                "name": rec.get("name", rec.get("physician", rec.get("url", ""))),
                "state": rec.get("state", ""),
                "city": rec.get("city", ""),
                "ingested_at": datetime.now().isoformat(),
            }
            # Add NTEE sector info for IRS 990 orgs
            ntee = rec.get("ntee_code", "")
            if ntee:
                meta["ntee_code"] = ntee
                meta["sector"] = _ntee_description(ntee)
            activity = rec.get("activity_status", "")
            if activity:
                meta["activity_status"] = activity
            filing_year = rec.get("filing_year", 0)
            if filing_year:
                meta["filing_year"] = str(filing_year)
            metadatas.append(meta)

    if docs:
        # upsert in batches of 500
        for i in range(0, len(docs), 500):
            collection.upsert(
                ids=ids[i:i+500],
                documents=docs[i:i+500],
                metadatas=metadatas[i:i+500],
            )
    return len(docs)


def get_collection_stats() -> dict:
    _, collection = _get_chroma()
    count = collection.count()
    return {"total_chunks": count}


# ── CLI for quick testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <irs990|cms|web> [args...]")
        print("  irs990 <query> [state] [sector]")
        print("  cms <physician_name|company> [state]")
        print("  web <url> [term1,term2,...]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "irs990":
        query = sys.argv[2] if len(sys.argv) > 2 else "hospital"
        state = sys.argv[3] if len(sys.argv) > 3 else ""
        sector = sys.argv[4] if len(sys.argv) > 4 else ""
        print(f"Fetching IRS 990 data for '{query}'...")
        recs = ingest_irs990(query, state=state, sector=sector)
        n = store_records(recs)
        print(f"Ingested {len(recs)} orgs → {n} chunks stored")

    elif cmd == "cms":
        name = sys.argv[2] if len(sys.argv) > 2 else ""
        state = sys.argv[3] if len(sys.argv) > 3 else ""
        print(f"Fetching CMS Open Payments for '{name}'...")
        recs = fetch_cms_payments(physician_name=name, state=state)
        n = store_records(recs)
        print(f"Ingested {len(recs)} payments → {n} chunks stored")

    elif cmd == "web":
        url = sys.argv[2]
        terms = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        print(f"Scraping {url}...")
        recs = scrape_url(url, terms=terms)
        n = store_records(recs)
        print(f"Ingested {len(recs)} paragraphs → {n} chunks stored")

    print(f"Collection stats: {get_collection_stats()}")
