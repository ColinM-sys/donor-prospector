"""
Donor Prospector — LangGraph ReAct Agent
Tools: semantic search, IRS 990 lookup, CMS payments lookup, web scrape, filter by sector/state/revenue.
Uses Ollama locally on DGX Spark for LLM inference.
"""

import json
import os
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

import chromadb
from ingest import (
    ingest_irs990,
    fetch_cms_payments,
    scrape_url,
    store_records,
    get_collection_stats,
    CHROMA_DIR,
)

# ── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "nemotron:70b"

SYSTEM_PROMPT = """You are Donor Prospector, an AI agent that helps users find potential donors and healthcare organizations for fundraising and partnership opportunities.

Your capabilities:
1. Search IRS 990 data for nonprofits by keyword, state, and sector (NTEE codes)
2. Look up CMS Open Payments for physician-company financial relationships
3. Scrape any website for donor/healthcare information
4. Semantically search all ingested data
5. Filter prospects by source, state, sector, revenue, and location
6. Match prospects to a specific charity — find the best outreach targets
7. Search by city, zipcode, or region using search_by_location

NTEE Major Codes (for sector filtering):
- A: Arts, Culture, Humanities
- B: Education
- E: Health (hospitals, clinics, mental health)
- F: Mental Health & Crisis
- G: Disease/Disorder/Medical
- H: Medical Research
- P: Human Services
- T: Philanthropy & Voluntarism
- U: Science & Technology

IMPORTANT: Focus on finding ACTUAL DONORS — grant-making foundations (NTEE T), philanthropic orgs, corporate donors, and high-asset organizations with giving capacity. Rank results by:
1. Donor type (grant-makers > corporate donors > general orgs)
2. Giving capacity (5% of assets + 10% of revenue as estimate)
3. Activity status (ACTIVE only — skip INACTIVE)
4. Location match and sector relevance

When the user says "I am [charity name]" or describes their charity and location:
1. Use match_prospects_for_charity to find the best outreach targets from the database.
2. Present results as a ranked list with: org name, location, revenue, giving capacity, donor type, and WHY they are a good match.
3. Suggest which ones to contact first and what outreach template to use.

When the user asks for donors by location (city, zipcode, state, region):
1. Use search_by_location to find prospects in that area.
2. Rank by giving capacity and donor type.
3. Present results with actionable next steps.

When the user asks to find donors or organizations:
1. First check database_stats to see if there's data. If empty, ingest relevant data first.
2. Then search and filter to find the best matches.
3. Always verify these are actual donors/givers, not just any nonprofit.
4. Present results clearly with key details (name, location, revenue, assets, giving capacity, donor type).
5. Suggest next steps (more specific searches, website scraping, outreach)."""

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def search_prospects(query: str, top_k: int = 10) -> str:
    """Search the donor prospect database by semantic similarity.
    Use this to find organizations, physicians, or web-scraped data matching a query.
    Args:
        query: Natural language search query (e.g. 'large hospitals in Texas', 'cancer research nonprofits')
        top_k: Number of results to return (default 10)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="donor_prospects")
    results = collection.query(query_texts=[query], n_results=min(top_k, 20))

    if not results["documents"][0]:
        return "No results found. Try ingesting data first with ingest_nonprofit_data or scrape_website."

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(f"[{meta.get('source', 'unknown')}] {meta.get('name', 'N/A')} ({meta.get('state', '')})\n{doc}\n")
    return "\n---\n".join(output)


@tool
def ingest_nonprofit_data(query: str, state: str = "", sector: str = "") -> str:
    """Search and ingest IRS 990 nonprofit data from ProPublica.
    Use this to pull in new nonprofit organization data by keyword search.
    Args:
        query: Search term (e.g. 'hospital', 'cancer foundation', 'children charity')
        state: Two-letter state code filter (e.g. 'TX', 'CA'). Optional.
        sector: NTEE major code filter (e.g. 'E' for health, 'T' for philanthropy). Optional.
    """
    records = ingest_irs990(query, state=state, sector=sector)
    if not records:
        return f"No nonprofits found for '{query}'"
    n = store_records(records)
    top = sorted(records, key=lambda r: r.get("total_revenue", 0), reverse=True)[:5]
    summary = f"Ingested {len(records)} organizations ({n} chunks stored).\n\nTop 5 by revenue:\n"
    for r in top:
        summary += f"  - {r['name']} ({r['city']}, {r['state']}) — Revenue: ${r['total_revenue']:,.0f}, Assets: ${r['total_assets']:,.0f}\n"
    return summary


@tool
def lookup_cms_payments(physician_name: str = "", company: str = "", state: str = "") -> str:
    """Look up CMS Open Payments data — payments from pharmaceutical/medical device companies to physicians.
    Use this to find financial relationships between companies and healthcare providers.
    Args:
        physician_name: Full name of physician (e.g. 'John Smith'). Optional.
        company: Company name to search (e.g. 'Pfizer', 'Medtronic'). Optional.
        state: Two-letter state code (e.g. 'MA'). Optional.
    """
    records = fetch_cms_payments(physician_name=physician_name, company=company, state=state)
    if not records:
        return "No CMS payment records found."
    n = store_records(records)
    total = sum(r["amount"] for r in records)
    summary = f"Found {len(records)} payment records (${total:,.2f} total), stored {n} chunks.\n\nTop payments:\n"
    top = sorted(records, key=lambda r: r["amount"], reverse=True)[:5]
    for r in top:
        summary += f"  - {r['physician']} <- {r['company']}: ${r['amount']:,.2f} ({r['payment_nature']})\n"
    return summary


@tool
def scrape_website(url: str, key_terms: str = "") -> str:
    """Scrape a webpage and store relevant content in the prospect database.
    Use this to pull in data from charity watchdog sites, hospital rankings, donor lists, etc.
    Args:
        url: Full URL to scrape (e.g. 'https://www.charitynavigator.org/...')
        key_terms: Comma-separated filter terms (e.g. 'donation,healthcare,grant'). Optional.
    """
    terms = [t.strip() for t in key_terms.split(",") if t.strip()] if key_terms else None
    records = scrape_url(url, terms=terms)
    if not records:
        return f"No relevant content found at {url}"
    n = store_records(records)
    preview = records[0]["text"][:200] if records else ""
    return f"Scraped {len(records)} paragraphs from {url}, stored {n} chunks.\nPreview: {preview}..."


@tool
def filter_prospects(source: str = "", state: str = "", sector: str = "", min_revenue: float = 0) -> str:
    """Filter stored prospects by source type, state, sector/NTEE code, or minimum revenue.
    Args:
        source: Filter by data source ('irs990', 'cms_open_payments', 'web_scrape'). Optional.
        state: Two-letter state code. Optional.
        sector: Filter by sector name or NTEE code (e.g. 'Health Care', 'E', 'Philanthropy'). Optional.
        min_revenue: Minimum revenue threshold (only applies to IRS 990 data). Optional.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="donor_prospects")

    conditions = []
    if source:
        conditions.append({"source": source})
    if state:
        conditions.append({"state": state.upper()})
    if sector:
        # Match by NTEE code letter or sector description
        if len(sector) <= 2:
            conditions.append({"ntee_code": {"$gte": sector[0].upper(), "$lt": chr(ord(sector[0].upper()) + 1)}})
        else:
            conditions.append({"sector": sector})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = collection.get(where=where, limit=50, include=["documents", "metadatas"])

    if not results["documents"]:
        return "No matching prospects found."

    output = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if min_revenue > 0 and "Revenue: $" in doc:
            match = re.search(r"Revenue: \$([\d,]+)", doc)
            if match:
                rev = float(match.group(1).replace(",", ""))
                if rev < min_revenue:
                    continue
        output.append(f"[{meta.get('source', '')}] {meta.get('name', 'N/A')} ({meta.get('state', '')})\n{doc}\n")

    if not output:
        return f"No prospects match the filters (source={source}, state={state}, min_revenue=${min_revenue:,.0f})"
    return f"Found {len(output)} matching prospects:\n\n" + "\n---\n".join(output[:20])


@tool
def search_by_location(location: str, sector: str = "", donors_only: bool = True, top_k: int = 15) -> str:
    """Search for donor prospects by city, zipcode, or state.
    Use when user asks for donors in a specific location like 'donors in Los Angeles' or 'donors by zipcode 90210'.
    Args:
        location: City name, zipcode, state code, or region (e.g. 'Los Angeles', '90210', 'CA', 'Bay Area')
        sector: Optional sector filter (e.g. 'health care', 'education')
        donors_only: If True, prioritize actual donors/grant-makers over general orgs (default True)
        top_k: Number of results (default 15)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="donor_prospects")

    if collection.count() == 0:
        return "Database is empty. Ingest data first."

    search_query = f"organizations donors foundations in {location}"
    if sector:
        search_query += f" {sector}"
    results = collection.query(query_texts=[search_query], n_results=min(top_k * 3, 60))

    if not results["documents"][0]:
        return f"No prospects found near {location}."

    prospects = []
    seen = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        name = meta.get("name", "Unknown")
        if name in seen or name == "Unknown":
            continue
        seen.add(name)

        # Check location match — city, state, or in doc text
        loc_lower = location.lower().strip()
        doc_lower = doc.lower()
        state_meta = meta.get("state", "")
        city_meta = meta.get("city", "")

        location_match = (
            loc_lower in doc_lower or
            loc_lower in city_meta.lower() or
            city_meta.lower() == loc_lower or
            state_meta.lower() == loc_lower or
            (len(location) == 2 and state_meta.upper() == location.upper())
        )

        # Zipcode: check if it appears in the doc text
        if location.strip().isdigit() and len(location.strip()) == 5:
            location_match = location.strip() in doc

        if not location_match:
            continue

        activity = meta.get("activity_status", "UNKNOWN")
        if activity == "INACTIVE":
            continue

        # Parse revenue and assets
        rev_match = re.search(r"Revenue: \$([\d,]+)", doc)
        revenue = float(rev_match.group(1).replace(",", "")) if rev_match else 0
        asset_match = re.search(r"Assets: \$([\d,]+)", doc)
        assets = float(asset_match.group(1).replace(",", "")) if asset_match else 0
        giving_capacity = assets * 0.05 + revenue * 0.1

        ntee = meta.get("ntee_code", "")
        donor_type = "Organization"
        if ntee and ntee[0].upper() == "T":
            donor_type = "Grant-Maker"
        elif any(kw in doc_lower for kw in ["foundation", "grant", "endowment", "donor"]):
            donor_type = "Likely Donor"
        elif meta.get("source") == "cms_open_payments":
            donor_type = "Corporate Donor"

        if donors_only and donor_type == "Organization" and giving_capacity < 100_000:
            continue

        prospects.append({
            "name": name, "city": city_meta, "state": state_meta,
            "sector": meta.get("sector", ntee),
            "revenue": revenue, "assets": assets,
            "giving_capacity": giving_capacity,
            "activity": activity, "donor_type": donor_type,
        })

    prospects.sort(key=lambda p: p["giving_capacity"], reverse=True)
    prospects = prospects[:top_k]

    if not prospects:
        return f"No verified donors found in {location}. Try broadening the search or ingesting more data for that area."

    output = f"## Donors near {location}\n\n"
    output += "| # | Organization | City | State | Sector | Revenue | Giving Capacity | Status | Type |\n"
    output += "|---|-------------|------|-------|--------|---------|-----------------|--------|------|\n"
    for i, p in enumerate(prospects, 1):
        rev = f"${p['revenue']:,.0f}" if p['revenue'] else "N/A"
        cap = f"${p['giving_capacity']:,.0f}" if p['giving_capacity'] else "N/A"
        output += f"| {i} | {p['name']} | {p['city']} | {p['state']} | {p['sector']} | {rev} | {cap} | {p['activity']} | {p['donor_type']} |\n"

    return output


@tool
def match_prospects_for_charity(charity_name: str, charity_sector: str, charity_state: str, top_k: int = 20) -> str:
    """Find the best-matched outreach prospects for a specific charity.
    Searches the database and ranks results by relevance — prioritizing ACTIVE orgs in the same state/sector with high revenue.
    Args:
        charity_name: Name of the charity looking for prospects (e.g. 'Hope Cancer Foundation')
        charity_sector: What sector the charity works in (e.g. 'health care', 'cancer research', 'education', 'mental health')
        charity_state: State where the charity operates (e.g. 'CA', 'TX')
        top_k: Number of results to return (default 20)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="donor_prospects")

    if collection.count() == 0:
        return "Database is empty. Ingest data first using ingest_nonprofit_data."

    # Semantic search using the charity's description
    search_query = f"{charity_sector} organizations in {charity_state} donors philanthropy grants {charity_name}"
    results = collection.query(query_texts=[search_query], n_results=min(top_k * 3, 100))

    if not results["documents"][0]:
        return "No matching prospects found."

    # Score and rank prospects
    prospects = []
    seen_names = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        name = meta.get("name", "Unknown")
        if name in seen_names or name == "Unknown":
            continue
        seen_names.add(name)

        score = 0
        # Prefer ACTIVE orgs
        activity = meta.get("activity_status", "UNKNOWN")
        if activity == "ACTIVE":
            score += 30
        elif activity == "POSSIBLY ACTIVE":
            score += 15
        elif activity == "INACTIVE":
            score -= 20

        # Same state bonus
        if meta.get("state", "").upper() == charity_state.upper():
            score += 20

        # Revenue from doc text
        rev_match = re.search(r"Revenue: \$([\d,]+)", doc)
        revenue = 0
        if rev_match:
            revenue = float(rev_match.group(1).replace(",", ""))
            if revenue > 1_000_000_000:
                score += 25
            elif revenue > 100_000_000:
                score += 20
            elif revenue > 10_000_000:
                score += 15
            elif revenue > 1_000_000:
                score += 10

        # Assets from doc text
        asset_match = re.search(r"Assets: \$([\d,]+)", doc)
        assets = 0
        if asset_match:
            assets = float(asset_match.group(1).replace(",", ""))

        # Sector relevance
        sector_meta = meta.get("sector", "")
        if sector_meta and charity_sector.lower() in sector_meta.lower():
            score += 15

        # Philanthropy/grant-making orgs get bonus (they GIVE money)
        ntee = meta.get("ntee_code", "")
        if ntee and ntee[0].upper() == "T":
            score += 25  # Philanthropy orgs are prime targets

        source = meta.get("source", "unknown")
        prospects.append({
            "name": name,
            "state": meta.get("state", ""),
            "sector": sector_meta or meta.get("ntee_code", ""),
            "revenue": revenue,
            "assets": assets,
            "activity": activity,
            "source": source,
            "score": score,
            "snippet": doc[:200],
        })

    # Sort by score descending, then revenue
    prospects.sort(key=lambda p: (p["score"], p["revenue"]), reverse=True)
    prospects = prospects[:top_k]

    if not prospects:
        return "No suitable prospects found. Try ingesting more data."

    output = f"## Top {len(prospects)} Outreach Prospects for {charity_name}\n"
    output += f"Matched by: {charity_sector} in {charity_state}\n\n"
    output += "| Rank | Organization | State | Sector | Revenue | Assets | Status | Match Score |\n"
    output += "|------|-------------|-------|--------|---------|--------|--------|-------------|\n"
    for i, p in enumerate(prospects, 1):
        rev_str = f"${p['revenue']:,.0f}" if p['revenue'] else "N/A"
        asset_str = f"${p['assets']:,.0f}" if p['assets'] else "N/A"
        output += f"| {i} | {p['name']} | {p['state']} | {p['sector']} | {rev_str} | {asset_str} | {p['activity']} | {p['score']} |\n"

    # Add recommendations
    active_high_rev = [p for p in prospects if p["activity"] == "ACTIVE" and p["revenue"] > 10_000_000]
    if active_high_rev:
        output += f"\n**Recommended first contacts:** {', '.join(p['name'] for p in active_high_rev[:5])}"
        output += "\nThese are ACTIVE organizations with significant revenue — highest likelihood of engagement."

    philanthropy = [p for p in prospects if "Philanthropy" in p.get("sector", "")]
    if philanthropy:
        output += f"\n\n**Grant-making orgs (they give grants):** {', '.join(p['name'] for p in philanthropy[:5])}"

    return output


@tool
def database_stats() -> str:
    """Show current database statistics — how many records are stored."""
    stats = get_collection_stats()
    return f"Database contains {stats['total_chunks']} chunks across all sources."


# ── Agent setup ──────────────────────────────────────────────────────────────

TOOLS = [
    search_prospects,
    ingest_nonprofit_data,
    lookup_cms_payments,
    scrape_website,
    filter_prospects,
    search_by_location,
    match_prospects_for_charity,
    database_stats,
]


def create_agent_executor(model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
    llm = ChatOllama(
        model=model,
        base_url=ollama_url,
        temperature=0.1,
    )
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def run_query(query: str, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL) -> dict:
    agent = create_agent_executor(model=model, ollama_url=ollama_url)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    # Extract final AI message
    messages = result.get("messages", [])
    output = ""
    steps = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({"tool": tc["name"], "input": tc["args"]})
        if msg.type == "tool":
            if steps:
                steps[-1]["output"] = msg.content[:500]
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            output = msg.content
    return {"output": output or "No response generated.", "steps": steps}


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Find large hospitals in Massachusetts"
    print(f"\nQuery: {query}\n")
    result = run_query(query)
    print(f"\nResult:\n{result['output']}")
