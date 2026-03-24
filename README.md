# Donor Prospector

AI-powered donor prospecting agent for healthcare and nonprofit fundraising. Built at Hack for Impact @ GTC 2026 using NVIDIA NeMo Agent Toolkit and LangGraph.

## Features

- **IRS 990 Nonprofit Search** — Search ProPublica's IRS 990 database for nonprofits by keyword, state, and NTEE sector code
- **Search by Zipcode/City/State** — Find donor prospects by location with giving capacity estimates
- **CMS Open Payments Lookup** — Find pharmaceutical/medical device company payments to physicians
- **Web Scraping** — Scrape any website for donor/healthcare data and store in vector DB
- **Smart Prospect Matching** — Tell the agent your charity's name and sector, it finds the best outreach targets ranked by match score
- **Semantic Search** — RAG-powered search across all ingested data using ChromaDB
- **Email Outreach Templates** — Pre-built email templates for donor outreach with SMTP integration
- **Auto-Sync** — Scheduled data ingestion from multiple sources
- **ReAct Agent** — LangGraph-based agent that autonomously decides which tools to use

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Nemotron 70B via Ollama (local) |
| Agent Framework | LangGraph ReAct Agent |
| Vector DB | ChromaDB (persistent) |
| Data Sources | ProPublica IRS 990, CMS Open Payments, Web Scraping |
| Frontend | Gradio |
| Email | SMTP with customizable templates |

## Agent Tools (8 total)

1. `search_prospects` — Semantic search over all ingested data
2. `ingest_nonprofit_data` — Pull IRS 990 data from ProPublica
3. `lookup_cms_payments` — Search CMS Open Payments database
4. `scrape_website` — Scrape and ingest any webpage
5. `filter_prospects` — Filter by source, state, sector, revenue
6. `search_by_location` — Find donors by city, zipcode, or state
7. `match_prospects_for_charity` — AI-ranked prospect matching for your charity
8. `database_stats` — Current database statistics

## Quick Start

```bash
# Install dependencies
pip install gradio chromadb langchain-ollama langgraph beautifulsoup4 requests

# Make sure Ollama is running with a model
ollama pull nemotron:70b

# Run
python app.py
```

Open http://localhost:7870

## Example Queries

- "Find large hospitals in Massachusetts"
- "Search for cancer research foundations in California"
- "Find donors by zipcode 90210"
- "I am Hope Cancer Foundation in Texas — find me the best outreach targets"
- "Look up CMS payments to physicians in New York"

## Built For

**Hack for Impact @ GTC 2026**
