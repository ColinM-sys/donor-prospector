"""
Donor Prospector — Gradio Frontend
RAG-based agent for healthcare & donation donor prospecting.
"""

import json
import os
import threading
import gradio as gr
import requests
from ingest import (
    ingest_irs990,
    fetch_cms_payments,
    scrape_url,
    store_records,
    get_collection_stats,
)
from agent import create_agent_executor, run_query, TOOLS, DEFAULT_MODEL
from auto_sync import run_sync, load_config, save_config, add_irs990_search, add_cms_search, add_web_url, SyncScheduler
from outreach import (
    load_smtp_config, save_smtp_config, fill_template, send_email,
    preview_email, get_outreach_history, get_template_names, TEMPLATES,
)

# Use localhost when running on Spark, remote when on laptop
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_ollama_models(url=OLLAMA_URL):
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return models if models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


# ── Agent chat ───────────────────────────────────────────────────────────────

def agent_chat(message, history, model_name):
    """Run the ReAct agent and stream steps."""
    if not message.strip():
        return history, ""

    history = history + [[message, None]]
    yield history, ""

    try:
        result = run_query(message, model=model_name, ollama_url=OLLAMA_URL)

        response = ""
        if result.get("steps"):
            response += "**Agent Steps:**\n"
            for i, step in enumerate(result["steps"], 1):
                response += f"\n**Step {i}:** `{step['tool']}({json.dumps(step['input'])})`\n"
                if step.get("output"):
                    response += f"*Result:* {step['output'][:300]}\n"
            response += "\n---\n\n"

        response += result.get("output", "No response generated.")
        history[-1][1] = response

    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"

    yield history, ""


# ── Data ingestion callbacks ─────────────────────────────────────────────────

def ingest_irs(query, state, sector, progress=gr.Progress()):
    progress(0.1, desc="Searching ProPublica...")
    records = ingest_irs990(query, state=state, sector=sector)
    if not records:
        return f"No results for '{query}'", get_stats_text()
    progress(0.6, desc="Storing in vector DB...")
    n = store_records(records)
    top = sorted(records, key=lambda r: r.get("total_revenue", 0), reverse=True)[:10]
    table = "| Name | City | State | Revenue | Assets |\n|------|------|-------|---------|--------|\n"
    for r in top:
        table += f"| {r['name']} | {r['city']} | {r['state']} | ${r['total_revenue']:,.0f} | ${r['total_assets']:,.0f} |\n"
    return f"Ingested **{len(records)}** orgs ({n} chunks).\n\n**Top by revenue:**\n{table}", get_stats_text()


def ingest_cms(physician, company, state, progress=gr.Progress()):
    progress(0.1, desc="Querying CMS...")
    records = fetch_cms_payments(physician_name=physician, company=company, state=state)
    if not records:
        return "No CMS payment records found.", get_stats_text()
    progress(0.6, desc="Storing...")
    n = store_records(records)
    total = sum(r["amount"] for r in records)
    top = sorted(records, key=lambda r: r["amount"], reverse=True)[:10]
    table = "| Physician | Company | Amount | Nature |\n|-----------|---------|--------|--------|\n"
    for r in top:
        table += f"| {r['physician']} | {r['company']} | ${r['amount']:,.2f} | {r['payment_nature']} |\n"
    return f"Found **{len(records)}** payments (${total:,.2f} total), stored {n} chunks.\n\n{table}", get_stats_text()


def ingest_web(url, terms, progress=gr.Progress()):
    progress(0.1, desc="Scraping...")
    term_list = [t.strip() for t in terms.split(",") if t.strip()] if terms else None
    records = scrape_url(url, terms=term_list)
    if not records:
        return f"No content found at {url}", get_stats_text()
    progress(0.6, desc="Storing...")
    n = store_records(records)
    return f"Scraped **{len(records)}** paragraphs, stored {n} chunks.\n\nPreview:\n> {records[0]['text'][:300]}...", get_stats_text()


def get_stats_text():
    stats = get_collection_stats()
    return f"Database: **{stats['total_chunks']}** chunks stored"


# ── UI ───────────────────────────────────────────────────────────────────────

def build_ui():
    models = get_ollama_models()

    with gr.Blocks(
        title="Donor Prospector",
    ) as app:
        gr.Markdown("# Donor Prospector\n**RAG-based healthcare & donation donor prospecting agent**")

        # ── Your Charity Identity (always visible at top) ──────────
        with gr.Accordion("Your Charity Profile", open=True):
            gr.Markdown("*Enter your charity details once. This is used across all tabs to match and personalize outreach.*")
            with gr.Row():
                my_charity_name = gr.Textbox(label="Charity Name", placeholder="e.g. Hope Cancer Foundation", scale=2)
                my_charity_sector = gr.Textbox(label="Sector / Mission", placeholder="e.g. cancer research, health care", scale=2)
                my_charity_state = gr.Textbox(label="State", placeholder="e.g. CA", value="CA", scale=1)

        with gr.Row():
            model_dd = gr.Dropdown(choices=models, value=models[0], label="LLM Model", scale=1)
            stats_box = gr.Markdown(get_stats_text())
            refresh_btn = gr.Button("Refresh Models", scale=0)

        with gr.Tabs():
            # ── Chat tab ────────────────────────────────────────────────
            with gr.Tab("Agent Chat"):
                gr.Markdown("Ask the agent to find donors, search data, or analyze prospects. It will automatically use the right tools.")
                chatbot = gr.Chatbot(height=500, )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. 'Find large hospitals in Texas with revenue over $100M' or 'Search for cancer research nonprofits in California'",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")

                send_btn.click(agent_chat, [msg_input, chatbot, model_dd], [chatbot, msg_input])
                msg_input.submit(agent_chat, [msg_input, chatbot, model_dd], [chatbot, msg_input])
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

            # ── IRS 990 tab ─────────────────────────────────────────────
            with gr.Tab("IRS 990 Nonprofits"):
                gr.Markdown("Search & ingest IRS 990 nonprofit data from ProPublica Nonprofit Explorer.")
                with gr.Row():
                    irs_query = gr.Textbox(label="Search Query", placeholder="hospital, cancer foundation, etc.")
                    irs_state = gr.Textbox(label="State (optional)", placeholder="TX, CA, MA...")
                    irs_sector = gr.Dropdown(
                        label="Sector (NTEE)",
                        choices=["", "E - Health", "G - Disease/Medical", "H - Medical Research", "P - Human Services", "T - Philanthropy", "B - Education", "A - Arts"],
                        value="",
                    )
                irs_btn = gr.Button("Ingest Data", variant="primary")
                irs_output = gr.Markdown()
                irs_btn.click(
                    lambda q, s, sec: ingest_irs(q, s, sec[0] if sec else ""),
                    [irs_query, irs_state, irs_sector],
                    [irs_output, stats_box],
                )

            # ── CMS Open Payments tab ───────────────────────────────────
            with gr.Tab("CMS Open Payments"):
                gr.Markdown("Look up payments from pharma/medical device companies to physicians (public data).")
                with gr.Row():
                    cms_physician = gr.Textbox(label="Physician Name", placeholder="John Smith")
                    cms_company = gr.Textbox(label="Company", placeholder="Pfizer, Medtronic...")
                    cms_state = gr.Textbox(label="State", placeholder="MA, TX...")
                cms_btn = gr.Button("Search Payments", variant="primary")
                cms_output = gr.Markdown()
                cms_btn.click(ingest_cms, [cms_physician, cms_company, cms_state], [cms_output, stats_box])

            # ── Web Scraper tab ─────────────────────────────────────────
            with gr.Tab("Web Scraper"):
                gr.Markdown("Scrape any webpage and filter by key terms. Good for charity watchdog sites, hospital rankings, donor lists.")
                web_url = gr.Textbox(label="URL", placeholder="https://...")
                web_terms = gr.Textbox(label="Key Terms (comma-separated, optional)", placeholder="donation, healthcare, grant")
                web_btn = gr.Button("Scrape & Ingest", variant="primary")
                web_output = gr.Markdown()
                web_btn.click(ingest_web, [web_url, web_terms], [web_output, stats_box])

            # ── Auto Sync tab ──────────────────────────────────────────
            with gr.Tab("Auto Sync"):
                gr.Markdown("Configure automatic data syncing. Add searches and URLs, then run sync to pull all data at once.")

                with gr.Row():
                    sync_btn = gr.Button("Run Sync Now", variant="primary", scale=1)
                    sync_status = gr.Markdown("Ready")

                gr.Markdown("### Current Config")
                config_display = gr.JSON(value=load_config(), label="Sync Configuration")

                gr.Markdown("### Add IRS 990 Search")
                with gr.Row():
                    add_irs_q = gr.Textbox(label="Query", placeholder="hospital, foundation...")
                    add_irs_st = gr.Textbox(label="State", placeholder="TX")
                    add_irs_sec = gr.Textbox(label="NTEE Sector", placeholder="E")
                    add_irs_btn = gr.Button("Add", scale=0)

                gr.Markdown("### Add CMS Search")
                with gr.Row():
                    add_cms_co = gr.Textbox(label="Company", placeholder="Pfizer")
                    add_cms_ph = gr.Textbox(label="Physician", placeholder="John Smith")
                    add_cms_st2 = gr.Textbox(label="State", placeholder="MA")
                    add_cms_btn = gr.Button("Add", scale=0)

                gr.Markdown("### Add Web URL")
                with gr.Row():
                    add_web_u = gr.Textbox(label="URL", placeholder="https://...", scale=3)
                    add_web_t = gr.Textbox(label="Terms (comma-sep)", placeholder="donation,grant", scale=2)
                    add_web_btn = gr.Button("Add", scale=0)

                sync_log = gr.Markdown("")

                def do_sync():
                    stats = run_sync()
                    summary = (
                        f"**Sync complete** ({stats['finished_at']})\n\n"
                        f"- IRS 990: {stats['irs990_orgs']} orgs ({stats['irs990_chunks']} chunks)\n"
                        f"- CMS: {stats['cms_records']} payments ({stats['cms_chunks']} chunks)\n"
                        f"- Web: {stats['web_pages']} pages ({stats['web_chunks']} chunks)\n"
                        f"- **Total DB: {stats['total_db_chunks']} chunks**\n"
                    )
                    if stats["errors"]:
                        summary += f"\nErrors ({len(stats['errors'])}):\n"
                        for e in stats["errors"]:
                            summary += f"- {e}\n"
                    return summary, get_stats_text(), load_config()

                sync_btn.click(do_sync, outputs=[sync_log, stats_box, config_display])

                def add_irs_cfg(q, st, sec):
                    if q:
                        add_irs990_search(q, st, sec)
                    return load_config()
                add_irs_btn.click(add_irs_cfg, [add_irs_q, add_irs_st, add_irs_sec], [config_display])

                def add_cms_cfg(co, ph, st):
                    if co or ph:
                        add_cms_search(physician_name=ph, company=co, state=st)
                    return load_config()
                add_cms_btn.click(add_cms_cfg, [add_cms_co, add_cms_ph, add_cms_st2], [config_display])

                def add_web_cfg(u, t):
                    if u:
                        terms = [x.strip() for x in t.split(",") if x.strip()] if t else None
                        add_web_url(u, terms)
                    return load_config()
                add_web_btn.click(add_web_cfg, [add_web_u, add_web_t], [config_display])

            # ── Outreach tab ──────────────────────────────────────────
            with gr.Tab("Outreach"):
                gr.Markdown("### Email Outreach\nSelect prospects from the database and send personalized outreach emails.")

                with gr.Accordion("SMTP Settings", open=False):
                    smtp_cfg = load_smtp_config()
                    smtp_server = gr.Textbox(label="SMTP Server", value=smtp_cfg.get("smtp_server", ""), placeholder="smtp.gmail.com")
                    smtp_port = gr.Number(label="SMTP Port", value=smtp_cfg.get("smtp_port", 587))
                    smtp_user = gr.Textbox(label="Username/Email", value=smtp_cfg.get("username", ""))
                    smtp_pass = gr.Textbox(label="Password/App Password", value=smtp_cfg.get("password", ""), type="password")
                    with gr.Row():
                        sender_name = gr.Textbox(label="Your Name", value=smtp_cfg.get("sender_name", ""))
                        sender_title = gr.Textbox(label="Your Title", value=smtp_cfg.get("sender_title", ""))
                    with gr.Row():
                        sender_email = gr.Textbox(label="Your Email", value=smtp_cfg.get("sender_email", ""))
                        sender_phone = gr.Textbox(label="Your Phone", value=smtp_cfg.get("sender_phone", ""))
                    with gr.Row():
                        charity_name = gr.Textbox(label="Your Charity Name", value=smtp_cfg.get("charity_name", ""))
                        charity_mission = gr.Textbox(label="Your Mission (brief)", value=smtp_cfg.get("charity_mission", ""))
                    save_smtp_btn = gr.Button("Save Settings")
                    smtp_status = gr.Markdown("")

                    def save_smtp(server, port, user, pw, sn, st, se, sp, cn, cm):
                        cfg = {
                            "smtp_server": server, "smtp_port": int(port),
                            "username": user, "password": pw,
                            "sender_name": sn, "sender_title": st,
                            "sender_email": se, "sender_phone": sp,
                            "charity_name": cn, "charity_mission": cm,
                        }
                        save_smtp_config(cfg)
                        return "Settings saved."
                    save_smtp_btn.click(save_smtp,
                        [smtp_server, smtp_port, smtp_user, smtp_pass, sender_name, sender_title, sender_email, sender_phone, charity_name, charity_mission],
                        [smtp_status])

                gr.Markdown("### Compose Email")
                with gr.Row():
                    template_dd = gr.Dropdown(choices=get_template_names(), value="introduction", label="Template")
                    to_email = gr.Textbox(label="Recipient Email", placeholder="contact@org.com")
                with gr.Row():
                    contact_name_in = gr.Textbox(label="Contact Name", placeholder="Jane Smith")
                    org_name_in = gr.Textbox(label="Organization Name", placeholder="ABC Hospital")
                with gr.Row():
                    sector_in = gr.Textbox(label="Sector", placeholder="Health Care", value="Health Care")
                    state_in = gr.Textbox(label="State", placeholder="CA", value="California")
                with gr.Row():
                    impact1 = gr.Textbox(label="Impact Point 1", placeholder="Fund 50 new patient beds")
                    impact2 = gr.Textbox(label="Impact Point 2", placeholder="Expand mental health programs")
                    impact3 = gr.Textbox(label="Impact Point 3", placeholder="Train 20 new healthcare workers")

                with gr.Row():
                    preview_btn = gr.Button("Preview Email")
                    send_btn_email = gr.Button("Send Email", variant="primary")

                email_preview = gr.Textbox(label="Email Preview", lines=15, interactive=False)
                email_status = gr.Markdown("")

                def do_preview(template, contact, org, sector, state, i1, i2, i3):
                    cfg = load_smtp_config()
                    variables = {
                        "contact_name": contact or "Sir/Madam",
                        "org_name": org, "sector": sector, "state": state,
                        "sender_name": cfg.get("sender_name", "[Your Name]"),
                        "sender_title": cfg.get("sender_title", "[Your Title]"),
                        "sender_email": cfg.get("sender_email", "[Your Email]"),
                        "sender_phone": cfg.get("sender_phone", ""),
                        "charity_name": cfg.get("charity_name", "[Your Charity]"),
                        "charity_mission": cfg.get("charity_mission", "[Your Mission]"),
                        "impact_point_1": i1 or "Direct community impact",
                        "impact_point_2": i2 or "Expanded program reach",
                        "impact_point_3": i3 or "Long-term sustainability",
                    }
                    return preview_email(template, variables)

                def do_send(template, to, contact, org, sector, state, i1, i2, i3):
                    if not to:
                        return "Please enter a recipient email.", email_preview.value
                    cfg = load_smtp_config()
                    variables = {
                        "contact_name": contact or "Sir/Madam",
                        "org_name": org, "sector": sector, "state": state,
                        "sender_name": cfg.get("sender_name", ""),
                        "sender_title": cfg.get("sender_title", ""),
                        "sender_email": cfg.get("sender_email", ""),
                        "sender_phone": cfg.get("sender_phone", ""),
                        "charity_name": cfg.get("charity_name", ""),
                        "charity_mission": cfg.get("charity_mission", ""),
                        "impact_point_1": i1 or "Direct community impact",
                        "impact_point_2": i2 or "Expanded program reach",
                        "impact_point_3": i3 or "Long-term sustainability",
                    }
                    subject, body = fill_template(template, variables)
                    result = send_email(to, subject, body)
                    status = f"**{result['status'].upper()}**: {result['message']}"
                    return status, f"Subject: {subject}\n\n{body}"

                preview_btn.click(do_preview,
                    [template_dd, contact_name_in, org_name_in, sector_in, state_in, impact1, impact2, impact3],
                    [email_preview])
                send_btn_email.click(do_send,
                    [template_dd, to_email, contact_name_in, org_name_in, sector_in, state_in, impact1, impact2, impact3],
                    [email_status, email_preview])

                gr.Markdown("### Outreach History")
                history_btn = gr.Button("Load History")
                history_display = gr.Dataframe(headers=["To", "Subject", "Sent At", "Status"], label="Sent Emails")

                def load_history():
                    hist = get_outreach_history()
                    if not hist:
                        return []
                    return [[h.get("to", ""), h.get("subject", ""), h.get("sent_at", ""), h.get("status", "")] for h in hist]
                history_btn.click(load_history, outputs=[history_display])

            # ── Prospect Matcher tab ──────────────────────────────────
            with gr.Tab("Prospect Matcher"):
                gr.Markdown("### Find Verified Donor Targets\nUses your charity profile from the top. Finds **actual donors** — grant-making foundations, philanthropy orgs, and high-capacity givers — ranked by donation value.")

                with gr.Row():
                    match_top_k = gr.Slider(label="Number of Results", minimum=5, maximum=50, value=25, step=5)
                    match_btn = gr.Button("Find Donor Prospects", variant="primary", scale=2)
                    auto_email_btn = gr.Button("Find & Auto-Compose Emails", variant="secondary", scale=2)

                match_summary = gr.Markdown("Enter your charity info above and click 'Find Donor Prospects'.")

                # Selectable results table
                prospect_table = gr.Dataframe(
                    headers=["Select", "Rank", "Organization", "State", "Sector", "Revenue", "Assets", "Giving Capacity", "Status", "Donor Type", "Match Score"],
                    label="Matched Prospects (check 'Select' column to choose for outreach)",
                    interactive=True,
                )

                # Hidden state to hold prospect data for email
                prospect_state = gr.State([])

                with gr.Row():
                    email_selected_btn = gr.Button("Compose Emails for Selected", variant="primary")
                    select_all_btn = gr.Button("Select Top 10")

                email_queue_output = gr.Markdown("")

                def do_match(name, sector, state, top_k):
                    if not name or not sector:
                        return "Please enter your charity name and sector/mission.", [], []

                    import chromadb
                    import re as _re
                    from ingest import CHROMA_DIR, NTEE_LABELS

                    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                    collection = client.get_or_create_collection(name="donor_prospects")

                    if collection.count() == 0:
                        return "Database is empty. Run Auto Sync first to pull in data.", [], []

                    # Multiple semantic searches to cast a wide net
                    queries = [
                        f"{sector} foundation grant donor philanthropy {state}",
                        f"{sector} organizations donors {name}",
                        f"grant making foundation {sector} {state}",
                    ]
                    all_docs, all_metas = [], []
                    for q in queries:
                        results = collection.query(query_texts=[q], n_results=min(int(top_k) * 2, 60))
                        if results["documents"][0]:
                            all_docs.extend(results["documents"][0])
                            all_metas.extend(results["metadatas"][0])

                    if not all_docs:
                        return "No matching prospects found. Try running Auto Sync to pull more data.", [], []

                    prospects = []
                    seen = set()
                    for doc, meta in zip(all_docs, all_metas):
                        pname = meta.get("name", "Unknown")
                        if pname in seen or pname == "Unknown" or not pname.strip():
                            continue
                        seen.add(pname)

                        score = 0
                        donor_type = "Organization"
                        source = meta.get("source", "")

                        # --- Donor verification: classify what kind of donor this is ---
                        ntee = meta.get("ntee_code", "")
                        sector_meta = meta.get("sector", "")
                        activity = meta.get("activity_status", "UNKNOWN")

                        # NTEE T = Philanthropy/Voluntarism = GRANT MAKERS (highest priority)
                        if ntee and ntee[0].upper() == "T":
                            score += 40
                            donor_type = "Grant-Maker/Foundation"

                        # Check doc text for foundation/grant/donor keywords
                        doc_lower = doc.lower()
                        is_foundation = any(kw in doc_lower for kw in ["foundation", "grant", "endowment", "donor", "philanthropi", "fund"])
                        if is_foundation and donor_type == "Organization":
                            score += 20
                            donor_type = "Likely Donor/Foundation"

                        # CMS payments = pharma companies paying physicians (these ARE donors)
                        if source == "cms_open_payments":
                            score += 30
                            donor_type = "Corporate Donor (Pharma/Medical)"

                        # Activity status
                        if activity == "ACTIVE":
                            score += 25
                        elif activity == "POSSIBLY ACTIVE":
                            score += 10
                        elif activity == "INACTIVE":
                            score -= 30  # Heavy penalty — don't waste time on dead orgs

                        # Same state = more likely to donate locally
                        if meta.get("state", "").upper() == state.upper():
                            score += 15

                        # Revenue = giving capacity indicator
                        rev_match = _re.search(r"Revenue: \$([\d,]+)", doc)
                        revenue = float(rev_match.group(1).replace(",", "")) if rev_match else 0

                        asset_match = _re.search(r"Assets: \$([\d,]+)", doc)
                        assets = float(asset_match.group(1).replace(",", "")) if asset_match else 0

                        # Giving capacity = assets (endowment) weighted more than revenue
                        giving_capacity = assets * 0.05 + revenue * 0.1  # Rough: 5% of assets + 10% of rev
                        if giving_capacity > 100_000_000:
                            score += 30
                        elif giving_capacity > 10_000_000:
                            score += 25
                        elif giving_capacity > 1_000_000:
                            score += 20
                        elif giving_capacity > 100_000:
                            score += 10

                        # Sector match to YOUR charity
                        if sector_meta and sector.lower() in sector_meta.lower():
                            score += 20
                        elif any(kw in doc_lower for kw in sector.lower().split()):
                            score += 10

                        # CMS payment amount
                        pay_match = _re.search(r"Payment Amount: \$([\d,.]+)", doc)
                        if pay_match:
                            pay_amt = float(pay_match.group(1).replace(",", ""))
                            giving_capacity = max(giving_capacity, pay_amt * 10)

                        # Skip orgs with zero capacity and inactive
                        if giving_capacity == 0 and activity == "INACTIVE":
                            continue

                        prospects.append({
                            "name": pname, "state": meta.get("state", ""),
                            "sector": sector_meta or ntee, "revenue": revenue,
                            "assets": assets, "giving_capacity": giving_capacity,
                            "activity": activity, "donor_type": donor_type,
                            "source": source, "score": score,
                        })

                    # Sort by score first, then giving capacity
                    prospects.sort(key=lambda p: (p["score"], p["giving_capacity"]), reverse=True)
                    prospects = prospects[:int(top_k)]

                    if not prospects:
                        return "No verified donor prospects found. Run Auto Sync to pull more data.", [], []

                    # Build summary
                    grant_makers = sum(1 for p in prospects if "Grant" in p["donor_type"])
                    corp_donors = sum(1 for p in prospects if "Corporate" in p["donor_type"])
                    active_count = sum(1 for p in prospects if p["activity"] == "ACTIVE")
                    total_capacity = sum(p["giving_capacity"] for p in prospects)

                    summary = f"## Donor Prospects for **{name}** ({sector}, {state})\n\n"
                    summary += f"Found **{len(prospects)}** verified prospects | "
                    summary += f"**{grant_makers}** grant-makers | **{corp_donors}** corporate donors | "
                    summary += f"**{active_count}** active | Est. giving capacity: **${total_capacity:,.0f}**\n\n"
                    summary += "**Check the 'Select' column below, then click 'Compose Emails for Selected' to draft outreach.**"

                    # Build table data
                    table_data = []
                    for i, p in enumerate(prospects, 1):
                        rev = f"${p['revenue']:,.0f}" if p['revenue'] else "N/A"
                        ast = f"${p['assets']:,.0f}" if p['assets'] else "N/A"
                        cap = f"${p['giving_capacity']:,.0f}" if p['giving_capacity'] else "N/A"
                        table_data.append([
                            False,  # Select checkbox
                            i,
                            p["name"],
                            p["state"],
                            p["sector"],
                            rev,
                            ast,
                            cap,
                            p["activity"],
                            p["donor_type"],
                            p["score"],
                        ])

                    return summary, table_data, prospects

                match_btn.click(do_match,
                    [my_charity_name, my_charity_sector, my_charity_state, match_top_k],
                    [match_summary, prospect_table, prospect_state])

                def select_top_10(table_data):
                    if not table_data:
                        return table_data
                    for i, row in enumerate(table_data):
                        row[0] = True if i < 10 else row[0]
                    return table_data

                select_all_btn.click(select_top_10, [prospect_table], [prospect_table])

                def compose_for_selected(table_data, prospects, charity_name, charity_sector, charity_state):
                    if not table_data or not prospects:
                        return "No prospects to compose emails for. Run 'Find Donor Prospects' first."

                    selected = []
                    for i, row in enumerate(table_data):
                        if row[0] and i < len(prospects):
                            selected.append(prospects[i])

                    if not selected:
                        return "No prospects selected. Check the 'Select' column for the organizations you want to email."

                    cfg = load_smtp_config()
                    md = f"## Email Queue: {len(selected)} prospects selected\n\n"

                    for i, p in enumerate(selected, 1):
                        variables = {
                            "contact_name": "Sir/Madam",
                            "org_name": p["name"],
                            "sector": p.get("sector", charity_sector),
                            "state": p.get("state", charity_state),
                            "sender_name": cfg.get("sender_name", "[Your Name]"),
                            "sender_title": cfg.get("sender_title", "[Your Title]"),
                            "sender_email": cfg.get("sender_email", "[Your Email]"),
                            "sender_phone": cfg.get("sender_phone", ""),
                            "charity_name": charity_name or cfg.get("charity_name", "[Your Charity]"),
                            "charity_mission": cfg.get("charity_mission", charity_sector),
                            "impact_point_1": "Direct community impact",
                            "impact_point_2": "Expanded program reach",
                            "impact_point_3": "Long-term sustainability",
                        }

                        # Pick template based on donor type
                        if "Grant" in p.get("donor_type", ""):
                            template = "donation_request"
                        else:
                            template = "introduction"

                        subject, body = fill_template(template, variables)

                        cap = f"${p['giving_capacity']:,.0f}" if p.get('giving_capacity') else "N/A"
                        md += f"### {i}. {p['name']} ({p['state']}) — {p['donor_type']}\n"
                        md += f"**Giving Capacity:** {cap} | **Status:** {p['activity']} | **Template:** {template}\n\n"
                        md += f"**Subject:** {subject}\n\n"
                        md += f"```\n{body}\n```\n\n"
                        md += "---\n\n"

                    md += "\n**To send these emails:** Go to the **Outreach** tab, enter the recipient's email address, and use the template shown above. Configure your SMTP settings first."
                    return md

                email_selected_btn.click(compose_for_selected,
                    [prospect_table, prospect_state, my_charity_name, my_charity_sector, my_charity_state],
                    [email_queue_output])

                def auto_find_and_compose(charity_name, charity_sector, charity_state, top_k):
                    """One-click: find prospects, select top 10, compose emails."""
                    summary, table_data, prospects = do_match(charity_name, charity_sector, charity_state, top_k)
                    if not prospects:
                        return summary, table_data, prospects, "No prospects found to compose emails for."
                    # Auto-select top 10
                    for i, row in enumerate(table_data):
                        row[0] = True if i < 10 else False
                    # Auto-compose
                    emails = compose_for_selected(table_data, prospects, charity_name, charity_sector, charity_state)
                    return summary, table_data, prospects, emails

                auto_email_btn.click(auto_find_and_compose,
                    [my_charity_name, my_charity_sector, my_charity_state, match_top_k],
                    [match_summary, prospect_table, prospect_state, email_queue_output])

            # ── Dashboard tab ─────────────────────────────────────────
            with gr.Tab("Dashboard"):
                gr.Markdown("### Prospect Analytics Dashboard")
                dash_btn = gr.Button("Refresh Dashboard", variant="primary")
                dash_output = gr.Markdown("Click 'Refresh Dashboard' to load analytics from the database.")

                def build_dashboard():
                    import chromadb
                    from ingest import CHROMA_DIR
                    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                    collection = client.get_or_create_collection(name="donor_prospects")
                    total = collection.count()
                    if total == 0:
                        return "No data yet. Ingest some data first using the other tabs or run Auto Sync."

                    # Get all metadata
                    all_data = collection.get(limit=min(total, 5000), include=["metadatas", "documents"])
                    metas = all_data["metadatas"]
                    docs = all_data["documents"]

                    # Source breakdown
                    sources = {}
                    states = {}
                    for m in metas:
                        src = m.get("source", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                        st = m.get("state", "")
                        if st:
                            states[st] = states.get(st, 0) + 1

                    # Revenue analysis from IRS 990 docs
                    import re
                    revenues = []
                    orgs_by_revenue = []
                    for doc, meta in zip(docs, metas):
                        if meta.get("source") == "irs990":
                            match = re.search(r"Revenue: \$([\d,]+)", doc)
                            if match:
                                rev = float(match.group(1).replace(",", ""))
                                revenues.append(rev)
                                name_match = re.search(r"Organization: (.+)", doc)
                                name = name_match.group(1) if name_match else meta.get("name", "Unknown")
                                orgs_by_revenue.append((name, rev, meta.get("state", "")))

                    md = f"## Database Overview\n**Total chunks:** {total}\n\n"

                    # Source breakdown table
                    md += "### Data Sources\n| Source | Chunks |\n|--------|--------|\n"
                    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                        label = {"irs990": "IRS 990 Nonprofits", "cms_open_payments": "CMS Open Payments", "web_scrape": "Web Scrape"}.get(src, src)
                        md += f"| {label} | {count} |\n"

                    # Activity status breakdown
                    activity_counts = {}
                    sectors = {}
                    for m in metas:
                        act = m.get("activity_status", "")
                        if act:
                            activity_counts[act] = activity_counts.get(act, 0) + 1
                        sec = m.get("sector", "")
                        if sec:
                            sectors[sec] = sectors.get(sec, 0) + 1

                    if activity_counts:
                        md += "\n### Activity Status\n| Status | Count |\n|--------|-------|\n"
                        for status, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
                            md += f"| {status} | {count} |\n"

                    # Sector breakdown
                    if sectors:
                        md += "\n### By Sector (NTEE)\n| Sector | Count |\n|--------|-------|\n"
                        for sec, count in sorted(sectors.items(), key=lambda x: -x[1]):
                            md += f"| {sec} | {count} |\n"

                    # State breakdown (top 15)
                    if states:
                        md += "\n### Top States\n| State | Records |\n|-------|--------|\n"
                        for st, count in sorted(states.items(), key=lambda x: -x[1])[:15]:
                            md += f"| {st} | {count} |\n"

                    # Top ACTIVE orgs by revenue
                    if orgs_by_revenue:
                        orgs_by_revenue.sort(key=lambda x: -x[1])
                        md += "\n### Top Organizations by Revenue\n| Organization | State | Revenue | Status |\n|-------------|-------|---------|--------|\n"
                        seen = set()
                        for doc, meta in zip(docs, metas):
                            if meta.get("source") != "irs990":
                                continue
                            match = re.search(r"Revenue: \$([\d,]+)", doc)
                            if not match:
                                continue
                            rev = float(match.group(1).replace(",", ""))
                            name_match = re.search(r"Organization: (.+)", doc)
                            name = name_match.group(1) if name_match else meta.get("name", "Unknown")
                            if name in seen:
                                continue
                            seen.add(name)
                            status = meta.get("activity_status", "UNKNOWN")
                            if len(seen) <= 20:
                                md += f"| {name} | {meta.get('state', '')} | ${rev:,.0f} | {status} |\n"

                        # Revenue distribution
                        if revenues:
                            avg_rev = sum(revenues) / len(revenues)
                            max_rev = max(revenues)
                            min_rev = min(revenues)
                            over_100m = sum(1 for r in revenues if r > 100_000_000)
                            over_1b = sum(1 for r in revenues if r > 1_000_000_000)
                            md += f"\n### Revenue Stats\n"
                            md += f"- **Avg Revenue:** ${avg_rev:,.0f}\n"
                            md += f"- **Max Revenue:** ${max_rev:,.0f}\n"
                            md += f"- **Min Revenue:** ${min_rev:,.0f}\n"
                            md += f"- **Orgs > $100M:** {over_100m}\n"
                            md += f"- **Orgs > $1B:** {over_1b}\n"

                    # CMS payment stats
                    payments = []
                    for doc, meta in zip(docs, metas):
                        if meta.get("source") == "cms_open_payments":
                            match = re.search(r"Payment Amount: \$([\d,.]+)", doc)
                            if match:
                                payments.append(float(match.group(1).replace(",", "")))
                    if payments:
                        md += f"\n### CMS Payment Stats\n"
                        md += f"- **Total Payments:** {len(payments)}\n"
                        md += f"- **Total Value:** ${sum(payments):,.2f}\n"
                        md += f"- **Avg Payment:** ${sum(payments)/len(payments):,.2f}\n"
                        md += f"- **Max Payment:** ${max(payments):,.2f}\n"

                    return md

                dash_btn.click(build_dashboard, outputs=[dash_output])

        refresh_btn.click(lambda: gr.update(choices=get_ollama_models()), outputs=[model_dd])

    return app


if __name__ == "__main__":
    # Auto-sync on startup if configured
    config = load_config()
    if config.get("auto_sync_on_startup", False):
        print("[Startup] Running initial data sync in background...")
        sync_thread = threading.Thread(target=run_sync, daemon=True)
        sync_thread.start()

    # Start background scheduler for periodic syncs
    scheduler = SyncScheduler()
    interval = config.get("sync_interval_hours", 24)
    print(f"[Startup] Starting auto-sync scheduler (every {interval}h)...")
    scheduler.start(interval_hours=interval)

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7870, share=False)
