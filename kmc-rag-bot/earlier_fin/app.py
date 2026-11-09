"""KMC Parts Assistant: chat UI. Run locally with:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from engine import LLM, Index, load_pdfs, respond

st.set_page_config(page_title="KMC Parts Assistant", page_icon="⚙️", layout="centered")
DOCS = Path(__file__).parent / "docs"
INTENT_COLOR = {"greeting": "gray", "product_question": "blue", "buying": "green", "delivery": "violet",
                "escalation": "orange", "out_of_scope": "red"}
INTENT_NAME = {"greeting": "Greeting", "product_question": "Product question", "buying": "Buying",
               "delivery": "Order enquiry", "escalation": "Hand-off to specialist", "out_of_scope": "Out of scope"}


@st.cache_resource(show_spinner="Reading the manuals…")
def build_index(extra_files: tuple = ()):
    paths = sorted(DOCS.glob("*.pdf")) + [Path(p) for p in extra_files]
    chunks = load_pdfs(paths)
    return Index(chunks), paths, len(chunks)


@st.cache_resource
def get_llm():
    return LLM()


# ---------- sidebar: documents and settings
with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add PDFs (this session only)", type="pdf", accept_multiple_files=True)
    extra = []
    if uploads:
        tmp = Path(tempfile.gettempdir()) / "kmc_uploads"
        tmp.mkdir(exist_ok=True)
        for u in uploads:
            p = tmp / u.name
            p.write_bytes(u.getbuffer())
            extra.append(str(p))
    index, paths, n_chunks = build_index(tuple(extra))
    st.caption(f"{len(paths)} PDFs · {n_chunks} searchable passages")
    with st.expander("Documents"):
        for p in paths:
            st.write(f"• {p.stem}")
    llm = get_llm()
    st.header("Model")
    if llm.online:
        st.success("Online. Tried in order:\n\n" + "\n\n".join(f"{i}. {p['name'].title()}: {p['model']}" for i, p in enumerate(llm.providers, 1)))
    else:
        st.warning("Offline mode: no API key set. Intent uses keyword rules and answers show the closest manual passage. "
                   "Add GROQ_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY in Secrets to get written answers.")
    show_json = st.toggle("Show structured output (JSON)", value=True)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.shown_products = []
        st.rerun()

# ---------- chat
st.title("KMC Parts Assistant")
st.caption("Answers from Kaveri Motion Components' product manuals, live stock and order status. Fictional company, for demonstration.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show(res):
    st.markdown(f":{INTENT_COLOR.get(res['intent'], 'gray')}-badge[{INTENT_NAME.get(res['intent'], res['intent'])}]"
                + (" :orange-badge[Needs a person · " + res["ticket"] + "]" if res["needs_human"] else ""))
    # models sometimes send table rows with literal "\n" text instead of line breaks; fix before showing
    answer = res["answer"].replace("\\n", "\n")
    st.markdown(answer)
    if res["products"]:
        from engine import COLUMN_TITLES, show_codes
        rows = []
        for p in res["products"]:
            if not p.get("found"):
                continue
            row = {"Product": p["product_name"]}
            if show_codes():
                row["Part no."] = p["part_no"]
            if p.get("qty_requested"):
                row["Qty"] = p["qty_requested"]
            for key in ("price", "total", "availability", "delivery", "warranty", "specs", "tax"):
                if key in p:  # only the columns the buyer asked about
                    row[COLUMN_TITLES[key]] = f"{p[key]:,}" if isinstance(p[key], int) else p[key]
            rows.append(row)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
    if res["order"] and res["order"].get("verified"):  # one order: only the details asked for
        from engine import ORDER_COLUMN_TITLES
        o = res["order"]
        head = {"Order": o["order_no"], "Placed on": o["order_date"]}
        head.update({title: o[key] for key, title in ORDER_COLUMN_TITLES.items() if key in o})
        st.dataframe([head], hide_index=True, width="stretch")
        if o.get("items"):
            st.dataframe([{"Product": i["product_name"], "Part no.": i["part_no"], "Ordered": i["qty_ordered"],
                           "Dispatched": i["qty_dispatched"], "Item status": i["line_status"]} for i in o["items"]],
                         hide_index=True, width="stretch")
    if res.get("order_list") and res["order_list"].get("verified") and res["order_list"]["orders"]:  # the customer's orders
        st.dataframe([{"Order": r["order_no"], "Placed on": r["order_date"], "Status": r["status"], "Total ₹ (incl. GST)": f"{r['total_inr']:,}",
                       "Payment": r["payment"], "Delivery": r["delivery"]} for r in res["order_list"]["orders"]],
                     hide_index=True, width="stretch")
    if res.get("provider_failures") and res.get("answered_by", "").split(":")[0] in {"groq", "gemini", "openai", "custom"}:
        st.caption(f"Fallback used: answered by {res['answered_by']}")
    if res["citations"]:
        st.caption("Sources: " + " · ".join(f"{c['doc']} p.{c['page']}" + ("" if c["verified"] else " (unverified)") for c in res["citations"]))
    if show_json:
        with st.expander("Structured output"):
            st.json(res)


examples = ["What products do you sell?", "What is the pushing force of a 63 mm bore pneumatic cylinder at 6 bar?",
            "Need 20 pcs sealed 25 mm bearings with C3 clearance, price and stock?",
            "Where is order SO-24117? GSTIN ends 1ZK5", "Show our open orders, GSTIN 33ALCKA10001ZK5", "Our cylinders leak after 2 months, we want free replacements"]
if not st.session_state.messages:
    st.write("Try one:")
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, width="stretch"):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            show(m["res"])
        else:
            st.markdown(m["text"])

prompt = st.chat_input("Ask about a product, price, stock or your order…") or st.session_state.pop("pending", None)
if prompt:
    history = [m["text"] for m in st.session_state.messages if m["role"] == "user"]
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            transcript = [{"role": "customer", "text": m["text"]} if m["role"] == "user"
                          else {"role": "assistant", "text": m["res"]["answer"]} for m in st.session_state.messages[:-1][-6:]]
            res = respond(prompt, index, llm, history, st.session_state.get("shown_products"), transcript)
        show(res)
    st.session_state.messages.append({"role": "assistant", "res": res})
    if res.get("context_products"):  # remember the products this answer was about, so "both" / "them" / "as well" refer to them
        st.session_state.shown_products = res["context_products"]
