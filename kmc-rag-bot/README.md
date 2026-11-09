# KMC Parts Assistant: a RAG chatbot over PDF manuals

A customer-service chatbot for a fictional B2B industrial-parts supplier, Kaveri Motion Components (KMC). It reads 12 PDF manuals and policies (44 pages, named like *Pneumatic Cylinders Manual.pdf*), and for every message it:

1. **Understands the message in one AI step.** The AI receives the message, the previous question, the products currently on
   screen and the stock list, and fills in a form: intent (greeting · product question · buying · delivery status ·
   escalation · out of scope), which products (on screen / named / a category / the whole range), which details
   (price, availability, delivery, warranty, specs, GST/HSN), quantity, order number, GSTIN, and whether the product is
   one we don't carry. Code then checks the form against the data (unknown part numbers are dropped). Keyword rules are
   used only when no AI provider is reachable. The filled form is shown as `understood` in the structured output.
2. **Routes the message:**

   | Intent | What happens |
   |---|---|
   | Greeting | Fixed friendly reply, no model call (free, instant) |
   | Product question | Searches the PDFs → answer with page citations. "What do you sell?" and "price list" are answered with tables built straight from `data/families.csv` and `data/stock.csv` (exact, instant, no AI call) |
   | Buying | Finds the product by part number **or by description** ("sealed 25 mm bearing") and returns **only the columns the buyer asked about**: price (and total for the quantity), availability, delivery date, warranty, key specs, GST/HSN. If nothing specific is asked: price, availability and delivery. Delivery dates are calculated from stock, the 2 pm dispatch cut-off and Mon-Sat working days |
   | Order enquiry | One order: order no. (or PO number) + last 4 GSTIN characters → only the details asked (status, items, delivery/tracking, invoice, payment, value). All orders: full 15-character GSTIN → list (open, payment due, overdue, delivered, all). Data in `data/orders.csv`, `data/order_lines.csv`, `data/customers.csv` (100 orders, 10 customers). Contact details and discounts are never shown |
   | Escalation | Policy from the PDFs + hands off to the right team (sales, quality, engineering) with a ticket number |
   | Out of scope | Polite refusal |
| Product not carried | Any intent: "Currently we do not have this product. You can mail us directly at <CONTACT_EMAIL>." Set the address in secrets |
3. **Returns structured output** (JSON):
```json
{ "intent": "buying", "classified_by": "model", "answer": "...",
  "citations": [{"doc": "KMC-POL-ORD", "page": 2, "verified": true}],
  "products": [{"part_no": "6205-2RS-C3", "qty_requested": 20, "qty_available": 12, "unit_price_inr_ex_gst": 210, "lead_time_if_short": "10 working days"}],
  "order": null, "next_action": "none", "needs_human": false, "ticket": null,
  "sources_searched": [{"doc": "KMC-POL-ORD", "page": 2, "score": 21.7}] }
```
Answers name products in plain words ("sealed ball bearing, 25 mm bore"), with the part number in brackets; set `SHOW_PART_CODES = "never"` in secrets to hide codes. Stock and order figures are filled in by code, not written by the model, so they can't be made up. A citation the model gives that wasn't among the retrieved pages is marked `"verified": false`.

## Files
```
app.py              chat screen (Streamlit)
engine.py           PDF reading, search, intent, routing, structured output
docs/*.pdf          the knowledge base. Replace or add your own PDFs here (10-20 is fine)
data/stock.csv      product master: name, category, part no., price, GST %, HSN, stock, warranty, key specs,
                    restock_days_internal (any column ending _internal is never shown to buyers or sent to the AI)
data/families.csv   product families shown for "what do you sell?"
data/customers.csv  10 customers (GSTIN for verification; *_internal columns never shown)
data/orders.csv     100 orders: status, invoice, payment, dispatch, tracking
data/order_lines.csv items in each order
build_orders.py     regenerates the three order files
test_bot.py         runs sample messages and conversations without the UI
eval_rubric.py      accuracy rubric: scores 32 questions on routing, correct data, completeness, groundedness + safety
requirements.txt
.streamlit/secrets.toml.example
```

## 1. Run it on your laptop (VS Code)
1. Install Python 3.10+ and open this folder in VS Code.
2. In the VS Code terminal:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac)
   pip install -r requirements.txt
   streamlit run app.py
   ```
   The browser opens at http://localhost:8501. With no key it runs in **offline mode**: intent by keyword rules, and the answer is the closest manual passage.
3. For proper written answers, create `.streamlit/secrets.toml` with one or more keys (see `secrets.toml.example`):
   `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`. The bot tries them in `PROVIDER_ORDER` (default groq → gemini → openai) and
   falls back to the next one if a provider fails, hits a rate limit or times out (25 s). If all fail, it shows the closest
   manual passage. The JSON output records `answered_by` and `provider_failures`. Restart the app after editing secrets.

## 2. Put it online for free (Streamlit Community Cloud)
You need a GitHub account only.
1. Create a new **public** repository on github.com and upload this whole folder (drag and drop works; or in VS Code use *Source Control → Publish to GitHub*). `secrets.toml` is git-ignored, so your key is never uploaded.
2. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → pick the repo, branch `main`, file `app.py`.
3. Open **Advanced settings → Secrets** and paste the contents of your `secrets.toml` (all your keys).
4. Deploy. You get a public link like `https://kmc-parts.streamlit.app`. The app sleeps after 12 hours with no visitors; opening the link wakes it in about 30 seconds, so open it before your presentation.

## Measuring accuracy (eval_rubric.py)
`python eval_rubric.py` first tests the rubric itself (it feeds it 15 deliberately wrong answers: wrong price, wrong stock,
missing detail, invented number, missing citation, leaked contact... and checks each is caught), then runs 32 questions
through the bot and scores each one 0-2 on four criteria:

| Criterion | Weight | What is checked |
|---|---|---|
| Routing | 15% | correct intent |
| Correct data | 30% | right products / order, every price, stock, warranty, GST, status, invoice, items matches `data/*.csv` |
| Completeness | 25% | every asked detail present, nothing unasked shown |
| Groundedness | 30% | right document and page cited and verified; every number in the answer is found in the data or retrieved pages |
| Safety gate | pass/fail | no contact names, phones, discounts or `_internal` data; order details only with matching GSTIN |

Pass = score >= 80% and no criterion at 0. Expected answers are computed from the CSV files at run time. Results go to
`eval_results/results.csv` (open in Excel) and `eval_results/report.md`. Add `--judge` to also get the AI model's own grading
as a second opinion. Use the report numbers in the assignment's evaluation section.

## Using other PDFs or another model
- **Your own PDFs:** drop them into `docs/` (or upload in the sidebar for one session). Text-based PDFs only; scanned PDFs need OCR first. Then edit `data/stock.csv` and the examples in `app.py`.
- **Another model:** set `GROQ_MODEL`, `GEMINI_MODEL` or `OPENAI_MODEL` (and `..._FAST_MODEL`) in secrets. Run `python check_keys.py`: it tests each provider the way the bot calls it and lists the models your key can use.

## Design choices to defend
- **Two models, two jobs, three providers:** a small fast model labels the intent (Groq `openai/gpt-oss-20b`), a larger one writes answers (Groq `openai/gpt-oss-120b`). If Groq fails the bot falls back to Gemini, then OpenAI. Greetings use no model at all.
- **Keyword search (BM25), not embeddings:** part numbers like 6205 and 6305 look alike to embedding search but are different parts. BM25 needs no extra model and fits the free hosting memory. A customer-words glossary (`GLOSSARY` in engine.py) maps "free replacement" to "warranty claim".
- **Data protection:** the order lookup checks GSTIN and returns status fields only. The model never sees customer names or phone numbers.
- **Fails safe:** if the model is down or the key is missing, the app falls back to keyword rules and manual passages instead of crashing.

## No-code alternative (Dify Cloud, no programming)
Sign up at cloud.dify.ai with email or GitHub. The free Sandbox gives 200 message credits, 50 knowledge documents and 50 MB of storage.
1. **Knowledge → Create knowledge** → upload the 12 PDFs from `docs/`.
2. **Studio → Create app → Chatflow.** Add nodes: *Start → Question Classifier* with classes Greeting, Product question, Buying, Delivery, Escalation.
3. For each branch: Greeting → *Answer* (fixed text). Product question and Buying → *Knowledge Retrieval* → *LLM* (system prompt from `ANSWER_PROMPT` in engine.py, structured output on). Delivery → *LLM* that asks for order number and GSTIN. Escalation → *Answer* naming the team.
4. **Publish → Run app** gives a public web link. Limits: no live stock/order lookup without adding an HTTP tool, and 200 free messages.
