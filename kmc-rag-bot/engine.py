"""KMC Parts Assistant: the engine behind app.py.

Flow for every customer message:
  1. classify intent   greeting | product_question | buying | delivery | escalation | out_of_scope
  2. route             greeting -> fixed reply (no model call)
                       product_question -> search the PDFs -> answer with page citations
                       buying -> stock lookup (CSV) + ordering policy from the PDFs
                       delivery -> order lookup (CSV) after order no. + GSTIN check
                       escalation -> policy from the PDFs + hand-off to the right team
  3. return one structured JSON object (see RESPONSE SHAPE at the bottom)

Facts from the stock and order files are inserted by code, never typed by the model, so the model cannot invent
a stock figure or leak a customer's phone number. With no API key the app still runs in "offline mode":
keyword-based intent and the best-matching manual passage instead of a written answer.
"""
import time
import csv, json, math, os, re, uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).parent
INTENTS = ["greeting", "product_question", "buying", "delivery", "escalation", "out_of_scope"]

# ---------------------------------------------------------------- 1. reading the PDFs
def load_pdfs(paths, chunk_chars=1400, overlap=200):
    """One chunk per page, split further if a page is long. Every chunk remembers its file and page."""
    chunks = []
    for path in paths:
        reader = PdfReader(str(path))
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else Path(path).stem)
        doc = Path(path).stem
        pages = [(page.extract_text() or "").splitlines() for page in reader.pages]
        # drop running headers/footers: lines repeated on most pages, and "Page N"
        counts = Counter(l.strip() for lines in pages for l in set(lines))
        repeated = {l for l, c in counts.items() if len(pages) >= 2 and c >= 0.6 * len(pages)}
        for pno, lines in enumerate(pages, start=1):
            keep = [l for l in lines if l.strip() not in repeated and not re.fullmatch(r"\s*Page \d+\s*", l)]
            text = re.sub(r"[ \t]+", " ", "\n".join(keep)).strip()
            text = re.sub(r"\n(?=[a-z0-9(])", " ", text)  # re-join lines broken mid-sentence
            if len(text) < 40:
                continue
            start = 0
            while start < len(text):
                piece = text[start:start + chunk_chars]
                chunks.append({"doc": doc, "title": title, "page": pno, "text": piece})
                if start + chunk_chars >= len(text):
                    break
                start += chunk_chars - overlap
    return chunks


# ---------------------------------------------------------------- 2. search (BM25 keyword ranking)
STOP = set("a an the of to and or for in on at is are be with by as it its this that from what which when how can we our do does you your i my me any please have has need want tell show about give get".split())
# customer words -> words the manuals use; grow this list from real questions
GLOSSARY = {"replacement": "warranty claim", "replace": "warranty claim", "free": "warranty", "refund": "warranty claim credit",
            "faulty": "warranty defect", "defective": "warranty defect", "discount": "price quotation", "leaking": "leakage seal",
            "leak": "leakage seal", "price": "price gst", "cost": "price gst", "delivery": "dispatch transit", "buy": "order price",
            "pressure": "operating pressure", "bar": "operating pressure",
            "payment": "credit advance invoice accounts", "cutoff": "dispatch pm", "cut": "dispatch pm",
            "sell": "what we sell product families range", "sells": "what we sell product families range",
            "offer": "what we sell product families range", "products": "what we sell product families range",
            "catalogue": "what we sell product families range", "catalog": "what we sell product families range",
            "range": "product families", "portfolio": "product families range", "items": "popular stocked items"}


def expand(query):
    """Question words plus the manual words they map to."""
    raw = re.findall(r"[a-z]+", query.lower())
    return tokenize(query) + tokenize(" ".join(GLOSSARY.get(w, "") for w in raw))


def tokenize(text):
    text = text.lower().replace("×", "x")
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9./\-]*[a-z0-9]|[a-z0-9]", text):
        w = w.strip(".-/")
        if not w or w in STOP:
            continue
        if w.isalpha() and len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # "bearings" -> "bearing", "valves" -> "valve"
        out.append(w)
        if re.search(r"[-/.]", w):  # keep "6205-2rs" whole AND as "6205", "2rs": part numbers matter
            for p in re.split(r"[-/.]", w):
                if p and p not in STOP:
                    out.append(p[:-1] if p.isalpha() and len(p) > 3 and p.endswith("s") and not p.endswith("ss") else p)
    return out


class Index:
    def __init__(self, chunks, k1=1.4, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokenize(f"{c['title']} {c['text']}") for c in chunks]
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - m + 0.5) / (m + 0.5)) for t, m in df.items()}

    def search(self, query, k=4):
        q = expand(query)
        scores = []
        for i, (tf, L) in enumerate(zip(self.tf, self.len)):
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * L / self.avg)) for t in q if t in tf)
            scores.append((s, i))
        scores.sort(reverse=True)
        return [dict(self.chunks[i], score=round(s, 2)) for s, i in scores[:k] if s > 0]


# ---------------------------------------------------------------- 3. the "ERP": stock and orders
def _read_csv(name):
    with open(ROOT / "data" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


STOCK = _read_csv("stock.csv")
FAMILIES = _read_csv("families.csv")
ORDERS = _read_csv("orders.csv")
ORDER_LINES = _read_csv("order_lines.csv")
CUSTOMERS = {c["customer_id"]: c for c in _read_csv("customers.csv")}
_norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
# customer contact columns (*_internal) are never returned; buyers see only their own orders after a GSTIN check


def find_part_numbers(text):
    """Stock part numbers mentioned in the text, tolerant of spaces/hyphens ("6205-2RS C3" -> 6205-2RS-C3)."""
    t, found = _norm(text), []
    for p in sorted(STOCK, key=lambda r: -len(_norm(r["part_no"]))):
        n = _norm(p["part_no"])
        if n in t:
            found.append(p["part_no"])
            t = t.replace(n, "#")
    return found


def find_products_by_name(text, limit=4):
    """Match a description like "25 mm sealed bearing with C3" to stocked products. Numbers count double.
    Returns part numbers of the best matches (several if they score about the same, so the bot can offer options)."""
    q = set(tokenize(text))
    qty = find_quantity(text)
    if qty:
        q.discard(str(qty))  # "20 sealed bearings": 20 is a quantity, not a 20 mm size
    scored = []
    for r in STOCK:
        name = set(tokenize(f"{r['product_name']} {r['category']}"))
        hit = q & name
        score = sum(2 if t.isdigit() else 1 for t in hit)
        score -= 2 * len((name & {"kit", "seal", "spider", "coil", "spare"}) - q)  # don't offer spares unless asked
        has_family = bool(q & set(tokenize(r["category"])) or q & {"bearing", "cylinder", "valve", "coil", "motor", "coupling", "spider", "adapter", "seal", "kit"} & name)
        if has_family and score >= 2:
            scored.append((score, r["part_no"]))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    cut = best if best >= 3 else best - 1   # "sealed ball bearing" should not also return shielded ones
    return [p for s, p in sorted(scored, reverse=True) if s >= cut][:limit]


GENERIC = set(tokenize("price prices cost rate rates stock available availability need want buy purchase order quote quotation "
                        "pcs piece pieces nos unit units qty quantity urgent urgently kindly send much many lead time delivery chennai "
                        "sir madam hello thanks item items product products sell supply require required looking interested "
                        "number numbers per each also some like would could know check list enquiry inquiry"))


def unknown_codes(text):
    """Part-number-like words (letters + digits + hyphen, e.g. 6206-2RS, PC15-80-300) that are not in stock."""
    stocked = {_norm(r["part_no"]) for r in STOCK}
    codes = re.findall(r"\b(?=[A-Za-z0-9.-]*\d)(?=[A-Za-z0-9.-]*[A-Za-z])[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)+\b", text)
    return [c.upper() for c in codes if _norm(c) not in stocked and not c.upper().startswith("SO-")
            and not any(_norm(c) in s or s in _norm(c) for s in stocked)]


def mentions_product(text):
    """True if the message names a kind of product ("pump", "belts", "sensor"...), not just price/quantity words."""
    return any(t in PRODUCT_NOUNS for t in tokenize(text))


PRODUCT_NOUNS = {"bearing", "cylinder", "valve", "motor", "coupling", "adapter", "fitting", "kit", "coil", "spider",
                 "regulator", "filter", "pump", "belt", "gearbox", "sensor", "drive", "hose", "chain", "sprocket", "actuator"}


CARRIED_NOUNS = {t for r in STOCK for t in tokenize(f"{r['product_name']} {r['category']}")} & PRODUCT_NOUNS | {"filter", "regulator", "fitting"}


def unrecognised_words(text, index):
    """Unknown describing words placed right before a product noun, e.g. "servo" in "servo motor" or "vacuum" in
    "vacuum pump". Words we have never seen in our manuals or stock list mean we probably don't carry that product.
    Ordinary words elsewhere in the sentence ("quickly", "well") are ignored."""
    vocab = set(index.idf) | {t for r in STOCK for t in tokenize(f"{r['product_name']} {r['category']}")}
    toks = tokenize(text)
    out = set()
    for a, b in zip(toks, toks[1:]):
        if b in PRODUCT_NOUNS and a.isalpha() and len(a) >= 3 and a not in vocab and a not in GENERIC:
            out.add(a)
    # a kind of product that none of our stocked items or catalogues covers ("pump", "belt", "sensor")
    out |= {t for t in toks if t in PRODUCT_NOUNS and t not in CARRIED_NOUNS}
    return sorted(out)


CATALOGUE_RE = re.compile(r"\b(sell|offer|catalog(ue)?|range|portfolio|product ?list|all (your |the |of your )?products|"
                          r"what (do|can) you (have|supply|stock)|price ?list|list (down |of |out )?(all |your |all your |all the )(prices?|products)|"
                          r"list (down |of |out )?(the )?products|"
                          r"prices? (of|for) (all|every|each|them|these|those)|all (the )?prices)\b", re.I)
FOLLOWUP_PRICE_RE = re.compile(r"\b(prices?|costs?|rates?|warranty|gst|hsn|availability|stock)\b.*\b(as well|also|too|them|these|those|all|each|every)\b|"
                               r"\b(their|those|these|all|each)\b.*\b(prices?|costs?|rates?|warranty|gst|hsn|availability|stock)\b", re.I)


LISTING_RE = re.compile(r"\b(different|types? of|kinds? of|varieties|variants|options|models|range of|list (of |down |all |out )?|"
                        r"(what|which) .*\b(do you|you|can you) (have|sell|offer|stock|supply|carry)|"
                        r"(what|which) .*\b(available|in stock))\b|\bproducts\b", re.I)


def is_catalogue(msg, recent):
    """'What do you sell?', 'price list', or a follow-up like 'list the prices as well' after such a question.
    Not when a specific kind of product is named ("do you sell vacuum pumps?" is an availability question)."""
    if mentions_product(msg) or find_part_numbers(msg) or unknown_codes(msg):
        return False
    if CATALOGUE_RE.search(msg):
        return True
    # follow-up ("price as well") counts as the full price list only if the previous question was about the whole range,
    # not about one kind of product ("what ball bearings do you sell?" -> prices of the ball bearings only)
    prev = recent or ""
    prev_is_general = CATALOGUE_RE.search(prev) and not (mentions_product(prev) or find_products_by_name(prev) or find_part_numbers(prev))
    return bool(FOLLOWUP_PRICE_RE.search(msg) and prev_is_general)


def wants_prices(msg):
    return bool(re.search(r"\b(prices?|costs?|rates?|₹|rs\.?|inr)\b", msg, re.I))


def find_products_by_category(text):
    """'couplings', 'gear motors', 'solenoid valves' -> every stocked item in that category."""
    q = set(tokenize(text))
    for cat in dict.fromkeys(r["category"] for r in STOCK):
        if tokenize(cat)[-1] in q:  # main noun of the category: bearing, cylinder, valve, motor, coupling, adapter
            return [r["part_no"] for r in STOCK if r["category"] == cat]
    return []


def _plain(t):
    return " " + re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() + " "


def find_exact_names(text):
    """Products whose full name appears in the message (the customer copied it from a table)."""
    t = _plain(text)
    hits = [r for r in STOCK if _plain(r["product_name"]) in t]
    hits.sort(key=lambda r: -len(r["product_name"]))  # the longest (most specific) name wins
    return [hits[0]["part_no"]] if hits else []


def find_products(text):
    """Exact product name first, then a match by description, else the whole category the customer named."""
    if LISTING_RE.search(text) and not find_exact_names(text) and find_products_by_category(text):
        return find_products_by_category(text)  # "what filter-regulators do you sell": the whole family, spares included
    return find_exact_names(text) or find_products_by_name(text) or find_products_by_category(text)


def product_range():
    """Short catalogue for the model: every stocked product by category (name, part no., price)."""
    by_cat = {}
    for r in STOCK:
        by_cat.setdefault(r["category"], []).append(f"{r['product_name']} [{r['part_no']}] Rs {int(r['unit_price_inr']):,}")
    return "\n".join(f"- {c}: " + "; ".join(items) for c, items in by_cat.items())


def product_table():
    """Markdown table of the stock list grouped by category (used when no AI model is available)."""
    by_cat = {}
    for r in STOCK:
        by_cat.setdefault(r["category"], []).append(r)
    rows = ["| Product family | Stocked items | Price from (₹, ex-GST) |", "|---|---|---|"]
    for cat, items in by_cat.items():
        names = "; ".join(i["product_name"] for i in items)
        rows.append(f"| {cat} | {names} | {min(int(i['unit_price_inr']) for i in items):,} |")
    return "\n".join(rows)


def show_codes():
    return str(get_setting("SHOW_PART_CODES", "brackets")).lower() != "never"


def price_table():
    """Every stocked product with its list price, straight from data/stock.csv (never written by the AI)."""
    codes = show_codes()
    head = "| Product | " + ("Part no. | " if codes else "") + "Price (₹, ex-GST) | Availability |"
    rows = [head, "|---|" + ("---|" if codes else "") + "---|---|"]
    for r in STOCK:
        rows.append(f"| {r['product_name']} | " + (f"{r['part_no']} | " if codes else "") + f"{int(r['unit_price_inr']):,} | {availability_text(r)} |")
    return "\n".join(rows)


def family_table():
    """The product families we sell, from data/families.csv."""
    rows = ["| Product family | What we offer | Sizes / ratings |", "|---|---|---|"]
    rows += [f"| {f['product_family']} | {f['what_we_offer']} | {f['sizes_ratings']} |" for f in FAMILIES]
    return "\n".join(rows)


def range_table(fields):
    """Every stocked product with only the asked columns (price, warranty, availability, specs, GST/HSN)."""
    keys = [k for k in ("price", "availability", "warranty", "specs", "tax") if k in fields] or ["price", "availability"]
    codes = show_codes()
    head = ["Product"] + (["Part no."] if codes else []) + [COLUMN_TITLES[k] for k in keys]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in STOCK:
        p = lookup_stock(r["part_no"], None, keys)
        cells = [p["product_name"]] + ([p["part_no"]] if codes else []) + [f"{p[k]:,}" if isinstance(p[k], int) else p[k] for k in keys]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def catalogue_answer(with_prices, fields=None):
    fields = fields or []
    if with_prices or set(fields) & {"warranty", "specs", "tax", "availability"}:
        text = ("Here is our full range" + (" with prices. Prices are ex-works Chennai, GST 18% extra" if "price" in fields or with_prices else "") +
                ".\n\n" + range_table(fields or ["price", "availability"]) +
                ("\n\nFor volume prices, ask for a quotation from our sales team." if "price" in fields or with_prices else ""))
    else:
        text = (f"We supply {len(FAMILIES)} product families from our Chennai stock:\n\n" + family_table() +
                "\n\nAsk me for the price list, or about any product for its details.")
    return {"answer": text, "citations": [{"doc": "Company Profile and Product Range", "page": 2}], "next_action": "none"}


UNIT_WORDS = r"(?:mm|cm|m|kw|w|v|vdc|vac|bar|nm|rpm|inch|in|hz|kn|n|kg|%|degrees?|deg|micron|percent|days?|hours?|h|months?)"


def find_quantity(text):
    m = (re.search(r"\b(\d{1,5})\s*(?:pcs|pieces|nos|no\.|units|numbers|qty|x\b(?!\s*\d))", text, re.I)  # "2 x cylinder", not "25 x 52"
         or re.search(r"\bqty\.?\s*(\d{1,5})", text, re.I)
         # "10 solenoid valves", "20 sealed bearings": a number followed (within 3 words) by a product word, not a unit
         or re.search(rf"\b(\d{{1,5}})\s+(?!{UNIT_WORDS}\b)(?:[a-z/-]+\s+){{0,3}}(?:{'|'.join(PRODUCT_NOUNS)})s?\b", text, re.I))
    return int(m.group(1)) if m else None


def find_order_no(text):
    m = re.search(r"\bSO[-\s]?(\d{4,6})\b", text, re.I)
    return f"SO-{m.group(1)}" if m else None


def find_gstin_full(text):
    m = re.search(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z0-9]{4})\b", text.upper())
    return m.group(1) if m else None


def find_gstin4(text):
    full = find_gstin_full(text)
    if full:
        return full[-4:]
    m = re.search(r"gstin[^A-Za-z0-9]{0,20}(?:no\.?|number)?\s*(?:ends?(?: with| in)?|last (?:4|four)(?: characters| digits)?(?: are| is)?)?\s*[:\-]?\s*([A-Z0-9]{4})\b", text, re.I)
    return m.group(1).upper() if m else None


def find_po(text):
    m = re.search(r"\bPO[-\s#:]*(\d{4,7})\b", text, re.I)
    return f"PO-{m.group(1)}" if m else None


# ---------- what the buyer asked about, and only that
# Columns in data/stock.csv ending in "_internal" are never shown to customers or sent to the AI.
FIELD_PATTERNS = {
    "price":        r"\b(price|prices|cost|costs|rate|rates|how much|quote|quotation|rs\.?|inr|amount|total)\b|₹",
    "availability": r"\b(stock|available|availability|in hand|how many|have you got|do you have|ready stock)\b",
    "delivery":     r"\b(deliver\w*|dispatch\w*|shipp?\w*|arriv\w*|how soon|lead time|eta|by when|when (will|can|would|could|shall|do|does) (it|they|you|we|i|the|my|this|these)\b.*\b(get|reach|come|arrive|send|ship|deliver)\w*)\b",
    "warranty":     r"\b(warr?an?t[iy]e?s?|warr?ent[iy]e?|guar?[ae]n?t[eiy]+e?|guaranty|garant\w*|guarantee period|warranty period)\b",
    "specs":        r"\b(spec|specs|specification|specifications|technical details|load rating|ratings?|torque|rpm|dimensions?)\b",
    "tax":          r"\b(gst|tax|taxes|hsn|invoice)\b",
}
DEFAULT_FIELDS = ["price", "availability", "delivery"]   # when the buyer doesn't say what they want to know
COLUMN_TITLES = {"price": "Price (₹/unit, ex-GST)", "total": "Total (₹, ex-GST)", "availability": "Availability",
                 "delivery": "Delivery", "warranty": "Warranty", "specs": "Key specs", "tax": "GST / HSN"}
IST = timezone(timedelta(hours=5, minutes=30))
DISPATCH_CUTOFF_HOUR = 14  # orders confirmed before 2:00 pm IST ship the same working day (Ordering and Delivery Terms)


# Words per field for typo-tolerant matching ("gurantee", "delivary", "prise", "availabilty")
FIELD_WORDS = {
    "price": ["price", "prices", "cost", "rate", "rates", "quotation"],
    "availability": ["stock", "available", "availability"],
    "delivery": ["delivery", "deliver", "delivered", "dispatch", "shipping", "arrival"],
    "warranty": ["warranty", "guarantee", "guaranty", "warrantee"],
    "specs": ["specs", "specification", "specifications", "dimensions", "torque", "rating"],
    "tax": ["gst", "hsn", "tax", "invoice"],
}


def explicit_fields(text):
    import difflib
    found = [f for f, pat in FIELD_PATTERNS.items() if re.search(pat, text, re.I)]
    for w in re.findall(r"[a-z]{4,}", text.lower()):  # catch misspellings
        for f, words in FIELD_WORDS.items():
            if f not in found and difflib.get_close_matches(w, words, n=1, cutoff=0.8):
                found.append(f)
    return [f for f in FIELD_PATTERNS if f in found]  # keep a fixed column order


def asked_fields(text):
    return explicit_fields(text) or list(DEFAULT_FIELDS)


POLICY_TOPIC_RE = re.compile(r"\b(discount\w*|policy|policies|terms|return\w*|refund\w*|replacement|payment|credit|minimum order|"
                             r"min\.? order|handling charge|warrant\w*|guarantee|delivery|shipping|freight|transport\w*|gst|"
                             r"cancel\w*|price list|pricing|quotation|bulk|volume|moq)\b", re.I)
POLICY_ASK_RE = re.compile(r"\b(policy|policies|terms|rules?|conditions?|procedure|process)\b|^\s*(what|how|is|are|do|does|can|will|"
                           r"tell me|explain)\b", re.I)
PROBLEM_RE = re.compile(r"\b(leak\w*|faulty|broken|damag\w*|defect\w*|not working|failed|stopped|burnt|cracked|we want|i want|"
                        r"give (us|me)|can you give|approve)\b|\d+\s*(%|percent|pcs|nos|pieces|units)", re.I)


def is_policy_question(text):
    """'what is the discount policy?', 'do you accept returns?': a question about KMC's terms, answered from the PDFs.
    Not a specific request ('give 15% discount on 200 pcs') or a problem report ('cylinder leaking')."""
    return bool(POLICY_TOPIC_RE.search(text) and POLICY_ASK_RE.search(text) and not PROBLEM_RE.search(text))


def is_claim_or_deal(text):
    """A warranty CLAIM, complaint, discount or custom request (needs a person), not a question about warranty terms."""
    if is_policy_question(text):
        return False
    return bool(re.search(r"\b(claim|replace\w*|refund|defect\w*|faulty|leak\w*|broken|damag\w*|not working|failed|complain\w*|"
                          r"return|discount|special price|negotiat\w*|custom|non.?catalogue|speak to|human|person|manager)\b", text, re.I))


def _working_day(d):
    return d.weekday() < 6  # Monday to Saturday


def _add_working_days(d, n):
    while n > 0:
        d += timedelta(days=1)
        if _working_day(d):
            n -= 1
    return d


def _day(d):
    return d.strftime("%a %d %b").replace(" 0", " ")


def dispatch_date(extra_working_days=0, now=None):
    now = now or datetime.now(IST)
    d = now.date()
    if not (_working_day(d) and now.hour < DISPATCH_CUTOFF_HOUR):
        d = _add_working_days(d, 1)
    return _add_working_days(d, extra_working_days)


def delivery_text(row, qty=None, city=None, now=None):
    """Dispatch and delivery dates from stock, restock time (internal), the 2 pm cut-off and Mon-Sat working days."""
    avail, restock = int(row["qty_available"]), int(row.get("restock_days_internal") or 10)
    need = qty or 1
    today_ship = dispatch_date(0, now)
    later_ship = dispatch_date(restock, now)

    def arrive(ship):
        chennai = f"Chennai {_day(_add_working_days(ship, 1))}"
        other = f"other cities {_day(_add_working_days(ship, 2))} to {_day(_add_working_days(ship, 5))}"
        return chennai if city == "chennai" else other if city == "other" else f"{chennai}; {other}"

    if avail >= need:
        return f"Dispatch {_day(today_ship)}; delivery {arrive(today_ship)}"
    if avail > 0:
        return (f"{avail} pcs dispatch {_day(today_ship)}, remaining {need - avail} pcs dispatch {_day(later_ship)}; "
                f"full delivery {arrive(later_ship)}")
    return f"Dispatch {_day(later_ship)}; delivery {arrive(later_ship)}"


def availability_text(row, qty=None):
    avail = int(row["qty_available"])
    if qty and qty > avail:
        return f"{avail} of {qty} available now" if avail else "Out of stock (made/ordered for you)"
    if avail == 0:
        return "Out of stock (made/ordered for you)"
    return "In stock" if avail >= 20 else f"Limited stock ({avail} available)"


def customer_city(text):
    t = text.lower()
    if "chennai" in t:
        return "chennai"
    if re.search(r"\b(bangalore|bengaluru|hyderabad|coimbatore|madurai|mumbai|pune|delhi|kochi|trichy|salem|outstation|outside chennai)\b", t):
        return "other"
    return None


def lookup_stock(part_no, qty=None, fields=None, text=""):
    """Only the fields the buyer asked about (plus product name and part number). Internal columns never leave here."""
    row = next((r for r in STOCK if r["part_no"] == part_no), None)
    if not row:
        return {"part_no": part_no, "found": False}
    fields = fields or list(DEFAULT_FIELDS)
    out = {"product_name": row["product_name"], "part_no": part_no, "found": True}
    if qty:
        out["qty_requested"] = qty
    price = int(row["unit_price_inr"])
    if "price" in fields:
        out["price"] = price
        if qty:
            out["total"] = price * qty
    if "availability" in fields:
        out["availability"] = availability_text(row, qty)
    if "delivery" in fields:
        out["delivery"] = delivery_text(row, qty, customer_city(text))
    if "warranty" in fields:
        out["warranty"] = row["warranty"]
    if "specs" in fields:
        out["specs"] = row["key_specs"]
    if "tax" in fields:
        out["tax"] = f"GST {row['gst_percent']}%, HSN {row['hsn_code']}"
    return out


# ---------- orders: only the buyer's own orders, only the details asked for
ORDER_FIELD_PATTERNS = {
    "status":   r"\b((?<!payment )(?<!delivery )status|where is|update|progress|stage|what happened|position)\b",
    "items":    r"\b(items?|products?|lines?|contents|what (did|have) (i|we) order\w*|ordered|quantit\w*|qty|how many)\b",
    "delivery": r"\b(deliver\w*|arriv\w*|reach\w*|eta|expected|when will|dispatch\w*|shipped|sent)\b",
    "tracking": r"\b(track\w*|lr|lorry receipt|transporter|courier|docket|consignment)\b",
    "invoice":  r"\b(invoice|bill)\b",
    "payment":  r"\b(payment|paid|pay|due|outstanding|overdue|balance|advance|credit)\b",
    "value":    r"\b(value|amount|total|how much|cost|worth|gst amount)\b",
    "po":       r"\b(po number|purchase order|our po|po no)\b",
}
ORDER_DEFAULT_FIELDS = ["status", "items", "delivery"]
ORDER_COLUMN_TITLES = {"status": "Status", "delivery": "Delivery", "tracking": "Transporter / LR no.", "invoice": "Invoice no.",
                       "payment": "Payment", "value": "Value (₹)", "po": "Your PO no."}


def order_fields(text):
    fields = [f for f, pat in ORDER_FIELD_PATTERNS.items() if re.search(pat, text, re.I)]
    if fields == ["status"]:  # "where is my order?" -> status plus delivery
        fields.append("delivery")
    return fields


def _date(iso):
    if not iso:
        return ""
    d = datetime.strptime(iso, "%Y-%m-%d")
    return d.strftime("%a %d %b %Y").replace(" 0", " ")


def _find_order(ref):
    ref = (ref or "").upper()
    return next((r for r in ORDERS if r["order_no"].upper() == ref or r["po_number"].upper() == ref), None)


def order_details(row, fields):
    """One order with only the asked details. Contact data never leaves here."""
    out = {"order_no": row["order_no"], "order_date": _date(row["order_date"])}
    if "status" in fields:
        out["status"] = row["status"]
    if "delivery" in fields:
        if row["delivered_on"]:
            out["delivery"] = f"Delivered on {_date(row['delivered_on'])} (dispatched {_date(row['dispatch_date'])})"
        elif row["status"] == "Partly dispatched":
            out["delivery"] = f"First part dispatched {_date(row['dispatch_date'])}; remaining items expected by {_date(row['expected_delivery'])}"
        elif row["expected_delivery"]:
            out["delivery"] = (f"Dispatched {_date(row['dispatch_date'])}; expected by {_date(row['expected_delivery'])}" if row["dispatch_date"]
                               else f"Expected by {_date(row['expected_delivery'])}")
        elif row["status"] == "Cancelled":
            out["delivery"] = "Order cancelled"
        else:
            out["delivery"] = "Not dispatched yet; dispatch date confirmed once the order is ready" + (
                " and the advance is received" if "advance" in row["status"].lower() else "")
    if "tracking" in fields:
        out["tracking"] = (f"{row['transporter']}, LR {row['lr_no']}" if row["lr_no"] else
                           row["transporter"] + " (no LR, local delivery)" if row["transporter"] else "Not dispatched yet")
    if "invoice" in fields:
        out["invoice"] = row["invoice_no"] or "Not invoiced yet"
    if "payment" in fields:
        out["payment"] = f"{row['payment_status']} ({row['payment_terms']})"
    if "value" in fields:
        out["value"] = f"{int(row['order_value_inr']):,} + GST {int(row['gst_inr']):,} = {int(row['total_inr']):,}"
    if "po" in fields:
        out["po"] = row["po_number"]
    if "items" in fields:
        out["items"] = [{"product_name": l["product_name"], "part_no": l["part_no"], "qty_ordered": int(l["qty_ordered"]),
                         "qty_dispatched": int(l["qty_dispatched"]), "line_status": l["line_status"]}
                        for l in ORDER_LINES if l["order_no"] == row["order_no"]]
    return out


def lookup_order(ref, gstin4, fields=None):
    """Order by order number or the customer's PO number, after checking the last 4 characters of the customer's GSTIN."""
    row = _find_order(ref)
    if not row or (gstin4 or "").upper() != CUSTOMERS[row["customer_id"]]["gstin_last4"].upper():
        return {"verified": False}
    return {"verified": True, **order_details(row, fields or ORDER_DEFAULT_FIELDS)}


LIST_FILTERS = {
    "open": lambda r: r["status"] not in ("Delivered", "Cancelled"),
    "payment_due": lambda r: r["payment_status"].startswith(("Due", "Overdue", "Advance pending")),
    "overdue": lambda r: r["payment_status"].startswith("Overdue"),
    "delivered": lambda r: r["status"] == "Delivered",
    "all": lambda r: True,
}


def list_filter(text):
    t = text.lower()
    if re.search(r"overdue", t):
        return "overdue"
    if re.search(r"\b(outstanding|due|unpaid|pending payment|payments?)\b", t):
        return "payment_due"
    if re.search(r"\b(open|pending|in transit|not (yet )?delivered|undelivered|running|active)\b", t):
        return "open"
    if re.search(r"\bdelivered\b", t):
        return "delivered"
    return "all"


def list_orders(gstin_full, flt="all", product_text="", limit=15):
    """All orders of the customer whose full GSTIN matches, newest first, filtered."""
    cust = next((c for c in CUSTOMERS.values() if c["gstin"].upper() == (gstin_full or "").upper()), None)
    if not cust:
        return {"verified": False}
    rows = [r for r in ORDERS if r["customer_id"] == cust["customer_id"] and LIST_FILTERS.get(flt, LIST_FILTERS["all"])(r)]
    parts = find_part_numbers(product_text) or find_products(product_text) if product_text else []
    if parts:  # "my orders for gear motors"
        with_part = {l["order_no"] for l in ORDER_LINES if l["part_no"] in parts}
        rows = [r for r in rows if r["order_no"] in with_part]
    rows.sort(key=lambda r: r["order_date"], reverse=True)
    return {"verified": True, "customer_name": cust["customer_name"], "filter": flt, "count": len(rows),
            "orders": [{"order_no": r["order_no"], "order_date": _date(r["order_date"]), "status": r["status"],
                        "total_inr": int(r["total_inr"]), "payment": r["payment_status"],
                        "delivery": (f"Delivered {_date(r['delivered_on'])}" if r["delivered_on"] else
                                     f"Expected {_date(r['expected_delivery'])}" if r["expected_delivery"] else "")}
                       for r in rows[:limit]]}


LIST_REQUEST_RE = re.compile(r"\b((my|our|all) (open |pending |recent |last |past |previous |running |delivered |\d+ )?(orders|order history|purchases)|"
                             r"order history|list (of )?(my |our |all )?orders|orders (we|i) (placed|have)|outstanding (payments?|invoices?|amount)|"
                             r"overdue|unpaid|pending payments?|what do (i|we) owe)\b", re.I)


# ---------------------------------------------------------------- 4. the language model (any OpenAI-compatible API)
def get_setting(name, default=None):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


# Providers tried in this order. Each needs only its key in secrets; models can be overridden in secrets too.
# Model names change often. Run `python check_keys.py` to test them and list the models your key can use.
PROVIDERS = {
    "groq":   {"key": "GROQ_API_KEY",   "base": "https://api.groq.com/openai/v1",
               "model": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b",
               "tokens_param": "max_completion_tokens", "temperature": True, "reasoning": "low", "timeout": 30},
    "gemini": {"key": "GEMINI_API_KEY", "base": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-3.5-flash-lite", "fast": "gemini-3.5-flash-lite",
               "tokens_param": "max_tokens", "temperature": True, "reasoning": None, "timeout": 45},
    "openai": {"key": "OPENAI_API_KEY", "base": "https://api.openai.com/v1",
               "model": "gpt-6-luna", "fast": "gpt-6-luna",
               "tokens_param": "max_completion_tokens", "temperature": False, "reasoning": "low", "timeout": 45},
}


class LLM:
    """Calls the first provider that works. If one fails (bad key, rate limit, outage, timeout), it moves to the next.
    If a provider rejects a setting (e.g. temperature, JSON mode, max_tokens), the call adapts and retries once per setting."""

    def __init__(self, only=None):
        from openai import OpenAI
        order = [p.strip().lower() for p in str(get_setting("PROVIDER_ORDER", "groq,gemini,openai")).split(",")]
        if only:
            order = [only]
        self.providers = []
        for name in order:
            cfg = PROVIDERS.get(name)
            key = get_setting(cfg["key"]) if cfg else None
            if not key:
                continue
            up = name.upper()
            self.providers.append({
                "name": name,
                "client": OpenAI(api_key=str(key).strip(), base_url=cfg["base"], timeout=cfg["timeout"], max_retries=0),
                "model": get_setting(f"{up}_MODEL", cfg["model"]),
                "fast": get_setting(f"{up}_FAST_MODEL", cfg["fast"]),
                "tokens_param": cfg["tokens_param"], "temperature": cfg["temperature"],
                "reasoning": cfg["reasoning"], "json_mode": True,
            })
        self.last_used = None   # "groq:openai/gpt-oss-120b" etc., shown in the answer
        self.failures = []      # what went wrong on the most recent call, shown in the JSON
        self.usage = []         # one record per model call (tokens, seconds), for cost and latency measurement
        self.wait_on_rate_limit = 0   # seconds the bot may wait when a provider says "too many requests" (0 = move on at once)
        self.waited = 0.0             # seconds spent waiting for rate limits since reset_usage()

    def reset_usage(self):
        self.usage = []
        self.waited = 0.0

    def usage_summary(self):
        """Totals for the calls since the last reset_usage(): tokens in/out, number of calls, time spent waiting on models."""
        u = self.usage
        return {"model_calls": len(u), "input_tokens": sum(x["input_tokens"] for x in u),
                "output_tokens": sum(x["output_tokens"] for x in u), "total_tokens": sum(x["total_tokens"] for x in u),
                "model_seconds": round(sum(x["seconds"] for x in u), 2), "failed_calls": sum(not x["ok"] for x in u),
                "rate_limit_wait_s": round(self.waited, 2),
                "errors": "; ".join(dict.fromkeys(x["error"] for x in u if x.get("error"))),
                "by_model": {m: sum(x["total_tokens"] for x in u if x["model"] == m) for m in dict.fromkeys(x["model"] for x in u)}}

    @property
    def online(self):
        return bool(self.providers)

    @property
    def model(self):
        return self.providers[0]["model"] if self.providers else None

    @property
    def fast_model(self):
        return self.providers[0]["fast"] if self.providers else None

    def _call(self, p, model, messages):
        """One provider, adapting to settings it rejects. Remembers the adaptation for later calls."""
        for _ in range(8):
            params = {"model": model, "messages": messages, p["tokens_param"]: 2000}
            if p["temperature"]:
                params["temperature"] = 0
            if p["json_mode"]:
                params["response_format"] = {"type": "json_object"}
            if p["reasoning"]:
                params["reasoning_effort"] = p["reasoning"]
            t0 = time.perf_counter()
            try:
                r = p["client"].chat.completions.create(**params)
                u = getattr(r, "usage", None)
                self.usage.append({"provider": p["name"], "model": model, "ok": True, "seconds": round(time.perf_counter() - t0, 2),
                                   "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                                   "output_tokens": getattr(u, "completion_tokens", 0) or 0,  # includes hidden reasoning tokens
                                   "total_tokens": getattr(u, "total_tokens", 0) or 0})
                return r.choices[0].message.content or ""
            except Exception as e:
                # a failed call uses no tokens (the provider refused it), so tokens are 0; the reason is kept
                self.usage.append({"provider": p["name"], "model": model, "ok": False, "seconds": round(time.perf_counter() - t0, 2),
                                   "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                                   "error": f"{p['name']} {type(e).__name__}"})
                if type(e).__name__ == "RateLimitError" and self.wait_on_rate_limit:
                    delay = _retry_after(e)
                    if self.waited + delay <= self.wait_on_rate_limit:
                        time.sleep(delay)  # free-tier limit (requests or tokens per minute): wait, then try the same call again
                        self.waited += delay
                        continue
                    raise
                if type(e).__name__ != "BadRequestError":
                    raise
                m = str(e).lower()
                if "max_tokens" in m and p["tokens_param"] == "max_tokens":
                    p["tokens_param"] = "max_completion_tokens"
                elif "max_completion_tokens" in m and p["tokens_param"] == "max_completion_tokens":
                    p["tokens_param"] = "max_tokens"
                elif "temperature" in m and p["temperature"]:
                    p["temperature"] = False
                elif "reasoning" in m and p["reasoning"]:
                    p["reasoning"] = None
                elif ("response_format" in m or "json" in m) and p["json_mode"]:
                    p["json_mode"] = False
                else:
                    raise
        raise RuntimeError("provider kept rejecting the request settings")

    def json(self, system, user, fast=False):
        self.failures = []
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for p in self.providers:
            model = p["fast"] if fast else p["model"]
            try:
                out = _parse_json(self._call(p, model, messages))
                if out:
                    self.last_used = f"{p['name']}:{model}"
                    return out
                self.failures.append(f"{p['name']}:{model} -> reply was not valid JSON")
            except Exception as e:
                self.failures.append(f"{p['name']}:{model} -> {type(e).__name__}: {str(e)[:160]}")
        raise RuntimeError("all providers failed: " + " | ".join(self.failures))


def _retry_after(e):
    """Seconds to wait after a 429 'too many requests': from the Retry-After header or the message, else 20 s."""
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return min(float(h) + 1, 65)
    except Exception:
        pass
    m = re.search(r"(?:try again in|retry in|retryDelay\W+)\s*(\d+(?:\.\d+)?)\s*(ms|s)?", str(e), re.I)
    if m:
        v = float(m.group(1)) / (1000 if (m.group(2) or "").lower() == "ms" else 1)
        return min(v + 1, 65)
    return 20.0


def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------- 5. intent classification
CLASSIFY_PROMPT = """You route customer messages for KMC, an industrial parts supplier (pneumatic cylinders, solenoid valves,
bearings, gear motors, couplings, hydraulic adapters). Return JSON only:
{"intent": one of ["greeting","product_question","buying","delivery","escalation","out_of_scope"],
 "reason": "<8 words>"}
Definitions:
- greeting: hello, thanks, bye, small talk with no question.
- product_question: specifications, selection, compatibility, installation, maintenance, troubleshooting, and questions
  about KMC's policies and terms (discount policy, minimum order, payment terms, returns, warranty terms, delivery terms).
- buying: wants to buy, asks price, stock, availability, lead time, or to place an order.
- delivery: status of an EXISTING order (order number, tracking, LR number, "where is my order").
  Delivery time for a product they want to buy ("when can you deliver 20 bearings?") is buying.
- escalation: a specific warranty CLAIM, replacement, refund, complaint, a request for a discount or special price on a
  deal, custom or non-catalogue product, operating beyond ratings, or asks for a human.
- out_of_scope: anything unrelated to KMC products, orders or KMC's own policies, or a request for another customer's details.
  A question about KMC's policies or terms is NEVER out_of_scope.
Examples: "what is the warranty on the 63 mm cylinder?" -> buying. "hi there" -> greeting. "what products do you sell?" -> product_question. "need 20 sealed 25 mm bearings" -> buying. "torque for G1/2 adapter?" -> product_question. "need 50 nos 6205-2RS, price?" -> buying.
"where is SO-24117" -> delivery. "what is the discount policy?" -> product_question. "give 15% discount on 200 bearings" -> escalation. "cylinder leaking, want free replacement" -> escalation."""

GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|hii+|good (morning|afternoon|evening)|namaste|vanakkam|thanks?( you)?|thank you|ok(ay)?|bye)\b[\s!.,]*\w{0,12}[\s!.]*$", re.I)


def keyword_intent(msg):
    m = msg.lower()
    if GREETING_RE.match(msg):
        return "greeting"
    if re.search(r"phone number|mobile number|email (id )?of|contact (details )?of|address of", m):
        return "out_of_scope"
    if is_claim_or_deal(msg):
        return "escalation"
    if find_order_no(msg) or re.search(r"\b(lr no|deliver(ed|y) status|where is my|when will .* arrive)", m) \
            or (re.search(r"\b(dispatch\w*|tracking|track)\b", m) and re.search(r"\b(my|our|order|consignment|shipment|parcel)\b", m)):
        return "delivery"  # "dispatch cut-off time?" alone is a policy question, not order tracking
    if re.search(r"\b(price|cost|rate|quote|quotation|stock|available|availability|lead time|buy|purchase|order \d|need \d+|\d+\s*(pcs|nos|pieces|units))", m):
        return "buying"
    return "product_question"


def team_for(msg):
    m = msg.lower()
    if re.search(r"discount|special price|quotation|quote|volume price|rate contract", m):
        return "handoff_sales"
    if re.search(r"warrant|replace|refund|defect|faulty|leak|credit note|damaged", m):
        return "handoff_quality"
    if re.search(r"custom|non.?catalogue|special|bar\b|rating|beyond|lifting|safety|bore", m):
        return "handoff_engineering"
    return "handoff_customer_service"


def classify(msg, llm):
    if GREETING_RE.match(msg):  # free fast path: no model call for "hi"
        return "greeting", "matched greeting pattern", "rule"
    if not llm.online:
        return keyword_intent(msg), "keyword rules (offline mode)", "rule"
    try:
        out = llm.json(CLASSIFY_PROMPT, msg, fast=True)
        intent = out.get("intent")
        if intent in INTENTS:
            return intent, out.get("reason", ""), f"model ({llm.last_used})"
    except Exception as e:
        return keyword_intent(msg), f"model unavailable ({type(e).__name__}); keyword rules used", "rule"
    return keyword_intent(msg), "model gave no valid label; keyword rules used", "rule"


# ---------------------------------------------------------------- 6. writing the answer
ANSWER_PROMPT = """You are the customer-service assistant of Kaveri Motion Components (KMC), Chennai, a B2B supplier of industrial
parts. Answer ONLY from the PRODUCT RANGE, MANUAL EXCERPTS and DATA given. Rules:
- Talk about products by their plain product NAME, as a sales engineer would ("sealed ball bearing, 25 mm bore",
  "5/2 solenoid valve, 24 V DC"). {code_rule}
- Questions about what we sell, our range or catalogue (DATA "catalogue": true): one short sentence, then a table.
  If DATA "with_prices" is true, columns | Product | Part no. | Price (₹, ex-GST) | with every item in PRODUCT RANGE.
  Otherwise | Product family | What we offer | Sizes / ratings |, one row per family. Never add the not-available sentence here.
- Only if the customer asks for a specific product, size or model that we clearly do not carry (not in PRODUCT RANGE
  or the excerpts, or DATA says "found": false or "not_available": true), include this sentence exactly: "{not_available}"
  Never invent a price or stock for such a product.
- Every technical number must come from the excerpts; cite it as (document name p.N). Never use outside knowledge for a number.
- If the excerpts do not contain the answer, say so and offer to pass it to an applications engineer.
- Stock, price and order facts are in DATA; restate them exactly, never change them.
- Never approve warranty claims, refunds, discounts or custom items; say which team will reply and how fast (from the excerpts).
- Refuse operation beyond published ratings and state the rating.
- FORMAT (except product lookups with DATA "products", which never get a table from you): use a markdown table whenever the answer lists 3 or more items or compares options or sizes (product range,
  specs across sizes, matching products, seal kits by bore, oil quantities by mounting position). Write one short sentence
  first, then the table: header row, a |---| separator row, one item per row, units in the header. Use a numbered list
  for steps (troubleshooting, how to claim). Use plain sentences for a single fact. Inside the JSON "answer" string,
  write line breaks as \n. Put the table INSIDE the "answer" string, never in another field.
- Never list our whole range or unrelated products unless DATA "catalogue" is true. If DATA has no "products" for a
  buying question, answer only what was asked or ask which product the customer means (use RECENT CONVERSATION).
- DATA "order" / "order_list" hold only the order details the customer asked about: restate those exactly, nothing else.
  If "verified" is false, say the details do not match our records and ask them to check. Never mention other customers.
- DATA "products" holds only the fields the customer asked about (fields_asked). Answer exactly those, nothing else:
  do not mention stock quantities, warranty, specs or delivery unless they appear in DATA. Delivery dates are in DATA
  "delivery": restate them, never calculate your own. Do NOT make a table of DATA "products": the app shows that table.
  Write 1-3 sentences: the answer for the best match (or, if several match, say how they differ and ask which one).
- Be concise: the answer first, then the key condition. Under 120 words, not counting tables. Prices are ex-works Chennai, GST 18% extra.
Return JSON only: {"answer": "<text for the customer>", "citations": [{"doc": "<document name exactly as in the excerpt header>", "page": <int>}],
 "next_action": one of ["none","ask_for_details","contact_by_email","handoff_sales","handoff_quality","handoff_engineering","handoff_customer_service"]}"""


CODE_RULES = {
    "brackets": "Give the part number once, in brackets after the name, only when the customer may order it, e.g. \"sealed ball bearing, 25 mm bore (6205-2RS)\". Never lead with a code.",
    "never": "Do not mention part numbers or codes at all; use names only.",
}


def contact_email():
    return str(get_setting("CONTACT_EMAIL", "kmcproducts@gmail.com"))


def not_available_sentence():
    return f"Currently we do not have this product. You can mail us directly at {contact_email()}."


def answer_prompt():
    return (ANSWER_PROMPT.replace("{code_rule}", CODE_RULES.get(str(get_setting("SHOW_PART_CODES", "brackets")).lower(), CODE_RULES["brackets"]))
            .replace("{not_available}", not_available_sentence()))


def strip_tables(text):
    """Remove markdown tables from an AI answer (used when the app already shows the product table).
    A table row holding a single sentence (e.g. "| Which one would you like? | | |") is kept as plain text."""
    out = []
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("|"):
            cells = [c.strip() for c in t.strip("|").split("|")]
            words = [c for c in cells if c and not set(c) <= set("-: ")]
            if len(words) == 1 and len(words[0].split()) >= 4:  # a sentence trapped in the table
                out.append(words[0])
            continue
        out.append(line[:-1].rstrip() + " (see the table below)." if line.rstrip().endswith(":") else line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text.rstrip(":").strip() + ("" if text.rstrip().endswith((".", "?", "!")) else ".") if text else text


def context_block(passages):
    return "\n\n".join(f"[{p['doc']} p.{p['page']}]\n{p['text']}" for p in passages) or "(no relevant excerpt found)"


def best_sentences(text, query, n=3):
    """The n sentences of a passage that share most words with the question, in their original order."""
    flat = re.sub(r"\s*\n\s*", " ", text)
    sents = [x.strip() for x in re.split(r"(?<=\.)\s+(?=[A-Z])", flat) if len(x.strip()) > 15]
    q = set(tokenize(query)) | {w for t in tokenize(query) for w in GLOSSARY.get(t, "").split()}
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & set(tokenize(sents[i]))))[:n]
    return " ".join(sents[i] for i in sorted(ranked))


def offline_answer(intent, passages, data, query=""):
    """No API key: show the best passage, plus any looked-up data, so the demo still works."""
    lines = []
    if data.get("not_available"):
        lines.append(not_available_sentence())
    if data.get("catalogue"):
        lines.append("Here is what we stock in Chennai:\n\n" + (price_table() if data.get("with_prices") else product_table()))
        return {"answer": "\n\n".join(lines), "citations": [], "next_action": "none"}
    if intent == "buying" and not data.get("products") and not data.get("not_available"):
        lines.append("Which product do you need? Tell me the product name (or part number) and the quantity, and I'll check stock and price.")
    if intent == "buying" and data.get("not_available") and any(p.get("found") for p in data.get("products", [])):
        lines.append("Closest items we stock are in the table below.")
    found = [p for p in data.get("products", []) if p.get("found")]
    if intent == "buying" and found:
        if data.get("fields_asked") == ["warranty"]:
            lines.append("Yes, we offer a warranty: 12 months from the invoice date on our products; wear parts such as seal kits "
                         "and coupling spiders are not covered. The table shows the products we were discussing.")
        elif len(found) > 1 and not data.get("not_available"):
            lines.append(f"We have {len(found)} products that match. Details are in the table below; tell me which one you need.")
        elif len(found) == 1:
            p = found[0]
            bits = [f"₹{p['price']:,} per piece + GST" if "price" in p else "", p.get("availability", ""), p.get("delivery", ""),
                    f"warranty {p['warranty']}" if "warranty" in p else "", p.get("specs", ""), p.get("tax", "")]
            lines.append(f"{p['product_name']}: " + "; ".join(b for b in bits if b) + ".")
    if intent == "delivery" and data.get("order"):
        o = data["order"]
        if not o["verified"]:
            lines.append("Those details don't match our records. Please check the order number and your GSTIN.")
        else:
            bits = [f"{ORDER_COLUMN_TITLES[k].lower()}: {o[k]}" for k in ("status", "delivery", "tracking", "invoice", "payment", "value", "po") if k in o]
            lines.append(f"Order {o['order_no']} (placed {o['order_date']})" + (": " + "; ".join(bits) + "." if bits else ".") +
                         (" The items are listed below." if "items" in o else ""))
    if intent == "delivery" and data.get("order_list"):
        L = data["order_list"]
        lines.append("That GSTIN doesn't match our customer records." if not L["verified"] else
                     f"{L['customer_name']}: {L['count']} order(s) found" + ("" if L["filter"] == "all" else f" ({L['filter'].replace('_', ' ')})") +
                     (". Showing the latest 15 below." if L["count"] > 15 else ". Details below." if L["count"] else "."))
    if intent in ("buying", "delivery"):
        pass  # product and order lookups are answered from the data files; no manual passage needed
    elif intent == "product_question" and not explicit_fields(query) and re.search(r"\b(sell|offer|products|catalog(ue)?|range|portfolio|what do you (have|stock))\b", query, re.I):
        lines.append("We supply these product families from Chennai stock:\n\n" + product_table())
    elif passages:
        p = passages[0]
        lines.append(f"From {p['doc']} page {p['page']}: {best_sentences(p['text'], query)}")
    return {"answer": "\n\n".join(lines) or "I couldn't find that in the manuals.",
            "citations": [{"doc": p["doc"], "page": p["page"]} for p in passages[:1]], "next_action": "none"}


def respond(msg, index, llm, history=None, shown_products=None, transcript=None):
    """transcript: the last few messages as [{"role": "customer"|"assistant", "text": ...}] so the AI sees the conversation."""
    """Main entry. Returns the structured response dict."""
    llm.last_used, llm.failures = None, []
    res = _respond(msg, index, llm, history, shown_products, transcript)
    found = [p["part_no"] for p in res["products"] if p.get("found")]
    res["context_products"] = found or find_part_numbers(msg) or find_products(msg) or None  # app remembers these
    res["answered_by"] = ("product tables from data files (no AI needed)" if res.get("topic") == "catalogue"
                          else llm.last_used or "no model (rules / manual passage)")
    res["provider_failures"] = llm.failures  # e.g. Groq rate-limited -> Gemini answered
    return res


# ---------------------------------------------------------------- 7. understanding the message (one AI step)
UNDERSTAND_PROMPT = """You read customer messages for KMC, a B2B industrial-parts supplier, and fill in a form. Return JSON only:
{"intent": "greeting" | "product_question" | "buying" | "delivery" | "escalation" | "out_of_scope",
 "scope": "on_screen" | "named" | "category" | "whole_range" | "none",
 "part_numbers": [part numbers from STOCK LIST that the customer means, only when scope is "named"],
 "category": one category name from STOCK LIST when scope is "category", else null,
 "fields": the details asked about, any of ["price","availability","delivery","warranty","specs","tax"],
 "quantity": number or null,
 "order_ref": the order number ("SO-24117") or the customer's PO number ("PO-018841") they ask about, or null,
 "gstin_last4": 4 characters or null,
 "order_fields": for order enquiries, the details asked, any of ["status","items","delivery","tracking","invoice","payment","value","po"],
 "order_list": true if they want a list of their orders (order history, open orders, outstanding payments),
 "order_filter": for lists, one of "open" | "payment_due" | "overdue" | "delivered" | "all",
 "not_carried": true if they ask for a kind of product that is not in STOCK LIST or PRODUCT FAMILIES,
 "general_question": true if it is a yes/no or policy question ("do you offer warranty?", "do you deliver to Pune?"),
 "reason": "<10 words>"}

Intent:
- greeting: hello / thanks / bye, nothing else.
- buying: price, stock, availability, delivery time, warranty, GST/HSN or specs of products we sell, wanting to order,
  or asking which products of a kind we have ("different gear motors you sell").
- product_question: how a product works, selection, installation, maintenance, troubleshooting, technical facts from manuals,
  and questions about KMC's policies and terms: discount policy, minimum order, payment terms, returns, warranty terms,
  delivery terms, GST ("what is the discount policy?", "do you accept returns?").
- delivery: anything about the customer's EXISTING orders: status, tracking/LR, items ordered, invoice, payment due,
  order value, order history / list of orders. Follow-ups about the order in RECENT CONVERSATION count too.
  Not the delivery time of a product they want to buy.
- escalation: a specific warranty CLAIM / replacement / refund / complaint, a request for a discount or special price
  on a deal ("give 15% off on 200 pcs"), custom or non-catalogue product, use beyond ratings, or asks for a person.
- out_of_scope: unrelated to KMC, or asks for another customer's details. Questions about KMC's own policies or terms
  are NEVER out_of_scope.

Scope (which products the message is about):
- "on_screen": a follow-up about the products already being discussed (PRODUCTS ON SCREEN), e.g. "warranty?",
  "do you offer warranty?", "price as well", "for both", "and delivery?", "which is cheaper?".
- "named": the message names particular products (by name, size or part number). Put their part numbers.
- "category": the message names a whole kind of product ("couplings", "gear motors") without picking one.
- "whole_range": everything we sell ("what do you sell?", "full price list", "warranty on all your products").
- "none": no products involved (order status, greeting, general policy with nothing on screen).
If the previous message was about a kind of product (e.g. "different filter-regulators") and this message is a
follow-up ("price as well", "warranty?"), use scope "category" with that category, not "whole_range".
Use "whole_range" ONLY when the customer clearly means everything we sell.
Fields: only what is actually asked. "do you offer warranty?" -> ["warranty"]. "different gear motors" -> [].
Spelling mistakes are common ("warantee", "delivary", "prise"): read them as meant."""


def llm_understand(msg, last, context, llm, convo="(start of conversation)"):
    stock = "\n".join(f"{r['part_no']} | {r['product_name']} | {r['category']}" for r in STOCK)
    families = ", ".join(f["product_family"] for f in FAMILIES)
    on_screen = "\n".join(f"{r['part_no']} | {r['product_name']}" for r in STOCK if r["part_no"] in (context or [])) or "(none)"
    user = (f"STOCK LIST (part no | name | category):\n{stock}\n\nPRODUCT FAMILIES: {families}\n\n"
            f"RECENT CONVERSATION:\n{convo}\n\nPRODUCTS ON SCREEN (from the last answer):\n{on_screen}\n\n"
            f"PREVIOUS CUSTOMER MESSAGE: {last or '(none)'}\n\nCUSTOMER MESSAGE: {msg}")
    # ROUTER_MODEL = "fast" in secrets runs this understanding step on the small open-weight model (e.g. gpt-oss-20b)
    return llm.json(UNDERSTAND_PROMPT, user, fast=str(get_setting("ROUTER_MODEL", "main")).lower() == "fast")


POLICY_WORDS = {"warranty": "warranty months invoice defects wear parts excluded", "delivery": "dispatch delivery transit working days",
                "price": "price GST minimum order", "tax": "GST price"}


def make_plan(msg, last, recent, context, index, llm, convo="(start of conversation)"):
    """Decide intent, products, fields and order details. AI first; keyword rules if no AI is available or it fails."""
    plan = {"intent": None, "scope": "none", "parts": [], "fields": [], "qty": None, "order_no": None, "gstin": None,
            "gstin_full": None, "order_fields": [], "order_list": False, "order_filter": "all",
            "not_carried": False, "general": False, "listing": False, "reason": "", "how": "rule"}
    if GREETING_RE.match(msg):
        plan.update(intent="greeting", reason="matched greeting pattern")
        return plan
    stocked = {r["part_no"] for r in STOCK}
    if llm.online:
        try:
            u = llm_understand(msg, last, context, llm, convo)
            intent = u.get("intent") if u.get("intent") in INTENTS else None
            if intent:
                scope = u.get("scope") if u.get("scope") in ("on_screen", "named", "category", "whole_range", "none") else "none"
                parts = []
                if scope == "on_screen":
                    parts = list(context or [])
                elif scope == "named":
                    parts = [p for p in (u.get("part_numbers") or []) if p in stocked] or find_part_numbers(msg) or find_products(msg)
                elif scope == "category":
                    cat = str(u.get("category") or "")
                    parts = [r["part_no"] for r in STOCK if r["category"].lower() == cat.lower()] or find_products_by_category(msg + " " + cat)
                fields = [f for f in (u.get("fields") or []) if f in FIELD_PATTERNS]
                qty = u.get("quantity") if isinstance(u.get("quantity"), int) and 0 < u.get("quantity") < 100000 else None
                plan.update(intent=intent, scope=scope, parts=parts, fields=fields, qty=qty or find_quantity(msg),
                            order_no=(find_order_no(str(u.get("order_ref") or "")) or find_po(str(u.get("order_ref") or ""))
                                      or find_order_no(msg) or find_po(msg) or find_order_no(recent) or find_po(recent)),
                            gstin=(str(u.get("gstin_last4") or "").upper()[:4] or None) or find_gstin4(msg) or find_gstin4(recent),
                            gstin_full=find_gstin_full(msg) or find_gstin_full(recent),
                            order_fields=[f for f in (u.get("order_fields") or []) if f in ORDER_FIELD_PATTERNS] or order_fields(msg),
                            order_list=bool(u.get("order_list")) or bool(LIST_REQUEST_RE.search(msg)),
                            order_filter=u.get("order_filter") if u.get("order_filter") in LIST_FILTERS else list_filter(msg),
                            not_carried=bool(u.get("not_carried")), general=bool(u.get("general_question")),
                            listing=(scope == "category" and not fields), reason=u.get("reason", ""), how=f"model ({llm.last_used})")
                # --- check the AI's form against the words the buyer actually typed
                other_customer = re.search(r"phone number|mobile number|email (id )?of|contact (details )?of|address of|"
                                           r"who placed|other customer", msg, re.I)
                if plan["intent"] in ("out_of_scope", "escalation") and is_policy_question(msg) and not other_customer \
                        and not find_order_no(msg):
                    plan["intent"] = "product_question"  # "what is the discount policy?": answer KMC's terms from the PDFs
                    plan["reason"] += " (policy question: answered from documents)"
                typed = explicit_fields(msg)
                if typed:  # details named in the message win over the AI's guess
                    plan["fields"] = typed
                elif scope == "named" and intent == "buying":
                    plan["fields"] = []  # nothing specific asked: price, availability and delivery
                stock_fields = {"price", "availability", "delivery", "warranty", "tax"}
                if plan["intent"] in ("buying", "product_question") and not plan["parts"] and scope in ("none", "named") \
                        and (set(plan["fields"]) & stock_fields or re.search(r"\bspec", msg, re.I)) and not plan["not_carried"]:
                    named = find_part_numbers(msg) or find_products(msg)
                    if named:  # "warranty on the 63 mm bore cylinder?": the product is named even if the AI missed it
                        plan.update(parts=named, scope="named")
                if plan["intent"] == "product_question" and plan["parts"] and (set(plan["fields"]) & stock_fields
                                                                                or re.search(r"\bspec", msg, re.I)):
                    plan["intent"] = "buying"  # price / stock / warranty of a product comes from the stock table
                typed_order = order_fields(msg)
                if typed_order:
                    plan["order_fields"] = typed_order
                elif plan["order_fields"] == ["status"]:
                    plan["order_fields"] = ["status", "delivery"]
                if plan["intent"] == "delivery" and not plan["order_no"] and parts:
                    plan["intent"] = "buying"  # delivery time of a product, not order tracking
                if plan["intent"] == "buying" and plan["general"] and not parts and scope == "none":
                    plan["intent"] = "product_question"  # "do you offer warranty?" with nothing on screen: answer the policy
                return plan
        except Exception as e:
            plan["reason"] = f"AI unavailable ({type(e).__name__}); keyword rules used. "
    return rules_plan(msg, last, recent, context, index, plan)


WHOLE_RANGE_RE = re.compile(r"\b(all (your |the |of your )?products|every (product|item)|whole|full|entire|catalog(ue)?|price ?list|everything|range)\b", re.I)


def rules_plan(msg, last, recent, context, index, plan):
    """Backup when no AI model is available: keyword rules."""
    intent = keyword_intent(msg)
    order_enquiry = ((LIST_REQUEST_RE.search(msg) or find_po(msg) or find_gstin_full(msg) or find_order_no(msg)) and not is_claim_or_deal(msg)) \
        or (intent != "delivery" and (find_order_no(last) or find_po(last)) and order_fields(msg)
            and not (find_part_numbers(msg) or find_products(msg)))   # "and the invoice number?" after an order question
    if order_enquiry or intent == "delivery":
        plan.update(intent="delivery", order_no=find_order_no(msg) or find_po(msg) or find_order_no(recent) or find_po(recent),
                    gstin=find_gstin4(msg) or find_gstin4(recent), gstin_full=find_gstin_full(msg) or find_gstin_full(recent),
                    order_fields=order_fields(msg), order_list=bool(LIST_REQUEST_RE.search(msg)), order_filter=list_filter(msg),
                    reason=plan["reason"] + "keyword rules (order enquiry)")
        return plan
    named = find_part_numbers(msg) or find_products(msg)
    fields = explicit_fields(msg)
    listing = bool(LISTING_RE.search(msg)) and bool(named) and not unrecognised_words(msg, index)
    scope = "named" if named else "none"
    if listing and intent in ("product_question", "buying"):
        intent = "buying"
    if intent == "delivery" and not find_order_no(msg) and (named or mentions_product(msg)):
        intent = "buying"
    if intent != "delivery" and find_order_no(last) and not mentions_product(msg) and re.search(FIELD_PATTERNS["delivery"] + r"|\bstatus\b", msg, re.I):
        intent = "delivery"
    if is_catalogue(msg, last) and not (fields and not WHOLE_RANGE_RE.search(msg)):
        scope = "whole_range"
        if intent == "escalation" and not is_claim_or_deal(msg):
            intent = "buying" if fields else "product_question"
    elif not named and context and set(fields) & {"price", "availability", "delivery", "warranty", "tax", "specs"} \
            and not is_claim_or_deal(msg) and not mentions_product(msg) and not find_order_no(msg + " " + last):
        intent, scope = "buying", "on_screen"   # "do you offer warranty?" / "price as well" about the products on screen
    elif named and intent in ("product_question", "escalation") and not is_claim_or_deal(msg) \
            and (set(fields) & {"price", "availability", "delivery", "warranty", "tax"} or re.search(r"\bspec", msg, re.I)):
        intent = "buying"
    if scope == "none" and fields and not named and not context and re.match(r"\s*(do|does|is|are|can|will)\b", msg, re.I):
        intent = "product_question"  # general policy question ("do you offer warranty?") with nothing being discussed
    parts = named if scope == "named" else (list(context) if scope == "on_screen" else [])
    if intent == "buying" and not parts and scope == "none" and not named:
        parts = list(context or [])
    plan.update(intent=intent, scope=scope, parts=parts, fields=fields, listing=listing,
                qty=find_quantity(msg) or (find_quantity(last) if scope == "on_screen" else None),
                order_no=find_order_no(msg) or find_po(msg) or find_order_no(recent) or find_po(recent),
                gstin=find_gstin4(msg) or find_gstin4(recent), gstin_full=find_gstin_full(msg) or find_gstin_full(recent),
                order_fields=order_fields(msg), order_list=bool(LIST_REQUEST_RE.search(msg)), order_filter=list_filter(msg),
                general=bool(re.match(r"\s*(do|does|is|are|can|will)\b", msg, re.I)),
                reason=plan["reason"] + "keyword rules")
    return plan


def history_messages():
    """How many earlier messages the AI sees: CHAT_HISTORY_EXCHANGES (default 10) questions + their 10 replies."""
    try:
        return 2 * max(0, int(get_setting("CHAT_HISTORY_EXCHANGES", 10)))
    except (TypeError, ValueError):
        return 20


def conversation_text(transcript, limit=None):
    limit = history_messages() if limit is None else limit
    lines = []
    for m in (transcript or [])[-limit:] if limit else []:
        # the full message goes to the model: no length cut, tables kept, so follow-ups have complete context
        lines.append(f"{m.get('role', 'customer')}: {str(m.get('text', '')).strip()}")
    return "\n".join(lines) or "(start of conversation)"


def _respond(msg, index, llm, history=None, shown_products=None, transcript=None):
    last = (history or [""])[-1]
    recent = " ".join(h for h in (history or [])[-2:])
    # what "it / them / both / as well" refers to: the previous question if it named products, else the products
    # of the last answer (passed in by the app), else the most recent earlier question that named products
    context = find_part_numbers(last) or find_products(last) or list(shown_products or [])
    if not context:
        for h in reversed((history or [])[-4:]):
            context = find_part_numbers(h) or find_products(h)
            if context:
                break
    convo = conversation_text(transcript)
    plan = make_plan(msg, last, recent, context, index, llm, convo)
    res = _route(msg, index, llm, plan, recent, convo)
    res["understood"] = {k: plan[k] for k in ("scope", "parts", "fields", "qty", "not_carried", "general",
                                               "order_no", "order_fields", "order_list", "order_filter")}  # shown in the JSON
    return res


def _route(msg, index, llm, plan, recent, convo="(start of conversation)"):
    if plan.get("order_list") and not (find_order_no(msg) or find_po(msg)):
        plan["order_no"] = None  # a list request: don't reuse an order number from earlier in the chat
    intent, reason, how = plan["intent"], plan["reason"], plan["how"]
    data, passages, next_action, needs_human = {}, [], "none", False

    if intent == "greeting":
        if re.match(r"\s*(thanks?|thank you|ok(ay)?|bye)", msg, re.I):
            return finish(msg, intent, reason, how, {"answer": "You're welcome. Message any time you need help with a KMC product or order.",
                                                      "citations": [], "next_action": "none"}, [], {}, False)
        return finish(msg, intent, reason, how, {
            "answer": "Hello! I'm KMC's parts assistant. I can answer technical questions from our product manuals, check stock and prices, "
                      "and track an order (send the order number and the last 4 characters of your GSTIN). How can I help?",
            "citations": [], "next_action": "none"}, [], {}, False)

    if intent == "out_of_scope":
        return finish(msg, intent, reason, how, {
            "answer": "I can only help with KMC products and your own orders. I can't share other customers' details or answer unrelated questions.",
            "citations": [], "next_action": "none"}, [], {}, False)

    if intent == "delivery":   # order enquiries: one order, or a list of the customer's orders
        if plan["order_list"] and not plan["order_no"]:
            full = plan["gstin_full"]
            if not full:
                return finish(msg, intent, reason, how, {
                    "answer": "I can list your orders. For your security, please send your company's full 15-character GSTIN.",
                    "citations": [], "next_action": "ask_for_details"}, [], {"order_list": None}, False)
            data["order_list"] = list_orders(full, plan["order_filter"], msg)
        else:
            order_no, gst = plan["order_no"], plan["gstin"]
            if not (order_no and gst):
                missing = [x for x, v in (("order number (or your PO number)", order_no),
                                          ("last 4 characters of your registered GSTIN", gst)) if not v]
                return finish(msg, intent, reason, how, {
                    "answer": f"I can check that. Please send the {' and the '.join(missing)}; we need both before sharing order details.",
                    "citations": [], "next_action": "ask_for_details"}, [], {"order": None}, False)
            data["order"] = lookup_order(order_no, gst, plan["order_fields"] or ORDER_DEFAULT_FIELDS)
            data["order_fields_asked"] = plan["order_fields"] or ORDER_DEFAULT_FIELDS
        passages = index.search("dispatch delivery transit LR transporter working days payment terms invoice", k=2)

    elif plan["scope"] == "whole_range" and intent in ("buying", "product_question"):
        data["catalogue"] = True
        data["with_prices"] = "price" in plan["fields"] or wants_prices(msg)
        passages = index.search("what we sell product families range popular stocked items", k=3)
        out = catalogue_answer(data["with_prices"], plan["fields"])  # built from the data files: complete, exact, instant
        return finish(msg, intent, reason, how, out, passages, data, False)

    elif intent == "buying":
        parts = plan["parts"]
        by_code = find_part_numbers(msg)
        missing_codes = unknown_codes(msg)
        strange = unrecognised_words(msg, index)
        fields = plan["fields"] or (["specs", "price"] if plan["listing"] else list(DEFAULT_FIELDS))
        data["fields_asked"] = fields
        data["products"] = [lookup_stock(p, plan["qty"], fields, msg) for p in parts] + [{"part_no": c, "found": False} for c in missing_codes]
        passages = index.search(msg + " " + " ".join(POLICY_WORDS.get(f, "") for f in fields), k=3)
        if plan["general"]:
            data["note"] = ("The customer asked a general yes/no question. Answer it first in general terms from the excerpts "
                            "(e.g. 'Yes, we give a 12-month warranty; wear parts are excluded'), then say the table shows the "
                            "products being discussed.")
        if missing_codes or plan["not_carried"] or (not parts and strange):
            data["not_available"] = True
        elif not parts and not llm.online and mentions_product(msg):
            data["not_available"] = True
        elif parts and not by_code and strange and plan["scope"] != "on_screen":
            data["not_available"] = True
            data["note"] = (f"The customer used words not in our range ({', '.join(strange)}). We do not carry that exact product. "
                            "Give the not-available sentence; you may mention the closest stocked item listed in DATA.")
        elif len(parts) > 1 and plan["scope"] in ("named", "category") and not by_code and not plan["listing"]:
            data["note"] = data.get("note", "") + " Several products match: say how they differ and ask which one the customer needs."
        if not parts and not data.get("not_available"):
            next_action = "ask_for_details"

    elif intent == "escalation":
        passages = index.search(msg + " warranty claim quality sales quotation applications engineering", k=4)
        needs_human = True

    else:  # product_question
        policy = " ".join(POLICY_WORDS.get(f, "") for f in (plan["fields"] or explicit_fields(msg)))
        passages = index.search(f"{msg} {recent} {policy} {policy}", k=4)
        asks_availability = re.search(r"\b(sell|have|stock|supply|offer|available|deal in|carry|provide|manufacture)\b", msg, re.I)
        strange = unrecognised_words(msg, index)
        if plan["not_carried"] or (asks_availability and (strange or unknown_codes(msg))):
            data["not_available"] = True
            data["note"] = "We do not carry this product. Give the not-available sentence."

    if llm.online:
        try:
            user = (f"RECENT CONVERSATION:\n{convo}\n\nINTENT: {intent}\n\nPRODUCT RANGE (stocked items, name [part no.] list price ex-GST):\n{product_range()}\n\nMANUAL EXCERPTS:\n{context_block(passages)}\n\nDATA (from ERP, exact):\n"
                    f"{json.dumps(data, ensure_ascii=False)}\n\nCUSTOMER MESSAGE:\n{msg}")
            out = llm.json(answer_prompt(), user)
        except Exception:
            out = offline_answer(intent, passages, data, msg)
            out["answer"] = "(All AI providers are unavailable right now, so here is the closest manual passage.)\n\n" + out["answer"]
    else:
        out = offline_answer(intent, passages, data, msg)

    if any(p.get("found") for p in data.get("products", [])) or (data.get("order") or {}).get("items") \
            or (data.get("order_list") or {}).get("orders"):  # the app shows the table: no second table
        out["answer"] = strip_tables(out.get("answer", "").replace("\\n", "\n"))
        out = {k: v for k, v in out.items() if k in ("answer", "citations", "next_action")}
    for k, v in list(out.items()):  # some models put the table in an extra field: keep it in the answer
        if k not in ("answer", "citations", "next_action") and isinstance(v, str) and "|" in v and "\n" in v:
            out["answer"] = out.get("answer", "").rstrip() + "\n\n" + v
    answer = out.get("answer", "")
    if data.get("not_available") and contact_email() not in answer:
        out["answer"] = (answer.rstrip() + "\n\n" + not_available_sentence()).strip()  # always give the email
    if not data.get("catalogue") and re.search(r"do not have this product|not (part of|in) our (product )?range|we (don't|do not) (sell|carry|supply|offer)",
                                                 out.get("answer", ""), re.I):
        data["not_available"] = True
        if contact_email() not in out["answer"]:
            out["answer"] = out["answer"].rstrip() + "\n\n" + not_available_sentence()
    if data.get("not_available"):
        out["next_action"] = "contact_by_email"
    if next_action != "none" and out.get("next_action", "none") == "none":
        out["next_action"] = next_action
    if needs_human and not str(out.get("next_action", "")).startswith("handoff"):
        out["next_action"] = team_for(msg)
    return finish(msg, intent, reason, how, out, passages, data, needs_human or str(out.get("next_action", "")).startswith("handoff"))


# ---------------------------------------------------------------- RESPONSE SHAPE
def finish(msg, intent, reason, how, out, passages, data, needs_human):
    cited = {(c.get("doc"), c.get("page")) for c in out.get("citations", []) if isinstance(c, dict)}
    valid = {(p["doc"], p["page"]) for p in passages}
    return {
        "intent": intent,
        "intent_reason": reason,
        "classified_by": how,  # "model" or "rule"
        "answer": out.get("answer", ""),
        "citations": [{"doc": d, "page": p, "verified": (d, p) in valid} for d, p in sorted(cited, key=str)],
        "products": data.get("products", []),
        "order": data.get("order"),
        "order_list": data.get("order_list"),
        "topic": "catalogue" if data.get("catalogue") else None,
        "product_available": (False if data.get("not_available") else (True if data.get("products") else None)),
        "contact_email": contact_email() if data.get("not_available") else None,
        "next_action": out.get("next_action", "none"),
        "needs_human": bool(needs_human),
        "ticket": f"KMC-T{uuid.uuid4().hex[:6].upper()}" if needs_human else None,
        "sources_searched": [{"doc": p["doc"], "page": p["page"], "score": p["score"]} for p in passages],
    }
