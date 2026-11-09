"""Quick check without the UI:  python test_bot.py
Runs sample customer messages through the full pipeline and prints intent, next action and answer."""
from engine import LLM, ROOT, Index, load_pdfs, respond

index = Index(load_pdfs(sorted((ROOT / "docs").glob("*.pdf"))))
llm = LLM()
print(f"{len(index.chunks)} passages indexed · model {'online: ' + llm.model if llm.online else 'OFFLINE (no API key in secrets)'}")
print("Understanding: " + ("AI form-filling step (keyword rules only as backup)" if llm.online else "keyword rules (backup mode)") + "\n")
SAMPLES = [
    ("hello", "greeting"),
    ("what are different products you sell", "product_question"),
    ("What is the pushing force of a 63 mm bore pneumatic cylinder at 6 bar?", "product_question"),
    ("Need 20 pcs sealed 25 mm bearings with C3 clearance, price and stock?", "buying"),
    ("Price of a 24 V DC solenoid valve?", "buying"),
    ("Need 10 pcs 6206-2RS, price?", "buying"),          # not stocked -> "we do not have this product" + email
    ("Price of a servo motor 2 kW?", "buying"),          # we don't sell servo motors -> email
    ("What is the extend force of a PC15 cylinder with 63 mm bore at 6 bar?", "product_question"),
    ("SV52 valve is not switching when energised. What should we check?", "product_question"),
    ("Do you have 20 pcs of 6205-2RS C3 in stock?", "buying"),
    ("Where is my order SO-24130?", "delivery"),
    ("What's the status of order SO-24117? Our GSTIN ends 1ZK5.", "delivery"),
    ("Our cylinders leak after 2 months, we want free replacements", "escalation"),
    ("Can you give 18% discount on 200 pcs of 6205-2RS?", "escalation"),
    ("Give me the phone number of the person who placed SO-24117", "out_of_scope"),
]
ok = 0
for msg, expected in SAMPLES:
    r = respond(msg, index, llm)
    ok += r["intent"] == expected
    print(f"{'PASS' if r['intent'] == expected else 'FAIL'}  [{r['intent']}] -> {r['next_action']}\n  Q: {msg}\n  A: {r['answer'][:220]}\n")
print(f"Intent correct: {ok}/{len(SAMPLES)}")

# "we do not have this product" must appear only when we really don't carry it
print("\nAvailability checks")
AVAIL = [  # (message, earlier user message, should the bot say "not available"?)
    ("list down the price as well", "what products do you sell", False),
    ("give me your price list", "", False),
    ("Price of a 24 V DC solenoid valve?", "", False),
    ("Do you sell vacuum pumps?", "", True),
    ("Price of a servo motor 2 kW?", "", True),
    ("Need 10 pcs 6206-2RS, price?", "", True),
]
ok2 = 0
for msg, prev, expect in AVAIL:
    r = respond(msg, index, llm, [prev] if prev else [])
    said = r["product_available"] is False
    ok2 += said == expect
    print(f"{'PASS' if said == expect else 'FAIL'}  not-available={said}  Q: {msg}")
print(f"Availability correct: {ok2}/{len(AVAIL)}")

# only the asked columns are returned, and internal columns never are
print("\nField checks")
FIELDS = [
    ("i want to order Sealed ball bearing, whats the price and when will it be delivered", {"price", "delivery"}),
    ("warranty on the 63 mm bore pneumatic cylinder?", {"warranty"}),
    ("gst and hsn for jaw coupling size 95", {"tax"}),
    ("need 20 pcs sealed 25 mm bearings with C3 clearance", {"price", "total", "availability", "delivery"}),
    ("whats the warantee period for Sealed ball bearing, 25 mm bore (25 x 52 x 15 mm)", {"warranty"}),  # no qty from "25 x 52"
]
ok3 = 0
for msg, expect in FIELDS:
    r = respond(msg, index, llm)
    got = set().union(*[set(p) - {"found", "product_name", "part_no", "qty_requested"} for p in r["products"]]) if r["products"] else set()
    leaked = any(k.endswith("_internal") for p in r["products"] for k in p)
    good = got == expect and not leaked
    ok3 += good
    print(f"{'PASS' if good else 'FAIL'}  columns={sorted(got)}  Q: {msg}")
print(f"Field selection correct: {ok3}/{len(FIELDS)}")
r = respond("whats the warantee period for Sealed ball bearing, 25 mm bore (25 x 52 x 15 mm)", index, llm)
one = [p["part_no"] for p in r["products"]] == ["6205-2RS"]
print(f"{'PASS' if one else 'FAIL'}  exact product name gives only that product: {[p['part_no'] for p in r['products']]}")

# follow-ups refer to the previous question
print("\nFollow-up checks")
FOLLOW = [
    ("give me the price as well as warranty", "what are different Ball bearings you sell", {"6205-2RS", "6205-2RS-C3", "6204-ZZ", "6305-2RS"}),
    ("and the delivery time?", "need 20 pcs sealed 25 mm bearings with C3 clearance", {"6205-2RS-C3"}),
    ("give me the gurantee period as well", "what are the prices of different Couplings", {"JC-095", "JC-SP-095-NBR"}),  # typo on purpose
]
ok4 = 0
for msg, prev, expect in FOLLOW:
    r = respond(msg, index, llm, [prev])
    got = {p["part_no"] for p in r["products"]}
    ok4 += got == expect and r.get("topic") != "catalogue"
    print(f"{'PASS' if got == expect else 'FAIL'}  products={sorted(got)}  Q: {msg}  (after: {prev})")
print(f"Follow-ups correct: {ok4}/{len(FOLLOW)}")

# chained follow-ups: "both" / "them" mean the products in the last table (the app passes them in)
print("\nConversation checks")
CONVS = [
    (["what couplings do you sell?", "give me the prices", "what is the warantee period for both"], {"JC-095", "JC-SP-095-NBR"}, "warranty"),
    (["i want to buy Pneumatic cylinders, what is the price", "and the delivery?", "warranty for them?"], {"PC15-63-200", "PC15-50-100"}, "warranty"),
    # topic switch: the latest product (couplings) must win over the earlier one (valves)
    (["different Solenoid valves products", "give me the warantee period as well", "what are different Couplings you sell",
      "give me the price as well"], {"JC-095", "JC-SP-095-NBR"}, "price"),
    # a general yes/no follow-up stays on the products being discussed, not the whole range
    (["different Gear motors you sell", "do you offer warantee"], {"GM-090-1.5-i20-M1", "GM-071-0.75-i20-M1"}, "warranty"),
    (["what are different Filter-regulators you sell", "give me price as well"], {"FR40-G12-40", "FR40-G12-05", "FR40-FE-40"}, "price"),  # spare element is in the range, like the coupling spider
]
ok5 = 0
for turns, expect, field in CONVS:
    hist, shown = [], []
    for t in turns:
        r = respond(t, index, llm, hist, shown)
        shown = r.get("context_products") or shown  # what the app remembers between turns
        hist.append(t)
    got = {p["part_no"] for p in r["products"]}
    good = got == expect and all(field in p for p in r["products"])
    ok5 += good
    print(f"{'PASS' if good else 'FAIL'}  {' -> '.join(turns)}  => {sorted(got)}")
print(f"Conversations correct: {ok5}/{len(CONVS)}")

# orders: right order, GSTIN check, only the asked details, never another customer's data
print("\nOrder checks")
from engine import CUSTOMERS
G = {c["customer_id"]: c["gstin"] for c in CUSTOMERS.values()}
ORDER_TESTS = [
    ("Where is order SO-24117? GSTIN ends 1ZK5", lambda r: r["order"]["verified"] and set(r["order"]) >= {"status", "delivery"} and "invoice" not in r["order"]),
    ("What's the invoice number and payment status of SO-24117, GSTIN 1ZK5?", lambda r: set(r["order"]) - {"verified", "order_no", "order_date"} == {"invoice", "payment"}),
    ("tracking details for PO-018841, our gstin ends 1ZK5", lambda r: r["order"]["order_no"] == "SO-24117" and "tracking" in r["order"]),
    ("what items are in order SO-24130? gstin ends with 2PQ8", lambda r: len(r["order"]["items"]) == 2),
    ("status of SO-24117, GSTIN ends 9AB3", lambda r: r["order"]["verified"] is False),              # wrong customer's GSTIN
    ("Where is my order SO-24130?", lambda r: r["order"] is None and "GSTIN" in r["answer"]),          # asks for GSTIN first
    ("show our open orders", lambda r: "15-character GSTIN" in r["answer"]),                          # list needs full GSTIN
    (f"Show our open orders, GSTIN {G['C01']}", lambda r: r["order_list"]["verified"] and all(o["status"] not in ("Delivered", "Cancelled") for o in r["order_list"]["orders"])),
    (f"list all our orders for gear motors, GSTIN {G['C02']}", lambda r: r["order_list"]["count"] >= 1),
]
ok6 = 0
for msg, check in ORDER_TESTS:
    r = respond(msg, index, llm)
    try:
        good = bool(check(r))
    except Exception:
        good = False
    leaked = any(k in str(r) for k in ("contact_name_internal", "contact_phone_internal", "Venkatesh", "+91 98"))
    good = good and not leaked
    ok6 += good
    print(f"{'PASS' if good else 'FAIL'}  [{r['intent']}] {msg}")
r = respond(f"Show our open orders, GSTIN {G['C01']}", index, llm, ["what items are in order SO-24130? gstin ends with 2PQ8"])
good = bool(r["order_list"] and r["order_list"]["verified"])
ok6 += good
print(f"{'PASS' if good else 'FAIL'}  list request after a single-order question uses the list, not the old order")
print(f"Order checks correct: {ok6}/{len(ORDER_TESTS) + 1}")

print("\nPolicy questions (answered from the documents, not refused or handed off)")
POLICY = [("what is the discount policy", "product_question"), ("do you accept returns?", "product_question"),
          ("what are your payment terms", "product_question"), ("is there a minimum order value?", "product_question"),
          ("Can you give 18% discount on 200 pcs of 6205-2RS?", "escalation"), ("Our cylinders leak after 2 months, we want free replacements", "escalation")]
ok7 = 0
for msg, want in POLICY:
    r = respond(msg, index, llm)
    good = r["intent"] == want
    ok7 += good
    print(f"{'PASS' if good else 'FAIL'}  [{r['intent']}] {msg}\n  A: {r['answer'][:160]}")
print(f"Policy questions correct: {ok7}/{len(POLICY)}")
