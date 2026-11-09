"""Generates the demo order book: data/customers.csv, data/orders.csv, data/order_lines.csv.
10 fictional customers, 8-12 orders each (100 orders), dated May-Nov 2026 relative to TODAY below.
Run again (python build_orders.py) to rebuild; the seed keeps the output identical. The three original demo orders
(SO-24117, SO-24130, SO-24142) are kept exactly as before.

Columns ending in _internal (contact names, phones, discounts) are never shown to buyers or sent to the AI."""
import csv, random
from datetime import date, timedelta
from pathlib import Path

TODAY = date(2026, 11, 7)
random.seed(42)
DATA = Path(__file__).parent / "data"
STOCK = list(csv.DictReader(open(DATA / "stock.csv", encoding="utf-8")))
PRICE = {r["part_no"]: int(r["unit_price_inr"]) for r in STOCK}
NAME = {r["part_no"]: r["product_name"] for r in STOCK}

#            id    name                                   city          state gstin_last4 terms              discount
CUSTOMERS = [("C01", "Sri Lakshmi Packaging Machines",    "Coimbatore", "33", "1ZK5", "30 days credit", 5),
             ("C02", "Anand Agro Equipments",             "Erode",      "33", "2PQ8", "100% advance",   0),
             ("C03", "Precision Pack Automation",         "Chennai",    "33", "9AB3", "100% advance",   0),
             ("C04", "Deccan Conveyor Systems",           "Hyderabad",  "36", "4CZ1", "45 days credit", 8),
             ("C05", "Malabar Food Processing Machinery", "Kochi",      "32", "7MZ3", "30 days credit", 3),
             ("C06", "Kaveri Textile Engineering",        "Tiruppur",   "33", "3TZ8", "30 days credit", 5),
             ("C07", "Bangalore Robotics and Fixtures",   "Bengaluru",  "29", "6RZ2", "45 days credit", 6),
             ("C08", "Godavari Process Equipment",        "Rajahmundry", "37", "5GZ9", "100% advance",  0),
             ("C09", "Pune Precision Tooling",            "Pune",       "27", "8PZ4", "30 days credit", 4),
             ("C10", "Madurai Plastic Moulders",          "Madurai",    "33", "2MZ6", "100% advance",   0)]
CONTACTS = ["R. Venkatesh", "S. Anand", "M. Fathima", "K. Srinivas Rao", "A. Thomas", "P. Karthikeyan", "N. Shetty",
            "V. Prasad", "S. Kulkarni", "J. Pandian"]
COUNTS = [11, 9, 8, 11, 8, 10, 12, 9, 10, 9]  # 97 generated + 3 original demo orders (C01-C03) = 100
QTY = {"Ball bearings": (10, 200), "Pneumatic cylinders": (1, 10), "Solenoid valves": (2, 20), "Filter-regulators": (1, 6),
       "Gear motors": (1, 3), "Couplings": (2, 12), "Hydraulic adapters": (10, 100)}
SPARE_QTY = (2, 25)
TRANSPORTERS = ["SwiftLine Transport", "Southern Roadways Cargo", "Coastal Express Logistics"]


def working(d):
    return d.weekday() < 6


def add_wd(d, n):
    while n > 0:
        d += timedelta(days=1)
        if working(d):
            n -= 1
    return d


def wd_between(a, b):
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        n += working(d)
    return n


def gstin(state, last4, i):
    pan = f"{'ABCDEFGHJK'[i]}{'LMNPQRSTUV'[i]}{'CFHPT'[i % 5]}{'KMRSV'[i % 5]}{'ABCDEFGHJK'[(i * 3) % 10]}{1000 + i * 731:04d}"[:9]
    return f"{state}{pan}{last4}"  # 2 + 9 + 4 = 15 characters


def pick_lines(cat_bias):
    n = random.choice([1, 1, 2, 2, 3, 4])
    pool = [r for r in STOCK if r["category"] in cat_bias] or STOCK
    parts = random.sample(pool, k=min(n, len(pool)))
    lines = []
    for r in parts:
        spare = any(w in r["product_name"].lower() for w in ("seal kit", "spare"))
        lo, hi = SPARE_QTY if spare else QTY[r["category"]]
        q = random.randint(lo, hi)
        if hi >= 50:
            q = max(lo, round(q, -1))
        lines.append((r["part_no"], q))
    return lines


# what each customer mostly buys
BIAS = {"C01": ["Pneumatic cylinders", "Solenoid valves", "Filter-regulators"], "C02": ["Gear motors", "Couplings", "Ball bearings"],
        "C03": ["Ball bearings", "Pneumatic cylinders"], "C04": ["Gear motors", "Couplings", "Ball bearings"],
        "C05": ["Pneumatic cylinders", "Solenoid valves", "Filter-regulators"], "C06": ["Ball bearings", "Gear motors"],
        "C07": ["Pneumatic cylinders", "Solenoid valves", "Filter-regulators", "Hydraulic adapters"],
        "C08": ["Hydraulic adapters", "Solenoid valves"], "C09": ["Hydraulic adapters", "Pneumatic cylinders", "Ball bearings"],
        "C10": ["Pneumatic cylinders", "Solenoid valves", "Filter-regulators"]}

# ---- customers
cust_rows = []
for i, (cid, name, city, state, last4, terms, disc) in enumerate(CUSTOMERS):
    cust_rows.append({"customer_id": cid, "customer_name": name, "city": city, "gstin": gstin(state, last4, i), "gstin_last4": last4,
                      "payment_terms": terms, "contact_name_internal": CONTACTS[i],
                      "contact_phone_internal": f"+91 9{8 - i % 3}{(40011223 + i * 1234567) % 100000000:08d}",
                      "discount_pct_internal": disc})

# ---- orders
orders, lines = [], []
plan = []
for (cid, *_), count in zip(CUSTOMERS, COUNTS):
    for _ in range(count):
        days_ago = random.randint(0, 10) if random.random() < 0.25 else random.randint(11, 185)  # a quarter are recent
        plan.append((cid, TODAY - timedelta(days=days_ago)))
plan.sort(key=lambda x: x[1])
KEEP = {"C01": ("SO-24117", date(2026, 10, 30)), "C02": ("SO-24130", date(2026, 10, 24)), "C03": ("SO-24142", date(2026, 11, 5))}
seq = 23900
cust = {c["customer_id"]: c for c in cust_rows}
inv = 1180
for cid, odate in plan:
    seq = max(seq + 1, 23935 + round((odate - date(2026, 5, 1)).days * 1.1))  # numbers rise with the date
    while f"SO-{seq}" in {v[0] for v in KEEP.values()}:
        seq += 1
    order_no = f"SO-{seq}"
    c = cust[cid]
    city, terms, disc = c["city"], c["payment_terms"], int(c["discount_pct_internal"])
    ol = pick_lines(BIAS[cid])
    age = wd_between(odate, TODAY)
    transit = 1 if city == "Chennai" else random.randint(2, 5)
    transporter = "KMC van" if city == "Chennai" else random.choice(TRANSPORTERS)
    r = random.random()
    o = {"order_no": order_no, "customer_id": cid, "po_number": f"PO-{cid[1:]}{random.randint(1000, 9999)}",
         "order_date": odate.isoformat(), "delivery_city": city, "payment_terms": terms}
    dispatch = delivered = expected = None
    if r < 0.03 and age > 10:
        status, pay = "Cancelled", "Not invoiced"
    elif age <= 1:
        status = "Confirmed, awaiting advance payment" if terms == "100% advance" else "Confirmed, being picked"
        pay = "Advance pending" if terms == "100% advance" else "Not invoiced yet"
    elif age <= 2:
        status, pay = "Ready for dispatch", ("Advance received" if terms == "100% advance" else "Not invoiced yet")
        expected = add_wd(add_wd(TODAY, 1), transit)
    else:
        dispatch = add_wd(odate, random.choice([0, 1, 1, 2]))
        arrive = add_wd(dispatch, transit)
        if arrive > TODAY:
            status, expected = "Dispatched, in transit", arrive
        else:
            status, delivered = "Delivered", arrive
        if random.random() < 0.3 and 3 <= age <= 20 and len(ol) > 1:  # recent multi-line orders: one line back-ordered
            status, delivered = "Partly dispatched", None
            expected = add_wd(add_wd(TODAY, 5), transit)  # the back-ordered line's delivery
    # money
    line_rows, value = [], 0
    for n, (part, q) in enumerate(ol, 1):
        unit = round(PRICE[part] * (100 - disc) / 100)
        shipped = q if status in ("Delivered", "Dispatched, in transit") else 0
        line_status = {"Cancelled": "Cancelled"}.get(status, "Delivered" if status == "Delivered" else
                                                     "In transit" if status == "Dispatched, in transit" else "Pending")
        if status == "Partly dispatched":
            shipped, line_status = (q, "Delivered") if n == 1 else (0, "Back-ordered, dispatch " + add_wd(TODAY, 5).strftime("%d %b %Y"))
        line_rows.append({"order_no": order_no, "line_no": n, "part_no": part, "product_name": NAME[part], "qty_ordered": q,
                          "qty_dispatched": shipped, "unit_price_inr": unit, "line_value_inr": unit * q, "line_status": line_status})
        value += unit * q
    invoice = None
    if dispatch:
        inv += 1
        invoice = f"INV/26-27/{inv:05d}"
        if terms == "100% advance":
            pay = "Paid (advance)"
        else:
            due = dispatch + timedelta(days=int(terms.split()[0]))
            pay = ("Paid" if due < TODAY - timedelta(days=5) and random.random() < 0.85 else
                   f"Overdue since {due.strftime('%d %b %Y')}" if due < TODAY else f"Due on {due.strftime('%d %b %Y')}")
    o.update({"status": status, "order_value_inr": value, "gst_inr": round(value * 0.18), "total_inr": value + round(value * 0.18),
              "invoice_no": invoice or "", "payment_status": pay, "dispatch_date": dispatch.isoformat() if dispatch else "",
              "transporter": transporter if dispatch else "", "lr_no": (str(random.randint(7000000, 7999999)) if dispatch and transporter != "KMC van" else ""),
              "expected_delivery": expected.isoformat() if expected else "", "delivered_on": delivered.isoformat() if delivered else ""})
    orders.append(o)
    lines += line_rows

# ---- put back the three original demo orders exactly as the report and earlier tests use them
def demo(order_no, cid, odate, status, lines_spec, terms_pay, dispatch, transporter, lr, expected, delivered, po):
    c = cust[cid]
    disc = int(c["discount_pct_internal"])
    rows, value = [], 0
    for n, (part, q, shipped, ls) in enumerate(lines_spec, 1):
        unit = round(PRICE[part] * (100 - disc) / 100)
        rows.append({"order_no": order_no, "line_no": n, "part_no": part, "product_name": NAME[part], "qty_ordered": q,
                     "qty_dispatched": shipped, "unit_price_inr": unit, "line_value_inr": unit * q, "line_status": ls})
        value += unit * q
    return ({"order_no": order_no, "customer_id": cid, "po_number": po, "order_date": odate, "delivery_city": c["city"],
             "payment_terms": c["payment_terms"], "status": status, "order_value_inr": value, "gst_inr": round(value * 0.18),
             "total_inr": value + round(value * 0.18), "invoice_no": terms_pay[0], "payment_status": terms_pay[1],
             "dispatch_date": dispatch, "transporter": transporter, "lr_no": lr, "expected_delivery": expected, "delivered_on": delivered}, rows)

demos = [demo("SO-24117", "C01", "2026-10-30", "Dispatched, in transit",
              [("PC15-63-200", 2, 2, "In transit"), ("SV52-S-24D", 4, 4, "In transit")], ("INV/26-27/01301", "Due on 02 Dec 2026"),
              "2026-11-03", "SwiftLine Transport", "7781452", "2026-11-07", "", "PO-018841"),
         demo("SO-24130", "C02", "2026-10-24", "Partly dispatched",
              [("GM-071-0.75-i20-M1", 1, 0, "Awaiting assembly, dispatch 18 Nov 2026"), ("JC-095", 4, 4, "Delivered")],
              ("INV/26-27/01288", "Paid (advance)"), "2026-10-29", "SwiftLine Transport", "7779910", "2026-11-18", "", "PO-023302"),
         demo("SO-24142", "C03", "2026-11-05", "Confirmed, awaiting advance payment",
              [("6205-2RS", 50, 0, "Pending")], ("", "Advance pending (proforma PI-8812)"), "", "", "", "", "", "PO-037120")]
for o, rows in demos:
    orders.append(o)
    lines += rows
orders.sort(key=lambda o: o["order_no"])

with open(DATA / "customers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cust_rows[0]))
    w.writeheader(); w.writerows(cust_rows)
with open(DATA / "orders.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(orders[0]))
    w.writeheader(); w.writerows(orders)
with open(DATA / "order_lines.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(lines[0]))
    w.writeheader(); w.writerows(lines)

from collections import Counter
print(len(cust_rows), "customers,", len(orders), "orders,", len(lines), "order lines")
print(Counter(o["customer_id"] for o in orders))
print(Counter(o["status"] for o in orders))
