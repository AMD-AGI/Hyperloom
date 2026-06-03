import json, os
d = json.load(open(os.path.join(os.path.dirname(__file__), "docstring_inventory.json"), encoding="utf-8"))
rs = sorted(d["reports"], key=lambda r: len(r["missing"]), reverse=True)
for r in rs:
    print(f"{len(r['missing']):4d}  {r['path']}")
print("FILES:", len(rs), "TOTAL:", sum(len(r['missing']) for r in rs))
