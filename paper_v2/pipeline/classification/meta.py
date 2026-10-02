#!/usr/bin/env python3
"""BraTS-Africa per-subject centre / scanner / category from the TCIA data-info xlsx
(parsed with the standard library; openpyxl is not installed) -> africa_meta.csv."""
import zipfile, xml.etree.ElementTree as ET, pandas as pd
P = "/mnt/data1/kamil_research/brats_africa/BraTS-Africa_TCIA_datainfo_v2.xlsx"
X = "/mnt/data1/kamil_research/experiments/external_africa"; D = "/mnt/data1/kamil_research/experiments/classification_africa"
z = zipfile.ZipFile(P); ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}; M = "{%s}" % ns["m"]
ss = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
def col(ref): return "".join(ch for ch in ref if ch.isalpha())
rows = []
for sh in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
    data = []
    for r in ET.fromstring(z.read(sh)).iter(M + "row"):
        d = {}
        for c in r.findall("m:c", ns):
            v = c.find("m:v", ns); x = v.text if v is not None else ""
            if c.get("t") == "s" and x: x = ss[int(x)]
            d[col(c.get("r"))] = x
        data.append(d)
    head = data[0]
    for d in data[1:]:
        rec = {head.get(k, k): v for k, v in d.items()}
        sid = str(rec.get("Subject ID (0x)", "")).strip()
        if sid.isdigit(): rows.append(dict(subject=int(sid), center=rec.get("Center", "") or "NA",
                                           scanner=rec.get("Scanner Information", "") or "NA",
                                           category=rec.get("Neoplasm Category", "glioma" if "sheet1" in sh else "") or "NA",
                                           sheet=sh[-10:-4]))
m = pd.DataFrame(rows).drop_duplicates("subject")
c = pd.read_csv(f"{X}/cases.csv")[["case", "subject", "group"]]
out = c.merge(m, on="subject", how="left"); out.to_csv(f"{D}/africa_meta.csv", index=False)
print(out.groupby(["group"]).center.value_counts(dropna=False).unstack(0).fillna(0).astype(int))
print("unmatched:", int(out.center.isna().sum()), "| sheet vs group mismatch:",
      int(((out.sheet == "sheet1") & (out.group != "glioma")).sum() + ((out.sheet == "sheet2") & (out.group == "glioma")).sum()))
print(out[out.group != "glioma"].category.value_counts().to_dict())
