import json

docs = json.load(open("data/ocr_report.json"))["documents"]
print(f"{'set':<34} {'num%':>7} {'expected':>10} {'missing':>8}  verdict")
for d in sorted(docs, key=lambda x: x["numeric_occurrence_accuracy"]):
    n = d["numeric_occurrences"]
    miss = n - round(n * d["numeric_occurrence_accuracy"] / 100)
    print(f"{d['document'][:33]:<34} {d['numeric_occurrence_accuracy']:>6.1f}% "
          f"{n:>10} {miss:>8}  {d['verdict']}")