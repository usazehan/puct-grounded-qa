"""Do 'requested' and 'approved' retrieve different chunks?

Both questions name return on equity, both are answered in item 792, and the
figures are 10.4% and 9.4%. If retrieval returns the same ranking for both,
nothing downstream can tell them apart -- the chunk will pass span and numeric
verification either way, because the figure it contains is genuinely in it.
"""

import psycopg
from sentence_transformers import SentenceTransformer

from puctqa.retrieve import search

DSN = "postgresql://puctqa:puctqa@localhost:5432/puctqa"
QUESTIONS = {
    "q001 requested": ("What return on equity did CenterPoint request in this rate case?", 2),
    "q002 approved": ("What return on equity was approved in the Final Order?", 11),
    "q011 ALJs": ("What return on equity did the SOAH administrative law judges recommend?", 2),
}

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

with psycopg.connect(DSN) as conn, conn.cursor() as cur:
    for label, (question, want_page) in QUESTIONS.items():
        embedding = model.encode(question, normalize_embeddings=True).tolist()
        hits = search(cur, question, embedding=embedding, limit=5)
        print(f"=== {label}  (expected 792 p{want_page}) ===")
        for rank, h in enumerate(hits, 1):
            mark = "*" if h.document.startswith("49421_792") and h.page_start == want_page else " "
            snippet = " ".join(h.text.split())[:90]
            print(f" {mark}{rank}. {h.document[-11:-4]} p{h.page_start:<3} {snippet}")
        print()