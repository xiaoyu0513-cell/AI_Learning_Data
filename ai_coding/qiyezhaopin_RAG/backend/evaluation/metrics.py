"""Retrieval metrics; empty gold queries are scored separately as abstention."""
def score_ids(retrieved, relevant, k):
    ids = list(dict.fromkeys(retrieved))[:k]
    gold = set(relevant)
    if not gold:
        return {"recall": None, "mrr": None, "hit": None, "abstention": float(not ids)}
    return {"recall": len(set(ids) & gold) / len(gold),
            "mrr": next((1 / (i+1) for i, node_id in enumerate(ids) if node_id in gold), 0.0),
            "hit": float(bool(set(ids) & gold)), "abstention": None}


def average(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return sum(values) / len(values) if values else None
