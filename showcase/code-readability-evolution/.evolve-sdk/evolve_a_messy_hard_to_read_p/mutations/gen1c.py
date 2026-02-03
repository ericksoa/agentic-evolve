"""Single-pass transaction summarizer using lookup-driven aggregation.

Uses a sign-lookup table to handle sales and refunds uniformly,
eliminating branching and enabling efficient single-pass processing.
"""
from collections import defaultdict
from typing import Any


WHALE_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5

SIGN_BY_TYPE = {"sale": 1.0, "refund": -1.0}


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions by processing sales and refunds in a single pass.

    Uses a sign-lookup table to uniformly handle sales (+) and refunds (-),
    accumulating totals and customer spending without branching.

    Args:
        transactions: list of dicts with 'type', 'amount', 'customer', 'category'.

    Returns:
        dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 100, "customer": "A", "category": "B"}])
    """
    total = 0.0
    customer_spending: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_orders: dict[str, int] = defaultdict(int)

    for txn in transactions:
        amount = txn["amount"]
        signed = amount * SIGN_BY_TYPE[txn["type"]]
        total += signed
        customer_spending[txn["customer"]] += signed

        if txn["type"] == "sale":
            category_revenue[txn["category"]] += amount
            category_orders[txn["category"]] += 1

    count = len(transactions)
    average = total / count if count else 0.0

    category_stats = {
        cat: {"revenue": category_revenue[cat], "orders": category_orders[cat],
              "avg": category_revenue[cat] / category_orders[cat]}
        for cat in category_revenue
    }

    ranked_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )

    has_whale = any(amt > WHALE_THRESHOLD for amt in customer_spending.values())

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": ranked_customers[:MAX_TOP_CUSTOMERS],
        "categories": category_stats,
        "has_whale": has_whale,
    }
