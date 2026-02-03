"""Summarize financial transactions with minimal nesting for clarity."""
import heapq
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_COUNT = 5
SIGN_BY_TYPE = {"sale": 1.0, "refund": -1.0}


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions into totals, customer rankings, and category breakdowns.

    Uses a sign-lookup table and setdefault to minimize branching, then heapq
    for efficient top-k customer selection.

    Args:
        transactions: each dict has 'type', 'amount', 'customer', 'category'.

    Returns:
        Summary with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([])
        {'total': 0.0, 'count': 0, 'avg': 0.0, 'top_customers': [], 'categories': {}, 'has_whale': False}
    """
    total = 0.0
    count = len(transactions)
    customer_totals: dict[str, float] = {}
    category_data: dict[str, list] = {}

    for txn in transactions:
        sign = SIGN_BY_TYPE[txn["type"]]
        signed_amount = sign * txn["amount"]
        total += signed_amount
        customer = txn["customer"]
        customer_totals[customer] = customer_totals.get(customer, 0.0) + signed_amount
        if sign > 0:
            entry = category_data.setdefault(txn["category"], [0.0, 0])
            entry[0] += txn["amount"]
            entry[1] += 1

    average = total / count if count else 0.0

    top_customers = heapq.nlargest(
        TOP_CUSTOMER_COUNT, customer_totals.items(), key=lambda pair: pair[1]
    )

    categories = {
        name: {"revenue": data[0], "orders": data[1], "avg": data[0] / data[1]}
        for name, data in category_data.items()
    }

    has_whale = any(spent > WHALE_THRESHOLD for spent in customer_totals.values())

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
