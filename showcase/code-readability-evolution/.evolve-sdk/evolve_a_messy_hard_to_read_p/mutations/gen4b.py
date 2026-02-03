"""Summarize financial transactions with single-pass aggregation and minimal branching."""
import heapq
from collections import defaultdict
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_COUNT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions into totals, customer rankings, and category breakdowns.

    Processes all transactions in a single pass using signed amounts to unify
    sale and refund handling, then selects top-k customers via heapq.

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
    customer_totals: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_orders: dict[str, int] = defaultdict(int)

    for txn in transactions:
        signed_amount = txn["amount"] if txn["type"] == "sale" else -txn["amount"]
        customer_totals[txn["customer"]] += signed_amount
        total += signed_amount

        if txn["type"] == "sale":
            category_revenue[txn["category"]] += txn["amount"]
            category_orders[txn["category"]] += 1

    average = total / count if count else 0.0

    top_customers = heapq.nlargest(
        TOP_CUSTOMER_COUNT, customer_totals.items(), key=lambda pair: pair[1]
    )

    categories = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_orders[name],
            "avg": category_revenue[name] / category_orders[name],
        }
        for name in category_revenue
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
