"""Single-pass transaction summarizer using defaultdict and heapq for top-k."""
from collections import defaultdict
import heapq
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_COUNT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions into totals, customer rankings, and category breakdowns.

    Performs a single pass over the data using defaultdict for cleaner accumulation,
    then uses heapq for efficient top-k customer selection.

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
        amount = txn["amount"]
        customer = txn["customer"]

        if txn["type"] == "sale":
            total += amount
            customer_totals[customer] += amount
            category = txn["category"]
            category_revenue[category] += amount
            category_orders[category] += 1
        else:
            total -= amount
            customer_totals[customer] -= amount

    average = total / count if count else 0.0

    top_customers = heapq.nlargest(
        TOP_CUSTOMER_COUNT, customer_totals.items(), key=lambda pair: pair[1]
    )

    categories = {
        name: {"revenue": category_revenue[name], "orders": category_orders[name],
               "avg": category_revenue[name] / category_orders[name]}
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
