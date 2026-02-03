"""Variant C: Single-pass with inlined logic and heapq for top-k customers."""
import heapq
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_COUNT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions into totals, customer rankings, and category breakdowns.

    Performs a single pass over the data, then uses heapq for efficient top-k selection.

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
        amount = txn["amount"]
        customer = txn["customer"]

        if txn["type"] == "sale":
            total += amount
            customer_totals[customer] = customer_totals.get(customer, 0.0) + amount
            category = txn["category"]
            if category in category_data:
                entry = category_data[category]
                entry[0] += amount
                entry[1] += 1
            else:
                category_data[category] = [amount, 1]
        else:
            total -= amount
            customer_totals[customer] = customer_totals.get(customer, 0.0) - amount

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
