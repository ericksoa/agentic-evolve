"""Variant gen4a: Branch-free aggregation via comprehension filters."""
from collections import defaultdict
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_LIMIT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions into aggregate metrics.

    Args:
        transactions: dicts with 'type', 'amount',
            'customer', 'category' keys.

    Returns:
        Summary with totals, top customers, categories.

    Example:
        >>> summarize_transactions([])
    """
    count = len(transactions)
    customer_spending: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_orders: dict[str, int] = defaultdict(int)

    sales = [txn for txn in transactions if txn["type"] == "sale"]
    refunds = [txn for txn in transactions if txn["type"] != "sale"]

    for txn in sales:
        amount = txn["amount"]
        customer_spending[txn["customer"]] += amount
        category = txn["category"]
        category_revenue[category] += amount
        category_orders[category] += 1

    for txn in refunds:
        customer_spending[txn["customer"]] -= txn["amount"]

    total = sum(customer_spending.values())
    average = total / count if count else 0.0

    sorted_customers = sorted(
        customer_spending.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_customers = [
        (name, spent)
        for name, spent in sorted_customers[:TOP_CUSTOMER_LIMIT]
    ]

    categories = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_orders[name],
            "avg": category_revenue[name] / category_orders[name],
        }
        for name in category_revenue
    }

    has_whale = any(
        spent > WHALE_THRESHOLD
        for spent in customer_spending.values()
    )

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
