"""Variant A: Clean defaultdict approach with single-pass aggregation."""
from collections import defaultdict
from typing import Any

#: Spending threshold above which a customer is classified as a whale.
WHALE_THRESHOLD: float = 10_000


def summarize_transactions(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a list of transactions into aggregate metrics.

    Args:
        transactions: list of dicts with 'type', 'amount',
            'customer', 'category' keys.

    Returns:
        Summary dict with 'total', 'count', 'avg',
            'top_customers', 'categories', 'has_whale'.

    Example:
        >>> txn = {"type": "sale", "amount": 100.0,
        ...        "customer": "A", "category": "B"}
        >>> summarize_transactions([txn])
    """
    total = 0.0
    count = len(transactions)
    customer_spending: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_orders: dict[str, int] = defaultdict(int)

    for transaction in transactions:
        amount = transaction["amount"]
        customer = transaction["customer"]
        is_sale = transaction["type"] == "sale"

        if is_sale:
            total += amount
            customer_spending[customer] += amount
            category = transaction["category"]
            category_revenue[category] += amount
            category_orders[category] += 1
        else:
            total -= amount
            customer_spending[customer] -= amount

    average = total / count if count else 0.0

    sorted_customers = sorted(
        customer_spending.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_customers = [
        (name, spent) for name, spent in sorted_customers[:5]
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
