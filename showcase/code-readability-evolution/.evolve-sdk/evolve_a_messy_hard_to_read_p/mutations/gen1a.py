"""Variant gen1a: Branch-free loop using separate sale/refund passes."""
from collections import defaultdict
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMER_LIMIT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of transactions into aggregate metrics.

    Args:
        transactions: list of dicts with 'type', 'amount', 'customer', 'category' keys.

    Returns:
        Summary dict with 'total', 'count', 'avg', 'top_customers', 'categories', 'has_whale'.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 100.0, "customer": "A", "category": "B"}])
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
        category_revenue[txn["category"]] += amount
        category_orders[txn["category"]] += 1

    for txn in refunds:
        customer_spending[txn["customer"]] -= txn["amount"]

    total = sum(customer_spending.values())
    average = total / count if count else 0.0

    sorted_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )
    top_customers = [
        (name, spent) for name, spent in sorted_customers[:TOP_CUSTOMER_LIMIT]
    ]

    categories = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_orders[name],
            "avg": category_revenue[name] / category_orders[name],
        }
        for name in category_revenue
    }

    has_whale = any(spent > WHALE_THRESHOLD for spent in customer_spending.values())

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
