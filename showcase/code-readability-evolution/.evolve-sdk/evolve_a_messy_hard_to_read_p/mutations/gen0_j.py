"""Variant J: Maximum performance - precomputed lookups, minimal branching, tight loop."""
from typing import Any


WHALE_THRESHOLD = 10_000


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions with maximum runtime performance.

    Minimizes per-iteration overhead through local variable binding,
    precomputed type checks, and inline accumulation.

    Args:
        transactions: list of dicts with type, amount, customer, category.

    Returns:
        dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 1, "customer": "A", "category": "B"}])
    """
    count = len(transactions)
    if not count:
        return {
            "total": 0.0,
            "count": 0,
            "avg": 0.0,
            "top_customers": [],
            "categories": {},
            "has_whale": False,
        }

    total = 0.0
    customer_spending: dict[str, float] = {}
    cat_rev: dict[str, float] = {}
    cat_ord: dict[str, int] = {}
    has_whale = False

    # Bind methods locally for tight loop performance
    cust_get = customer_spending.get
    rev_get = cat_rev.get
    ord_get = cat_ord.get
    threshold = WHALE_THRESHOLD

    for txn in transactions:
        amount = txn["amount"]
        customer = txn["customer"]

        if txn["type"] == "sale":
            total += amount
            spend = cust_get(customer, 0.0) + amount
            customer_spending[customer] = spend
            if spend > threshold:
                has_whale = True
            category = txn["category"]
            cat_rev[category] = rev_get(category, 0.0) + amount
            cat_ord[category] = ord_get(category, 0) + 1
        else:
            total -= amount
            customer_spending[customer] = cust_get(customer, 0.0) - amount

    average = total / count

    sorted_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )

    categories = {
        name: {"revenue": cat_rev[name], "orders": cat_ord[name], "avg": cat_rev[name] / cat_ord[name]}
        for name in cat_rev
    }

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": sorted_customers[:5],
        "categories": categories,
        "has_whale": has_whale,
    }
