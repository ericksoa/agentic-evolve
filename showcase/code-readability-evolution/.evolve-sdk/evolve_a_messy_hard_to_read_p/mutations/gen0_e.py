"""Variant E: Performance-optimized with local variable caching and minimal overhead."""
from typing import Any


WHALE_THRESHOLD = 10_000


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions with optimized single-pass processing.

    Uses local variable caching and dict.get for speed.

    Args:
        transactions: list of dicts with type, amount, customer, category keys.

    Returns:
        Summary dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 10, "customer": "A", "category": "B"}])
    """
    total = 0.0
    count = len(transactions)
    customer_totals: dict[str, float] = {}
    cat_revenue: dict[str, float] = {}
    cat_orders: dict[str, int] = {}
    has_whale = False

    # Local references for speed
    cust_get = customer_totals.get
    rev_get = cat_revenue.get
    ord_get = cat_orders.get

    for txn in transactions:
        amount = txn["amount"]
        customer = txn["customer"]
        txn_type = txn["type"]

        if txn_type == "sale":
            total += amount
            new_spend = cust_get(customer, 0.0) + amount
            customer_totals[customer] = new_spend
            if new_spend > WHALE_THRESHOLD:
                has_whale = True
            category = txn["category"]
            cat_revenue[category] = rev_get(category, 0.0) + amount
            cat_orders[category] = ord_get(category, 0) + 1
        else:
            total -= amount
            new_spend = cust_get(customer, 0.0) - amount
            customer_totals[customer] = new_spend

    average = total / count if count else 0.0

    sorted_customers = sorted(customer_totals.items(), key=lambda pair: pair[1], reverse=True)
    top_customers = sorted_customers[:5]

    categories = {
        name: {
            "revenue": cat_revenue[name],
            "orders": cat_orders[name],
            "avg": cat_revenue[name] / cat_orders[name],
        }
        for name in cat_revenue
    }

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
