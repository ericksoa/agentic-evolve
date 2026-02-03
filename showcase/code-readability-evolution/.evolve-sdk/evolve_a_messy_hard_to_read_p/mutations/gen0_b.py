"""Variant B: Functional decomposition with helper functions and Counter."""
from collections import Counter, defaultdict
from typing import Any


WHALE_SPENDING_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5


def _aggregate_customer_spending(transactions: list[dict]) -> dict[str, float]:
    """Calculate net spending per customer across sales and refunds."""
    spending: dict[str, float] = defaultdict(float)
    for txn in transactions:
        if txn["type"] == "sale":
            spending[txn["customer"]] += txn["amount"]
        else:
            spending[txn["customer"]] -= txn["amount"]
    return dict(spending)


def _aggregate_categories(transactions: list[dict]) -> dict[str, dict]:
    """Build category breakdown from sale transactions only."""
    revenue: dict[str, float] = defaultdict(float)
    orders: dict[str, int] = defaultdict(int)
    for txn in transactions:
        if txn["type"] == "sale":
            revenue[txn["category"]] += txn["amount"]
            orders[txn["category"]] += 1
    return {
        cat: {"revenue": revenue[cat], "orders": orders[cat], "avg": revenue[cat] / orders[cat]}
        for cat in revenue
    }


def _compute_total(transactions: list[dict]) -> float:
    """Sum sales and subtract refunds."""
    return sum(
        txn["amount"] if txn["type"] == "sale" else -txn["amount"]
        for txn in transactions
    )


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a comprehensive summary of transaction data.

    Processes sales and refunds to compute totals, per-customer spending,
    per-category breakdowns, and whale detection.

    Args:
        transactions: list of transaction dicts with keys 'type', 'amount', 'customer', 'category'.

    Returns:
        dict with keys: total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> result = summarize_transactions([{"type": "sale", "amount": 50, "customer": "A", "category": "X"}])
    """
    count = len(transactions)
    total = _compute_total(transactions)
    average = total / count if count else 0.0

    customer_spending = _aggregate_customer_spending(transactions)
    ranked_customers = sorted(customer_spending.items(), key=lambda item: item[1], reverse=True)

    categories = _aggregate_categories(transactions)
    has_whale = any(amount > WHALE_SPENDING_THRESHOLD for amount in customer_spending.values())

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": ranked_customers[:MAX_TOP_CUSTOMERS],
        "categories": categories,
        "has_whale": has_whale,
    }
