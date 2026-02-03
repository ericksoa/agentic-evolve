"""Variant F: Partition-first approach - separate sales and refunds, then aggregate."""
from collections import defaultdict
from typing import Any


WHALE_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions by partitioning into sales and refunds first.

    This approach separates concerns: first partition, then aggregate each group.

    Args:
        transactions: list of dicts with 'type', 'amount', 'customer', 'category'.

    Returns:
        dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 100, "customer": "A", "category": "B"}])
    """
    sales = [txn for txn in transactions if txn["type"] == "sale"]
    refunds = [txn for txn in transactions if txn["type"] == "refund"]

    sale_total = sum(txn["amount"] for txn in sales)
    refund_total = sum(txn["amount"] for txn in refunds)
    total = sale_total - refund_total
    count = len(transactions)
    average = total / count if count else 0.0

    # Customer spending: accumulate sales, subtract refunds
    customer_spending: dict[str, float] = defaultdict(float)
    for txn in sales:
        customer_spending[txn["customer"]] += txn["amount"]
    for txn in refunds:
        customer_spending[txn["customer"]] -= txn["amount"]

    ranked_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )

    # Category breakdown: only sales count
    category_stats: dict[str, dict] = {}
    for txn in sales:
        category = txn["category"]
        if category not in category_stats:
            category_stats[category] = {"revenue": 0.0, "orders": 0}
        category_stats[category]["revenue"] += txn["amount"]
        category_stats[category]["orders"] += 1

    for stats in category_stats.values():
        stats["avg"] = stats["revenue"] / stats["orders"]

    has_whale = any(amount > WHALE_THRESHOLD for amount in customer_spending.values())

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": ranked_customers[:MAX_TOP_CUSTOMERS],
        "categories": category_stats,
        "has_whale": has_whale,
    }
