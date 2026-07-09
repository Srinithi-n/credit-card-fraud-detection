def predict_transaction(amount, description):
    """
    Simple fraud prediction logic for demonstration.
    Returns prediction and confidence.
    """

    description = description.lower()

    suspicious_words = [
        "bitcoin",
        "crypto",
        "wire",
        "international",
        "gift card",
        "casino",
        "gambling",
        "unknown",
        "transfer"
    ]

    score = 0

    # Amount-based rules
    if amount > 10000:
        score += 2
    elif amount > 5000:
        score += 1

    # Description-based rules
    for word in suspicious_words:
        if word in description:
            score += 2

    if score >= 3:
        return "Fraud", 95
    elif score == 2:
        return "Fraud", 80
    else:
        return "Normal", 98