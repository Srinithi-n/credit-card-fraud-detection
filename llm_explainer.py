def generate_explanation(amount, description, prediction):
    """
    Generate a fraud explanation.
    """

    description = description.lower()

    reasons = []

    if amount > 10000:
        reasons.append(
            "The transaction amount is unusually high."
        )

    elif amount > 5000:
        reasons.append(
            "The transaction amount is higher than average."
        )

    suspicious_words = {
        "international": "international transfer",
        "wire": "wire transfer",
        "crypto": "cryptocurrency",
        "bitcoin": "cryptocurrency",
        "gift": "gift card purchase",
        "casino": "casino-related payment",
        "gambling": "gambling-related payment",
        "unknown": "unknown merchant"
    }

    for word, reason in suspicious_words.items():
        if word in description:
            reasons.append(f"Transaction involves {reason}.")

    if prediction == "Normal":
        return (
            "✅ This transaction appears to follow normal spending patterns. "
            "No unusual risk indicators were detected."
        )

    if not reasons:
        reasons.append(
            "The transaction pattern differs from normal behaviour."
        )

    explanation = "🚨 Fraud Alert\n\n"

    explanation += "Reasons:\n"

    for r in reasons:
        explanation += f"• {r}\n"

    explanation += "\nRecommended Action:\n"

    explanation += (
        "Verify customer identity before approving this transaction."
    )

    return explanation