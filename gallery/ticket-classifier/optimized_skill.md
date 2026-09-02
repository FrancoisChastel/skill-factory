---
name: ticket-classifier
description: Classify a customer support message into a single category. Use when
  routing an incoming support ticket, email, or chat message to the right queue.
---

# Ticket Classifier

Classify the customer's message into exactly one of the following categories:
- `billing`: Charges, invoices, refunds, or plan limits.
- `bug`: Technical errors, crashes, or unexpected behavior.
- `feature_request`: Suggestions for new functionality or improvements.
- `account`: Login issues, 2FA, password resets, or email changes.
- `other`: Anything that does not fit the above.

## Constraints
- Output ONLY the lowercase label.
- Do not include punctuation, quotes, explanations, or any other text.
- Use underscores for multi-word labels (e.g., `feature_request`).
