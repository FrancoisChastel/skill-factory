---
name: invoice-extractor
description: Extract key fields from messy invoice or receipt text into structured
  JSON. Use when the user pastes an invoice, receipt, or bill and wants the vendor,
  date, total, and currency pulled out.
---

# Invoice Extractor

Extract invoice data into a raw JSON object. 

## Output Format
Return ONLY the JSON object. Do not include markdown code fences (e.g., ```json), prose, or any text outside the JSON structure.

## Schema
The JSON must contain exactly these keys:
- `vendor`: The name of the company or provider.
- `date`: The date normalized to ISO 8601 format (YYYY-MM-DD).
- `total`: The final total amount as a number (float/integer). Normalize localized decimal separators (e.g., convert "7,50" to 7.5).
- `currency`: The ISO 4217 three-letter currency code (e.g., USD, EUR, JPY).

## Extraction Rules
1. **Date Normalization**: Convert any localized or regional date format into YYYY-MM-DD.
2. **Numeric Cleaning**: Remove currency symbols and thousands separators from the `total` value; ensure it is a valid JSON number.
3. **Currency Mapping**: Map currency symbols (e.g., €, ¥, $) to their respective ISO 4217 codes.
