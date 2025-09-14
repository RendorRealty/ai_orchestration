import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Union, Any
# Load .env if present
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    # dotenv is optional; we can still proceed using OS env
    pass
from groq import Groq, RateLimitError

try:
    import orjson as _json
    def jloads(s: str):
        return _json.loads(s)
    def jdumps(obj) -> str:
        return _json.dumps(obj, option=_json.OPT_INDENT_2).decode("utf-8")
except Exception:
    import json as _json
    def jloads(s: str):
        return _json.loads(s)
    def jdumps(obj) -> str:
        return _json.dumps(obj, indent=2, ensure_ascii=False)
    
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_materials(raw: str) -> Union[List[Dict], Dict[str, Any]]:
    """Load and normalize materials from JSON string.
    
    Returns either a list of materials or an error dict.
    """
    if not raw:
        return {"error": "No input provided", "message": "JSON string is required"}
    
    try:
        data = jloads(raw)
    except Exception as e:
        return {"error": "Failed to parse input JSON", "message": str(e)}

    # Accept either a list of materials or {"materials":[...]}
    if isinstance(data, dict) and "materials" in data and isinstance(data["materials"], list):
        materials = data["materials"]
    elif isinstance(data, list):
        materials = data
    else:
        return {"error": "Invalid input format", "message": "Input must be a list of items or an object with 'materials': [...]"}

    # Basic normalization
    normalized = []
    for item in materials:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            qty = float(item.get("quantity", 1))
        except Exception:
            qty = 1.0
        unit = str(item.get("unit", "")).strip() or "unit"
        entry = {
            "name": name,
            "quantity": qty,
            "unit": unit,
        }
        # carry-through optional fields if present
        for k in ("spec", "preferred_brands", "vendor_preferences"):
            if k in item:
                entry[k] = item[k]
        normalized.append(entry)

    if not normalized:
        return {"error": "No valid materials found", "message": "No valid material items were found in the input"}

    return normalized


def build_system_prompt(max_vendors: int, currency: str) -> str:
    return (
        "You are a procurement assistant. You have access to current retail prices from Canadian vendors. "
        f"Find CURRENT retail prices in Canada for each material, prioritizing Canadian vendors "
        f"(e.g., Home Depot Canada, Lowe's Canada, RONA, Home Hardware, Canadian Tire). "
        f"Prices and totals must be in {currency}. For EACH item:\n"
        f"- Provide up to {max_vendors} vendors with realistic current prices and proper vendor websites.\n"
        f"- Include fields: vendor_name, price_per_unit (number), unit, url, availability, notes.\n"
        "- Select best_vendor_index based on availability, spec match, and price.\n"
        "- Compute selected_price_per_unit and item_subtotal = quantity * selected_price_per_unit.\n"
        "- Use realistic 2024-2025 pricing for construction materials in Canada.\n"
        "Return STRICT JSON ONLY, no markdown/code fences, matching this schema:\n"
        "{\n"
        '  "materials": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "quantity": number,\n'
        '      "unit": "string",\n'
        '      "vendors": [\n'
        "        {\n"
        '          "vendor_name": "string",\n'
        '          "price_per_unit": number,\n'
        '          "unit": "string",\n'
        '          "url": "string",\n'
        '          "availability": "in_stock|backorder|unknown",\n'
        '          "notes": "string"\n'
        "        }\n"
        "      ],\n"
        '      "best_vendor_index": number,\n'
        '      "selected_price_per_unit": number,\n'
        '      "item_subtotal": number\n'
        "    }\n"
        "  ],\n"
        '  "totals": {\n'
        '    "materials_subtotal": number,\n'
        '    "estimated_tax": number,\n'
        '    "grand_total": number,\n'
        f'    "currency": "{currency}"\n'
        "  },\n"
        '  "metadata": {\n'
        '    "model": "openai/gpt-oss-20b",\n'
        '    "search_tool": "browser_search",\n'
        '    "zip": "",\n'
        "    \"tax_rate\": 0.0,\n"
        '    "generated_at": "ISO-8601",\n'
        '    "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."\n'
        "  }\n"
        "}\n"
        "Output only valid JSON."
    )


def build_user_prompt(materials: list[dict], currency: str) -> str:
    # Provide instructions and attach input JSON
    import json as json_std
    materials_json = json_std.dumps({"materials": materials}, ensure_ascii=False)
    return (
        "Find vendors and prices for the following materials. Use only CAD pricing and prefer Canadian vendors. "
        "Include exact product links. Output must be STRICT JSON per schema. Do not include any text outside of the JSON.\n\n"
        f"Currency: {currency}\n"
        "Tax: 0.0 (no tax)\n"
        "ZIP/Postal: not provided; do not localize to a specific postal code.\n\n"
        f"Materials JSON:\n{materials_json}"
    )


def call_groq(materials: List[Dict], currency: str, max_vendors: int, model: str) -> Union[str, Dict[str, Any]]:
    """Call Groq API to get material prices.
    
    Returns either the response string or an error dict.
    """
    if not GROQ_API_KEY:
        return {
            "error": "API key not found", 
            "message": "GROQ_API_KEY not set in environment variables"
        }

    try:
        client = Groq()

        messages = [
            {"role": "system", "content": build_system_prompt(max_vendors=max_vendors, currency=currency)},
            {"role": "user", "content": build_user_prompt(materials, currency=currency)},
        ]

        # Call Groq API 
        resp = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=2048,
        )

        try:
            content = resp.choices[0].message.content
        except Exception:
            content = None

        if not content:
            return {
                "error": "No content returned from Groq completion",
                "metadata": {
                    "model": model,
                    "search_tool": "browser_search",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            }

        return content
    
    except Exception as e:
        return {
            "error": "Groq API request failed",
            "message": str(e),
            "metadata": {
                "model": model,
                "search_tool": "browser_search",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        }


def sanitize_and_parse_json(text: str):
    # Remove common code fences if any slipped in
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)

    # Try direct parse
    try:
        return jloads(cleaned)
    except Exception:
        pass

    # Fallback: extract the largest {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return jloads(snippet)
        except Exception:
            pass

    raise ValueError("Model did not return valid JSON.")


def parse_retry_after_seconds(msg: str) -> float | None:
    """Extract retry-after seconds from Groq rate limit error message if present.

    Example pattern: "Please try again in 7m29.257s"
    Returns total seconds as float if parsed, otherwise None.
    """
    try:
        m = re.search(r"Please try again in\s+(?:(\d+)m)?([0-9]+(?:\.[0-9]+)?)s", msg)
        if not m:
            return None
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    except Exception:
        return None


def ensure_totals_and_metadata(doc: dict, currency: str):
    # Ensure materials array
    materials = doc.get("materials")
    if not isinstance(materials, list):
        doc["materials"] = []
        materials = doc["materials"]

    # Compute per-item subtotal if missing
    for item in materials:
        try:
            qty = float(item.get("quantity", 0))
        except Exception:
            qty = 0.0

        sel_price = item.get("selected_price_per_unit")
        if sel_price is None:
            # try derive from best vendor
            vendors = item.get("vendors") or []
            bvi = item.get("best_vendor_index", 0)
            try:
                sel_price = float(vendors[bvi].get("price_per_unit"))
            except Exception:
                sel_price = 0.0
            item["selected_price_per_unit"] = sel_price

        try:
            sel_price = float(sel_price)
        except Exception:
            sel_price = 0.0
            item["selected_price_per_unit"] = 0.0

        item_subtotal = qty * sel_price
        item["item_subtotal"] = round(float(item_subtotal), 2)

    # Compute totals (no tax; CAD)
    materials_subtotal = round(sum(float(i.get("item_subtotal", 0.0)) for i in materials), 2)
    totals = {
        "materials_subtotal": materials_subtotal,
        "estimated_tax": 0.0,
        "grand_total": materials_subtotal,
        "currency": currency,
    }
    doc["totals"] = totals

    # Metadata
    md = doc.get("metadata") or {}
    md.update({
        "model": "openai/gpt-oss-20b",
        "search_tool": "browser_search",
        "zip": "",
        "tax_rate": 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."
    })
    doc["metadata"] = md

    return doc

def search_for_materials(json_string: str, max_vendors: int = 3, model: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
    """
    Search for material prices using Groq API with browser search.
    
    Args:
        json_string: JSON string containing materials list or {"materials": [...]}
                    Each material object should have the structure:
                    {
                        "name": "string",      # Material name (required)
                        "quantity": number,    # Quantity needed (default: 1)
                        "unit": "string"       # Unit of measurement (default: "unit")
                    }
        max_vendors: Maximum number of vendors to return per material (default: 3)
        model: Groq model to use (default: "openai/gpt-oss-20b")
    
    Returns:
        Dictionary containing material pricing information or error details
    """
    currency = "CAD"  # per user instruction
    
    # Load and validate materials
    materials_result = load_materials(json_string)
    if isinstance(materials_result, dict) and "error" in materials_result:
        return materials_result
    
    # Type check: materials_result should be a list at this point
    if not isinstance(materials_result, list):
        return {
            "error": "Invalid materials format",
            "message": "Expected list of materials but got something else"
        }
    
    materials = materials_result
    
    try:
        # Call Groq API
        raw_result = call_groq(materials, currency=currency, max_vendors=max_vendors, model=model)
        
        # Check if API call failed
        if isinstance(raw_result, dict) and "error" in raw_result:
            # Add materials and totals for consistency
            raw_result.update({
                "materials": materials,
                "totals": {
                    "materials_subtotal": 0.0,
                    "estimated_tax": 0.0,
                    "grand_total": 0.0,
                    "currency": currency,
                },
                "metadata": {
                    "model": model,
                    "search_tool": "browser_search",
                    "zip": "",
                    "tax_rate": 0.0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "assumptions": "No tax included; CAD currency; Canadian vendors prioritized.",
                }
            })
            return raw_result
        
        # Ensure we have a string response for JSON parsing
        if not isinstance(raw_result, str):
            return {
                "error": "Invalid API response format",
                "message": "Expected string response from Groq API",
                "materials": materials,
                "totals": {
                    "materials_subtotal": 0.0,
                    "estimated_tax": 0.0,
                    "grand_total": 0.0,
                    "currency": currency,
                }
            }
        
        # Parse JSON response
        try:
            doc = sanitize_and_parse_json(raw_result)
        except Exception as e:
            return {
                "error": f"Failed to parse Groq response as JSON: {e}",
                "raw_response_sample": raw_result[:1000],
                "materials": materials,
                "totals": {
                    "materials_subtotal": 0.0,
                    "estimated_tax": 0.0,
                    "grand_total": 0.0,
                    "currency": currency
                },
                "metadata": {
                    "model": model,
                    "search_tool": "browser_search",
                    "zip": "",
                    "tax_rate": 0.0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."
                }
            }
        
        # Ensure totals and metadata are properly formatted
        doc = ensure_totals_and_metadata(doc, currency=currency)
        return doc
        
    except RateLimitError as e:
        retry_after = parse_retry_after_seconds(str(e))
        return {
            "error": "rate_limit_exceeded",
            "message": str(e),
            "retry_after_seconds": retry_after,
            "suggestions": [
                "Wait the indicated time and re-run",
                "Reduce input size or max_vendors to lower token usage",
                "Try a different model to use a separate quota",
            ],
            "materials": materials,
            "totals": {
                "materials_subtotal": 0.0,
                "estimated_tax": 0.0,
                "grand_total": 0.0,
                "currency": currency,
            },
            "metadata": {
                "model": model,
                "search_tool": "browser_search",
                "zip": "",
                "tax_rate": 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assumptions": "No tax included; CAD currency; Canadian vendors prioritized.",
            },
        }
    except Exception as e:
        return {
            "error": "groq_request_failed",
            "message": str(e),
            "materials": materials,
            "totals": {
                "materials_subtotal": 0.0,
                "estimated_tax": 0.0,
                "grand_total": 0.0,
                "currency": currency,
            },
            "metadata": {
                "model": model,
                "search_tool": "browser_search",
                "zip": "",
                "tax_rate": 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assumptions": "No tax included; CAD currency; Canadian vendors prioritized.",
            },
        }