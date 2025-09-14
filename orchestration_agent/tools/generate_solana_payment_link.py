import requests
from typing import Dict, Any

def generate_solana_payment_link(
    amount: float,
    phone_number: str
) -> Dict[str, Any]:
    """
    Generate a Solana payment link for the specified amount and phone number.
    
    This tool creates a Solana pay link by calling the payment generation API
    with the provided amount and phone number.
    
    Args:
        amount: The payment amount in USD (e.g., 25.50)
        phone_number: The phone number for the payment (e.g., "234567890")
    
    Returns:
        Dictionary containing the Solana payment link or error information
    """
    
    # Validate input parameters
    if amount <= 0:
        return {
            "success": False,
            "error": "Amount must be greater than 0",
            "payment_link": None
        }
    
    if not phone_number or not phone_number.strip():
        return {
            "success": False,
            "error": "Phone number cannot be empty",
            "payment_link": None
        }
    
    # Prepare the API request
    api_url = "https://solana-nextjs-renderrealty.vercel.app/api/generate-payment-link"
    
    payload = {
        "amount": float(amount),
        "phoneNumber": phone_number.strip()
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            return {
                "success": True,
                "payment_link": result.get("paymentLink") or result.get("link") or result.get("url"),
                "amount": amount,
                "phone_number": phone_number,
                "response_data": result,
                "instructions": {
                    "usage": "Share this Solana payment link with the recipient to complete the payment",
                    "note": "The link is generated for the specified amount and phone number"
                }
            }
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {response.text}",
                "payment_link": None
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "payment_link": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "payment_link": None
        }