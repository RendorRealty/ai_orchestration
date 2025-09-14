import os
import time
import requests
from typing import Tuple, List, Dict, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment
GOOGLE_PLACE_KEY = os.getenv("GOOGLE_PLACE_KEY")

def get_coordinates_from_address(address: str) -> Dict[str, Any]:
    """
    Convert an address string to longitude and latitude coordinates.
    
    Args:
        address: The address string to geocode (e.g., "123 Main St, San Francisco, CA")
    
    Returns:
        Dictionary containing latitude, longitude, and additional location details
    """
    if not GOOGLE_PLACE_KEY:
        return {
            "success": False,
            "error": "Google Places API key not found",
            "message": "GOOGLE_PLACE_KEY environment variable is not set"
        }
    
    if not address or not address.strip():
        return {
            "success": False,
            "error": "Invalid address",
            "message": "Address string cannot be empty"
        }
    
    try:
        # Use Google Geocoding API to get coordinates
        geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address.strip(),
            'key': GOOGLE_PLACE_KEY
        }
        
        response = requests.get(geocoding_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            return {
                "success": False,
                "error": f"Geocoding API error: {data.get('status', 'UNKNOWN')}",
                "message": data.get('error_message', 'Failed to geocode address'),
                "address": address
            }
        
        if not data.get('results'):
            return {
                "success": False,
                "error": "No results found",
                "message": "No coordinates found for the provided address",
                "address": address
            }
        
        # Get the first (best) result
        result = data['results'][0]
        location = result['geometry']['location']
        
        return {
            "success": True,
            "latitude": location['lat'],
            "longitude": location['lng'],
            "formatted_address": result['formatted_address'],
            "address_components": result.get('address_components', []),
            "place_id": result.get('place_id'),
            "location_type": result['geometry'].get('location_type'),
            "viewport": result['geometry'].get('viewport'),
            "input_address": address
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": "Network request failed",
            "message": f"Failed to connect to Google Geocoding API: {str(e)}",
            "address": address
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Unexpected error",
            "message": f"An unexpected error occurred: {str(e)}",
            "address": address
        }

PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

def nearby_search_by_keyword(api_key: str, location: Tuple[float, float], keyword: str, max_results: int = 10, rank_by_distance: bool = True) -> List[Dict]:
    """
    Perform a Nearby Search for a keyword and return up to max_results place dicts (raw results).
    Uses rankBy=distance if rank_by_distance True (do not pass radius with rankBy=distance).
    """
    lat, lng = location
    params = {
        "key": api_key,
        "location": f"{lat},{lng}",
        "keyword": keyword,
    }
    if rank_by_distance:
        params["rankby"] = "distance"
    else:
        # Example default radius 5000m if not ranking by distance
        params["radius"] = "5000"

    places = []
    resp = requests.get(PLACES_NEARBY_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(f"NearbySearch error: {data.get('status')} - {data.get('error_message')}")

    places.extend(data.get("results", []))

    # If you wanted extra pages, you would retrieve next_page_token here.
    # For rankby=distance you usually get the closest first. We'll just slice top results.
    return places[:max_results]

def get_place_details(api_key: str, place_id: str, fields: Optional[List[str]] = None) -> Dict:
    """
    Call Place Details to get phone/address. By requesting only required fields you save quota.
    Typical fields: name,formatted_address,formatted_phone_number,geometry,place_id,rating,user_ratings_total
    """
    if fields is None:
        fields = ["name", "formatted_address", "formatted_phone_number", "geometry", "place_id", "rating", "user_ratings_total"]

    params = {
        "key": api_key,
        "place_id": place_id,
        "fields": ",".join(fields)
    }
    resp = requests.get(PLACE_DETAILS_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        # Return minimal info but include status for debugging
        return {"place_id": place_id, "details_status": data.get("status"), "error_message": data.get("error_message")}

    result = data.get("result", {})
    return result

def get_nearest_trades(address: str, max_per_category: int = 10) -> Union[Dict[str, List[Dict]], Dict[str, str]]:
    """
    Find nearest plumbers and electricians using Google Places API.
    
    Args:
        address: The address to search around
        max_per_category: Maximum number of results to return per category (default: 10)
    
    Returns:
        Dictionary with "plumbers" and "electricians" keys, each containing a list of:
        - name: Business name
        - phone: Formatted phone number
        - address: Formatted address
        - location: Dict with lat/lng coordinates
        - place_id: Google Place ID
        - rating: Average rating
        - user_ratings_total: Total number of ratings
    """
    if not GOOGLE_PLACE_KEY:
        return {"error": "GOOGLE_PLACE_KEY not found in environment variables"}
    
    geo_result = get_coordinates_from_address(address)
    if not geo_result.get("success"):
        return {"error": f"Failed to geocode address: {geo_result.get('message', '')}"}

    latitude = geo_result.get("latitude")
    longitude = geo_result.get("longitude")
    if latitude is None or longitude is None:
        return {"error": "Latitude or longitude not found for the provided address."}
    location = (float(latitude), float(longitude))
    categories = {"plumbers": "plumber", "electricians": "electrician"}
    out: Dict[str, Any] = {}

    for out_key, keyword in categories.items():
        try:
            raw_places = nearby_search_by_keyword(GOOGLE_PLACE_KEY, location, keyword, max_results=max_per_category, rank_by_distance=True)
        except Exception as e:
            out[out_key] = {"error": str(e)}
            continue

        details_list = []
        # Fetch details for top results in parallel-ish (serial here; you can use ThreadPoolExecutor but be mindful of rate limits)
        for r in raw_places:
            place_id = r.get("place_id")
            if not place_id:
                continue
            # small pause is polite; also prevents quota spikes in rapid loops (adjust as needed)
            time.sleep(0.05)
            details = get_place_details(GOOGLE_PLACE_KEY, place_id)
            # normalize geometry to lat/lng values
            geom = details.get("geometry", {})
            loc = None
            if geom:
                geoloc = geom.get("location", {})
                loc = {"lat": float(geoloc.get("lat")), "lng": float(geoloc.get("lng"))} if geoloc else None

            details_list.append({
                "name": details.get("name") or r.get("name"),
                "phone": details.get("formatted_phone_number") or None,
                "address": details.get("formatted_address") or r.get("vicinity") or None,
                "location": loc,
                "place_id": details.get("place_id") or place_id,
                "rating": details.get("rating"),
                "user_ratings_total": details.get("user_ratings_total"),
                "raw": details  # include raw details if you want more fields later
            })

        out[out_key] = details_list

    return out