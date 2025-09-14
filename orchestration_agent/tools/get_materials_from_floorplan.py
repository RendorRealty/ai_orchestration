"""
Materials takeoff estimator for single-story wood-framed plans from floorplan images.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import List, Dict, Union

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


DEFAULT_VENDORS = ["Home Depot Canada", "RONA", "Lowe's Canada"]


def ceil_int(x: float) -> int:
    return int(math.ceil(x))


def estimate_studs(
    total_wall_length_ft: float,
    stud_spacing_in: float = 16.0,
    extra_studs_for_openings: int = 0,
    waste_pct: float = 10.0,
    plates_waste_pct: float = 10.0,
) -> int:
    """Estimate 2x4x8 count for studs + a simple allowance for plates.

    total_wall_length_ft: total linear feet of walls (exterior + interior)
    stud_spacing_in: on-center spacing (inches)
    extra_studs_for_openings: add trimmers/kings etc. for doors/windows
    waste_pct: percent extra studs for waste, cuts, corners
    plates_waste_pct: additional percent to roughly cover double top and
                      single bottom plates using the same 2x4x8 stock.
    """
    if stud_spacing_in <= 0:
        stud_spacing_in = 16.0

    spacing_ft = stud_spacing_in / 12.0

    # Base studs from spacing (very rough; ignores segment endpoints)
    base_studs = total_wall_length_ft / spacing_ft

    # Add opening studs (kings + jacks). User supplies a lump sum.
    base_studs += float(extra_studs_for_openings)

    # Apply waste/allowances
    base_studs *= (1.0 + waste_pct / 100.0)

    # Rough-in plates allowance using the same 2x4x8 stock bucket.
    base_studs *= (1.0 + plates_waste_pct / 100.0)

    return ceil_int(base_studs)


def estimate_osb_sheets(
    exterior_perimeter_ft: float,
    wall_height_ft: float = 8.0,
    waste_pct: float = 10.0,
) -> int:
    """Estimate 7/16 in 4x8 OSB sheets for exterior walls.

    Assumes full height sheathing: area = perimeter * wall_height.
    Each sheet covers 32 sq ft.
    """
    if exterior_perimeter_ft <= 0 or wall_height_ft <= 0:
        return 0

    area_sqft = exterior_perimeter_ft * wall_height_ft
    sheets = area_sqft / 32.0
    sheets *= (1.0 + waste_pct / 100.0)
    return ceil_int(sheets)


def build_materials_payload(
    total_wall_length_ft: float,
    exterior_perimeter_ft: float,
    wall_height_ft: float,
    stud_spacing_in: float,
    opening_count: int,
    studs_waste_pct: float,
    plates_waste_pct: float,
    osb_waste_pct: float,
    vendors: List[str],
) -> List[Dict]:
    # Simple allowance: assume each opening consumes ~4 extra studs
    extra_studs = max(0, int(opening_count)) * 4

    studs_qty = estimate_studs(
        total_wall_length_ft=total_wall_length_ft,
        stud_spacing_in=stud_spacing_in,
        extra_studs_for_openings=extra_studs,
        waste_pct=studs_waste_pct,
        plates_waste_pct=plates_waste_pct,
    )

    osb_qty = estimate_osb_sheets(
        exterior_perimeter_ft=exterior_perimeter_ft,
        wall_height_ft=wall_height_ft,
        waste_pct=osb_waste_pct,
    )

    materials = [
        {
            "name": "2x4x8 SPF lumber",
            "quantity": studs_qty,
            "unit": "piece",
            "spec": "SPF stud grade",
            "vendor_preferences": vendors,
        },
        {
            "name": "OSB sheathing 7/16 in 4x8",
            "quantity": osb_qty,
            "unit": "sheet",
            "spec": "7/16 in OSB",
            "vendor_preferences": vendors,
        },
    ]

    return materials


def _ftin_to_ft(ft: int | float, inches: int | float) -> float:
    try:
        return float(ft) + float(inches) / 12.0
    except Exception:
        return float(ft)


def _parse_dimensions_from_text(text: str) -> list[tuple[float, float]]:
    """Parse dimension strings like 22'11" x 18'0" into (w_ft, h_ft).

    Returns a list of width/height in feet.
    """
    dims: list[tuple[float, float]] = []

    # Normalize some unicode
    t = text.replace("\u2032", "'").replace("\u2033", '"').replace("\u00D7", "x")

    # Pattern with feet and inches on both sides
    pat_full = re.compile(
        r"(\d{1,2})\s*'\s*(\d{1,2})\s*\"?\s*[xX]\s*(\d{1,2})\s*'\s*(\d{1,2})\s*\"?"
    )
    # Pattern with only feet (no inches) on both sides
    pat_feet_only = re.compile(r"(\d{1,2})\s*'\s*[xX]\s*(\d{1,2})\s*'")

    for m in pat_full.finditer(t):
        a_ft, a_in, b_ft, b_in = m.groups()
        try:
            w = _ftin_to_ft(int(a_ft), int(a_in))
            h = _ftin_to_ft(int(b_ft), int(b_in))
            # Filter obviously bogus small/large rooms
            if 3 <= w <= 50 and 3 <= h <= 50:
                dims.append((w, h))
        except Exception:
            continue

    for m in pat_feet_only.finditer(t):
        a_ft, b_ft = m.groups()
        try:
            w = float(a_ft)
            h = float(b_ft)
            if 3 <= w <= 50 and 3 <= h <= 50:
                dims.append((w, h))
        except Exception:
            continue

    return dims


def _ocr_text_from_image(image_path: str) -> str | None:
    if Image is None or pytesseract is None:
        return None
    try:
        img = Image.open(image_path)
        # Simple pre-processing: convert to grayscale for OCR
        img = img.convert("L")
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return None


def _download_image_from_url(image_url: str) -> str:
    """Download image from URL and save to temporary file."""
    if requests is None:
        raise ImportError("requests library is required to download images from URLs")
    
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        raise Exception(f"Failed to download image from URL: {str(e)}")


def _find_floorplan_image(floorplan_path: str | None, floorplan_dir: str) -> str | None:
    if floorplan_path:
        return floorplan_path if os.path.exists(floorplan_path) else None
    
    # Common names
    candidates = [
        os.path.join(floorplan_dir, "floorplan.png"),
        os.path.join(floorplan_dir, "floorplan.jpg"),
        os.path.join(floorplan_dir, "floorplan.jpeg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    # Try common extensions
    for ext in ["png", "jpg", "jpeg", "webp"]:
        for name in ["floorplan", "plan", "layout"]:
            path = os.path.join(floorplan_dir, f"{name}.{ext}")
            if os.path.exists(path):
                return path
    return None


def estimate_from_floorplan(
    image_path: str,
    footprint_aspect_ratio: float = 1.6,
    interior_share_factor: float = 0.55,
) -> dict:
    """Estimate geometry metrics from a floorplan image via OCR heuristics.

    - Extract room dimension pairs using OCR (best-effort).
    - Approximate total footprint area from sum of room rectangles.
    - Derive an "equivalent rectangle" of aspect_ratio to estimate exterior perimeter.
    - Estimate interior wall length by scaling the sum of room perimeters
      to account for shared walls (interior_share_factor ~ 0.5-0.6 typical).

    Returns dict with keys: exterior_perimeter_ft, interior_wall_length_ft, openings
    """
    text = _ocr_text_from_image(image_path) or ""

    # Named dimension parsing: e.g., "LIVING ROOM 22'11\" x 18'0\""
    named_rooms: list[dict] = []
    try:
        # Normalize unicode and collapse spaces
        t = text.replace("\u2032", "'").replace("\u2033", '"').replace("\u00D7", "x")
        t = re.sub(r"\s+", " ", t)

        # Pattern with feet and inches both sides
        pat_named_full = re.compile(
            r"([A-Z][A-Z ]{2,}?)\s+(\d{1,2})\s*'\s*(\d{1,2})\s*\"?\s*[xX]\s*(\d{1,2})\s*'\s*(\d{1,2})\s*\"?"
        )
        for m in pat_named_full.finditer(t):
            name, a_ft, a_in, b_ft, b_in = m.groups()
            try:
                w = _ftin_to_ft(int(a_ft), int(a_in))
                h = _ftin_to_ft(int(b_ft), int(b_in))
                if 3 <= w <= 60 and 3 <= h <= 60:
                    named_rooms.append({"name": name.strip(), "w": w, "h": h})
            except Exception:
                continue

        # Fallback: feet-only
        if not named_rooms:
            pat_named_feet = re.compile(r"([A-Z][A-Z ]{2,}?)\s+(\d{1,2})\s*'\s*[xX]\s*(\d{1,2})\s*'")
            for m in pat_named_feet.finditer(t):
                name, a_ft, b_ft = m.groups()
                try:
                    w = float(a_ft)
                    h = float(b_ft)
                    if 3 <= w <= 60 and 3 <= h <= 60:
                        named_rooms.append({"name": name.strip(), "w": w, "h": h})
                except Exception:
                    continue
    except Exception:
        pass

    # Anonymous dimensions list as secondary signal
    dims = _parse_dimensions_from_text(text)

    if not dims and not named_rooms:
        # Heuristic fallback: use image size as proxy area and 8ft grid.
        try:
            if Image is not None:
                img = Image.open(image_path)
                w_px, h_px = img.size
                # Assume arbitrary scale: 25 px per foot (very rough)
                area_est = (w_px * h_px) / (25.0 * 25.0)
            else:
                area_est = 1500.0
        except Exception:
            area_est = 1500.0
        # Derive exterior perimeter from area and aspect ratio
        w = math.sqrt(area_est * footprint_aspect_ratio)
        h = area_est / w
        exterior_perimeter = 2 * (w + h)
        interior_wall_length = interior_share_factor * exterior_perimeter * 2.0
        openings = 16
        return {
            "exterior_perimeter_ft": float(exterior_perimeter),
            "interior_wall_length_ft": float(interior_wall_length),
            "openings": int(openings),
            "rooms_detected": 0,
            "named_rooms": [],
            "conditioned_ceiling_area_sqft": float(area_est * 0.7),
            "garage_ceiling_area_sqft": float(area_est * 0.3),
        }

    # Build room stats
    room_stats = []
    seen_any = False
    if named_rooms:
        for r in named_rooms:
            w = float(r["w"])
            h = float(r["h"])
            room_stats.append({
                "name": r["name"],
                "w": w,
                "h": h,
                "area": w * h,
                "perimeter": 2.0 * (w + h),
            })
            seen_any = True
    if not seen_any and dims:
        # Anonymous rooms; still capture areas for totals
        for (w, h) in dims:
            room_stats.append({
                "name": "ROOM",
                "w": w,
                "h": h,
                "area": w * h,
                "perimeter": 2.0 * (w + h),
            })

    # Compute total area and perimeters
    total_area = sum(r["area"] for r in room_stats)
    sum_room_perims = sum(r["perimeter"] for r in room_stats)

    # Equivalent rectangle
    w_equiv = math.sqrt(total_area * footprint_aspect_ratio)
    h_equiv = total_area / w_equiv if w_equiv > 0 else 0.0
    exterior_perimeter = 2.0 * (w_equiv + h_equiv)

    # Interior walls approximated from combined room perimeters with a share factor
    interior_wall_length = interior_share_factor * sum_room_perims

    # Determine basic counts for openings and areas
    names_lower = [r["name"].lower() for r in room_stats]
    conditioned_rooms = [r for r in room_stats if r["name"].lower() not in ("garage", "porch")]
    garage_rooms = [r for r in room_stats if r["name"].lower() == "garage"]
    porch_rooms = [r for r in room_stats if r["name"].lower() == "porch"]

    conditioned_area = sum(r["area"] for r in conditioned_rooms)
    garage_area = sum(r["area"] for r in garage_rooms) or 0.0

    # Heuristic openings
    likely_door_rooms = {"bedroom", "bath", "closet", "laundry room", "laundry", "office", "pantry", "room"}
    interior_door_count = sum(1 for r in room_stats if r["name"].lower() in likely_door_rooms)
    interior_door_count = max(8, min(interior_door_count, 20))
    exterior_door_count = 2
    window_count = max(8, min(int(round(exterior_perimeter / 16.0)), 20))
    openings = interior_door_count + exterior_door_count + window_count

    return {
        "exterior_perimeter_ft": float(exterior_perimeter),
        "interior_wall_length_ft": float(interior_wall_length),
        "openings": int(openings),
        "rooms_detected": len(room_stats),
        "named_rooms": room_stats,
        "conditioned_ceiling_area_sqft": float(conditioned_area),
        "garage_ceiling_area_sqft": float(garage_area),
        "interior_doors": int(interior_door_count),
        "exterior_doors": int(exterior_door_count),
        "windows": int(window_count),
    }


def build_detailed_materials_payload(
    inferred: dict,
    wall_height_ft: float,
    stud_spacing_in: float,
    studs_waste_pct: float,
    plates_waste_pct: float,
    osb_waste_pct: float,
    drywall_waste_pct: float,
    insulation_waste_pct: float,
    finish_waste_pct: float,
    vendors: List[str],
) -> List[Dict]:
    """Produce a richer materials list using floorplan-derived metrics."""
    ext_perim = float(inferred.get("exterior_perimeter_ft", 0.0))
    interior_len = float(inferred.get("interior_wall_length_ft", 0.0))
    total_wall_len = ext_perim + interior_len

    interior_doors = int(inferred.get("interior_doors", 10))
    exterior_doors = int(inferred.get("exterior_doors", 2))
    windows = int(inferred.get("windows", max(8, int(round(ext_perim / 16.0)))))

    # Studs (9ft walls use ~104-5/8" studs). Do NOT include plate allowance here.
    extra_studs = (interior_doors + exterior_doors + windows) * 4
    studs_qty = estimate_studs(
        total_wall_length_ft=total_wall_len,
        stud_spacing_in=stud_spacing_in,
        extra_studs_for_openings=extra_studs,
        waste_pct=studs_waste_pct,
        plates_waste_pct=0.0,
    )

    # Plates: double top + single bottom across all walls
    plates_linear = 3.0 * total_wall_len
    plates_pieces_10 = ceil_int(plates_linear * (1.0 + plates_waste_pct / 100.0) / 10.0)

    # Exterior sheathing area (no opening subtraction to stay conservative)
    osb_area = ext_perim * wall_height_ft
    osb_sheets = ceil_int(osb_area * (1.0 + osb_waste_pct / 100.0) / 32.0)

    # Drywall wall area (interior sides)
    interior_wall_area = interior_len * wall_height_ft * 2.0
    exterior_interior_area = ext_perim * wall_height_ft
    interior_door_area = 16.7 * interior_doors  # 30x80
    exterior_door_area = 21.0 * exterior_doors  # 36x84 approx
    window_area = 12.0 * windows                # 36x48 approx
    interior_wall_area_adj = max(0.0, interior_wall_area - interior_door_area)
    exterior_interior_area_adj = max(0.0, exterior_interior_area - exterior_door_area - window_area)
    drywall_walls_area = interior_wall_area_adj + exterior_interior_area_adj
    drywall_walls_sheets = ceil_int(drywall_walls_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)

    # Ceilings
    conditioned_area = float(inferred.get("conditioned_ceiling_area_sqft", 0.0))
    garage_area = float(inferred.get("garage_ceiling_area_sqft", 0.0))
    drywall_living_ceiling = ceil_int(conditioned_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)
    drywall_garage_ceiling = ceil_int(garage_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)

    # Insulation + wraps for exterior walls (subtract openings)
    ext_wall_area = ext_perim * wall_height_ft
    insul_area = max(0.0, ext_wall_area - window_area - exterior_door_area)
    insul_area = insul_area * (1.0 + insulation_waste_pct / 100.0)
    wrap_area = ext_wall_area * (1.0 + insulation_waste_pct / 100.0)

    # Baseboard: sum perimeters of conditioned rooms
    named_rooms = inferred.get("named_rooms") or []
    baseboard_lf = 0.0
    for r in named_rooms:
        nm = str(r.get("name", "")).lower()
        if nm in ("garage", "porch"):
            continue
        try:
            baseboard_lf += float(r.get("perimeter", 0.0))
        except Exception:
            pass
    baseboard_lf *= (1.0 + finish_waste_pct / 100.0)
    baseboard_lf_qty = ceil_int(baseboard_lf)

    materials: List[Dict] = [
        {
            "name": "2x4x104-5/8 in SPF studs",
            "quantity": studs_qty,
            "unit": "piece",
            "spec": "SPF stud grade; 9 ft walls; 16 in OC; includes openings/waste",
            "vendor_preferences": vendors,
        },
        {
            "name": "2x4x10 SPF lumber (plates)",
            "quantity": plates_pieces_10,
            "unit": "piece",
            "spec": "Double top + single bottom plates across all walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "OSB sheathing 7/16 in 4x8",
            "quantity": osb_sheets,
            "unit": "sheet",
            "spec": "Exterior walls ~9 ft height",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 1/2 in 4x8 (walls)",
            "quantity": drywall_walls_sheets,
            "unit": "sheet",
            "spec": "Interior walls both sides + interior face of exterior walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 1/2 in 4x8 (ceilings, living areas)",
            "quantity": drywall_living_ceiling,
            "unit": "sheet",
            "spec": "Conditioned area ceilings",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 5/8 in Type X 4x8 (garage ceiling)",
            "quantity": drywall_garage_ceiling,
            "unit": "sheet",
            "spec": "Fire-rated garage ceiling",
            "vendor_preferences": vendors,
        },
        {
            "name": "Fiberglass batts R-14 (3.5 in) for 2x4 walls",
            "quantity": ceil_int(insul_area),
            "unit": "sqft",
            "spec": "Exterior wall cavities",
            "vendor_preferences": vendors,
        },
        {
            "name": "House wrap (weather barrier)",
            "quantity": ceil_int(wrap_area),
            "unit": "sqft",
            "spec": "Tyvek-style wrap for exterior walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "Polyethylene vapor barrier 6 mil",
            "quantity": ceil_int(wrap_area),
            "unit": "sqft",
            "spec": "Interior side of exterior walls (where required)",
            "vendor_preferences": vendors,
        },
        {
            "name": "MDF baseboard 3-1/2 in",
            "quantity": baseboard_lf_qty,
            "unit": "linear_ft",
            "spec": "Perimeter of finished rooms",
            "vendor_preferences": vendors,
        },
        {
            "name": "Interior doors (prehung hollow-core 30x80)",
            "quantity": interior_doors,
            "unit": "piece",
            "spec": "Assorted swings",
            "vendor_preferences": vendors,
        },
        {
            "name": "Exterior doors (insulated steel 36x80)",
            "quantity": exterior_doors,
            "unit": "piece",
            "spec": "Front/rear or house-to-garage",
            "vendor_preferences": vendors,
        },
        {
            "name": "Vinyl windows (avg 36x48)",
            "quantity": windows,
            "unit": "piece",
            "spec": "Double-pane, new-construction flanged",
            "vendor_preferences": vendors,
        }
    ]

    return materials


def get_materials_from_floorplan(
    image_url: str,
    wall_height_ft: float = 9.0,
    stud_spacing_in: float = 16.0,
    aspect_ratio: float = 1.6,
    interior_share_factor: float = 0.55,
    detailed: bool = True,
    vendors: Union[List[str], None] = None
) -> Dict:
    """
    Estimate construction materials from a floorplan image via OCR.
    
    Args:
        image_url: URL to the floorplan image
        wall_height_ft: Wall height in feet (default: 9.0)
        stud_spacing_in: Stud spacing in inches OC (default: 16.0)
        aspect_ratio: Footprint aspect ratio for inference (default: 1.6)
        interior_share_factor: Interior wall factor (default: 0.55)
        detailed: Return detailed materials list (default: True)
        vendors: List of preferred vendors
    
    Returns:
        Dictionary with materials list and analysis metadata
    """
    if vendors is None:
        vendors = DEFAULT_VENDORS
    
    temp_image_path = None
    try:
        # Download image from URL
        temp_image_path = _download_image_from_url(image_url)
        
        # Analyze floorplan
        inferred = estimate_from_floorplan(
            temp_image_path,
            footprint_aspect_ratio=aspect_ratio,
            interior_share_factor=interior_share_factor,
        )
        
        # Generate materials list
        if detailed:
            materials = build_detailed_materials_payload(
                inferred=inferred,
                wall_height_ft=wall_height_ft,
                stud_spacing_in=stud_spacing_in,
                studs_waste_pct=10.0,
                plates_waste_pct=10.0,
                osb_waste_pct=10.0,
                drywall_waste_pct=10.0,
                insulation_waste_pct=10.0,
                finish_waste_pct=10.0,
                vendors=vendors,
            )
        else:
            exterior_perimeter = inferred.get("exterior_perimeter_ft", 0.0)
            interior_wall_length = inferred.get("interior_wall_length_ft", 0.0)
            openings = inferred.get("openings", 12)
            total_wall_length_ft = exterior_perimeter + interior_wall_length
            
            materials = build_materials_payload(
                total_wall_length_ft=total_wall_length_ft,
                exterior_perimeter_ft=exterior_perimeter,
                wall_height_ft=wall_height_ft,
                stud_spacing_in=stud_spacing_in,
                opening_count=openings,
                studs_waste_pct=10.0,
                plates_waste_pct=10.0,
                osb_waste_pct=10.0,
                vendors=vendors,
            )
        
        return {
            "success": True,
            "materials": materials,
            "analysis": inferred,
            "parameters": {
                "wall_height_ft": wall_height_ft,
                "stud_spacing_in": stud_spacing_in,
                "detailed": detailed
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to analyze floorplan: {str(e)}",
            "materials": [],
            "analysis": {}
        }
    
    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception:
                pass

