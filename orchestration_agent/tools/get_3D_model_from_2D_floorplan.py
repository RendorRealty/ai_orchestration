import os
import requests
from io import BytesIO
from typing import Dict, Any
import numpy as np
import trimesh
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import cv2
import easyocr
import tempfile
import uuid
import json
import base64

DEFAULT_INVERT = True
DEFAULT_BINARY = True
DEFAULT_SMOOTH_RADIUS = 0.25
DEFAULT_MEDIAN_SIZE = 3
DEFAULT_OPEN_SIZE = 2
DEFAULT_CLOSE_SIZE = 3
DEFAULT_DILATE_SIZE = 1
DEFAULT_KEEP_LARGEST = 150
DEFAULT_MIN_AREA_RATIO = 0.00003
DEFAULT_BASE_THICKNESS_MM = 1.0

# Initialize EasyOCR reader once (global to avoid repeated initialization)
_ocr_reader = None

def _upload_to_tmpfiles(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to tmpfiles.org (temporary file sharing service)"""
    try:
        files = {
            'file': (filename, file_bytes, 'application/sla')
        }
        response = requests.post('https://tmpfiles.org/api/v1/upload', files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                # tmpfiles.org returns a download URL in the data
                download_url = result['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                return {
                    'success': True,
                    'url': download_url,
                    'service': 'tmpfiles.org',
                    'expires': '1 hour'
                }
        
        return {'success': False, 'error': 'Upload failed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _upload_to_catbox(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to catbox.moe (temporary file sharing service)"""
    try:
        files = {
            'fileToUpload': (filename, file_bytes, 'application/sla')
        }
        data = {
            'reqtype': 'fileupload'
        }
        response = requests.post('https://catbox.moe/user/api.php', files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://files.catbox.moe/'):
                return {
                    'success': True,
                    'url': url,
                    'service': 'catbox.moe',
                    'expires': 'permanent'
                }
        
        return {'success': False, 'error': 'Upload failed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _upload_to_0x0(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to 0x0.st (temporary file sharing service)"""
    try:
        files = {
            'file': (filename, file_bytes, 'application/sla')
        }
        response = requests.post('https://0x0.st', files=files, timeout=30)
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                return {
                    'success': True,
                    'url': url,
                    'service': '0x0.st',
                    'expires': '365 days'
                }
        
        return {'success': False, 'error': 'Upload failed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _upload_to_transfer_sh(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to transfer.sh (temporary file sharing service)"""
    try:
        response = requests.put(
            f'https://transfer.sh/{filename}',
            data=file_bytes,
            headers={'Content-Type': 'application/sla'},
            timeout=30
        )
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                return {
                    'success': True,
                    'url': url,
                    'service': 'transfer.sh',
                    'expires': '14 days'
                }
        
        return {'success': False, 'error': 'Upload failed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _upload_to_cloud_storage(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Try multiple cloud storage services until one succeeds"""
    
    # List of upload functions to try
    upload_services = [
        _upload_to_transfer_sh,
        _upload_to_catbox,
        _upload_to_tmpfiles,
        _upload_to_0x0
    ]
    
    for upload_func in upload_services:
        result = upload_func(file_bytes, filename)
        if result['success']:
            return result
    
    # If all services fail, return local file as fallback
    return {
        'success': False,
        'error': 'All cloud storage services failed',
        'fallback': 'local_file'
    }

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

def _remove_text_with_ocr(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Remove text regions detected by OCR from the mask.
    
    Args:
        image: Original grayscale PIL image 
        mask: Binary mask to remove text from
    Returns:
        mask with text regions removed
    """
    try:
        # Convert PIL image to numpy array for OCR
        img_array = np.array(image)
        
        # Get OCR reader
        reader = _get_ocr_reader()
        
        # Detect text (returns list of [bbox, text, confidence])
        results = reader.readtext(img_array, paragraph=False)
        
        # Create output mask
        out_mask = mask.copy()
        
        # Remove each detected text region with padding
        for detection in results:
            bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            confidence = detection[2]
            
            # Only remove if confidence is high enough
            if confidence < 0.3:
                continue
                
            # Get bounding box coordinates
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            x_min, x_max = max(0, min(xs) - 2), min(mask.shape[1], max(xs) + 2)
            y_min, y_max = max(0, min(ys) - 2), min(mask.shape[0], max(ys) + 2)
            
            # Clear text region from mask
            out_mask[y_min:y_max, x_min:x_max] = False
            
        return out_mask
    except Exception as e:
        # If OCR fails, return original mask
        print(f"OCR text removal failed: {e}")
        return mask

def _remove_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """Fallback: Remove only very small noise components."""
    h, w = mask.shape
    total = h * w
    min_area = max(1, int(total * max(0.0, float(min_area_ratio))))
    
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                indices = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                indices.append((ny, nx))
                # Only remove if truly tiny
                if len(indices) >= min_area:
                    for (yy, xx) in indices:
                        out[yy, xx] = True
    return out

def _filter_components_by_area(mask: np.ndarray, keep_largest: int, min_area_ratio: float) -> np.ndarray:
    """Keep only the largest connected components (8-connectivity).
    mask: boolean array (H, W)
    keep_largest: number of components to keep
    min_area_ratio: minimum area relative to image to keep (0..1)
    """
    h, w = mask.shape
    total = h * w
    min_area = max(1, int(total * max(0.0, float(min_area_ratio))))

    visited = np.zeros_like(mask, dtype=bool)
    comps = []  # list of (area, indices_list)

    # Neighbor offsets for 8-connectivity
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                # BFS/DFS
                stack = [(y, x)]
                visited[y, x] = True
                indices = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                indices.append((ny, nx))
                comps.append((len(indices), indices))

    if not comps:
        return np.zeros_like(mask, dtype=bool)

    # Sort by area desc and keep top K meeting min_area
    comps.sort(key=lambda t: t[0], reverse=True)
    kept = 0
    out = np.zeros_like(mask, dtype=bool)
    for area, indices in comps:
        if kept >= int(max(1, keep_largest)):
            break
        if area < min_area:
            continue
        for (yy, xx) in indices:
            out[yy, xx] = True
        kept += 1
    return out


def _load_png_as_grayscale(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")
    
    # Convert to grayscale
    return img.convert("L")


def _downsample(img: Image.Image, max_dim: int) -> Image.Image:
    if max_dim is None or max_dim <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


def _otsu_threshold(arr_u8: np.ndarray) -> int:
    # arr_u8: uint8 array (H, W)
    hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.float64)
    total = arr_u8.size
    if total == 0:
        return 128
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return idx


def _preprocess_to_heightmap(
    img_gray: Image.Image,
    invert: bool,
    binary: bool,
    smooth_radius: float,
    open_size: int,
    close_size: int,
    dilate_size: int,
    median_size: int,
    keep_largest: int,
    min_area_ratio: float,
) -> np.ndarray:
    """Return float32 height factor in [0,1] from PIL L image using PIL filters only."""
    im = img_gray

    # Optional invert first to make features of interest bright
    if invert:
        im = ImageOps.invert(im)

    # Contrast enhancement to spread dynamic range
    im = ImageOps.autocontrast(im)

    # Optional denoise and smooth
    if median_size and median_size >= 3:
        im = im.filter(ImageFilter.MedianFilter(_ensure_odd(median_size)))
    if smooth_radius and smooth_radius > 0.0:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(smooth_radius)))

    if binary:
        arr_u8_a = np.asarray(im, dtype=np.uint8)
        thr_a = _otsu_threshold(arr_u8_a)
        mask_a = (arr_u8_a >= thr_a).astype(np.uint8)
        occ_a = mask_a.mean() if mask_a.size else 0.0

        # Try inverted variant and choose the one with lower but non-trivial occupancy
        arr_u8_b = 255 - arr_u8_a
        thr_b = _otsu_threshold(arr_u8_b)
        mask_b = (arr_u8_b >= thr_b).astype(np.uint8)
        occ_b = mask_b.mean() if mask_b.size else 0.0

        chosen = mask_a
        if 0.001 < occ_b < occ_a:
            chosen = mask_b
        # If occupancy still huge (> 40%), pick the smaller of the two
        if occ_a > 0.4 or occ_b > 0.4:
            chosen = mask_a if occ_a <= occ_b else mask_b

        bim = Image.fromarray((chosen * 255).astype(np.uint8), mode="L")
        orig_bim = bim.copy()

        # NOTE: Keep it simple to avoid grid artifacts: rely on global Otsu only.

        # Morphology (opening then closing). We'll adapt if this is too destructive.
        def _apply_morph(img_l: Image.Image, os: int, cs: int) -> Image.Image:
            os = _ensure_odd(os)
            cs = _ensure_odd(cs)
            out = img_l.filter(ImageFilter.MinFilter(os))
            out = out.filter(ImageFilter.MaxFilter(os))
            out = out.filter(ImageFilter.MaxFilter(cs))
            out = out.filter(ImageFilter.MinFilter(cs))
            return out

        # Slightly stronger despeckle to kill grid noise
        os_eff = max(3, open_size)
        bim = _apply_morph(bim, os_eff, close_size)
        # If after morph the area is extremely low, relax morphology
        pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)
        area_frac = pre_mask.mean() if pre_mask.size else 0.0
        if area_frac < 0.01:
            bim = _apply_morph(orig_bim, 3, 3)
            pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)

        # Optional thickness via dilation
        dsz = _ensure_odd(dilate_size)
        if dsz > 1:
            bim = bim.filter(ImageFilter.MaxFilter(dsz))
            pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)

        # Remove fancy fusions; stick to morphology + component pruning only.

        # First remove tiny noise
        clean_mask = _remove_small_components(pre_mask, min_area_ratio=max(min_area_ratio, 0.00015))
        
        # Then use OCR to detect and remove text
        clean_mask = _remove_text_with_ocr(img_gray, clean_mask)
        
        # If removal was too aggressive, fall back to pre_mask
        if clean_mask.mean() < 0.005:
            clean_mask = pre_mask
        return clean_mask.astype(np.float32)
    else:
        # Grayscale heightmap
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr


def _heightmap_mesh(z_map: np.ndarray, scale_mm_per_px: float) -> trimesh.Trimesh:
    # z_map: (H, W) float32 in mm
    h, w = z_map.shape
    # Create grid of XY in mm
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    x = x_idx * scale_mm_per_px
    y = y_idx * scale_mm_per_px

    # Flatten vertices
    vertices = np.column_stack((x.ravel(), y.ravel(), z_map.ravel())).astype(np.float32)

    # Build faces: two triangles per grid cell
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    v00 = idx[:-1, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel()
    v11 = idx[1:, 1:].ravel()

    faces1 = np.column_stack((v00, v01, v10))
    faces2 = np.column_stack((v11, v10, v01))
    faces = np.vstack((faces1, faces2)).astype(np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def convert_png_to_stl(
    image_bytes: bytes,
    max_height_mm: float,
    scale_mm_per_px: float,
    downsample: int,
    invert: bool,
    binary: bool,
    smooth_radius: float,
    open_size: int,
    close_size: int,
    dilate_size: int,
    median_size: int,
    keep_largest: int,
    min_area_ratio: float,
) -> bytes:
    # 1) Load and validate
    img = _load_png_as_grayscale(image_bytes)

    # 2) Downsample
    downsample = int(max(16, min(4096, downsample)))
    img = _downsample(img, downsample)

    # 3) Preprocess (denoise, threshold/morph) to reduce spikes
    arr01 = _preprocess_to_heightmap(
        img_gray=img,
        invert=invert,
        binary=binary,
        smooth_radius=smooth_radius,
        open_size=open_size,
        close_size=close_size,
        dilate_size=dilate_size,
        median_size=median_size,
        keep_largest=keep_largest,
        min_area_ratio=min_area_ratio,
    )
    # No final blur: keep flat base and avoid grid ripples
    arr01 = np.clip(arr01, 0.0, 1.0)
    z_map = DEFAULT_BASE_THICKNESS_MM + arr01 * float(max_height_mm)

    # 4) Triangulate grid
    mesh = _heightmap_mesh(z_map, scale_mm_per_px=float(scale_mm_per_px))

    # 5) Export binary STL
    buffer = BytesIO()
    mesh.export(buffer, file_type="stl")
    return buffer.getvalue()

def generate_3d_model_from_floorplan(
    image_url: str,
    max_height_mm: float = 10.0,
    scale_mm_per_px: float = 0.2,
    downsample: int = 1024
) -> Dict[str, Any]:
    """
    Convert a 2D floorplan image from URL to a 3D STL model file.
    
    Args:
        image_url: URL of the floorplan image to process
        max_height_mm: Maximum height of the 3D model in millimeters (default: 10.0)
        scale_mm_per_px: Scale factor from pixels to millimeters (default: 0.2)
        downsample: Maximum dimension for image processing (default: 1024)
    
    Returns:
        Dictionary containing the STL file path and metadata
    """
    
    # Validate parameters
    if max_height_mm <= 0:
        return {
            "success": False,
            "error": "Invalid parameter: max_height_mm must be > 0",
            "file_data": None
        }
    
    if scale_mm_per_px <= 0:
        return {
            "success": False,
            "error": "Invalid parameter: scale_mm_per_px must be > 0", 
            "file_data": None
        }
    
    if not image_url or not image_url.strip():
        return {
            "success": False,
            "error": "Image URL cannot be empty",
            "file_data": None
        }
    
    try:
        # Download image from URL
        response = requests.get(image_url.strip(), timeout=30)
        response.raise_for_status()
        
        # Check file size (limit to 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(response.content) > max_size:
            return {
                "success": False,
                "error": f"Image file too large. Maximum size is {max_size // (1024*1024)} MB",
                "file_data": None
            }
        
        # Process the image and generate STL
        stl_bytes = convert_png_to_stl(
            image_bytes=response.content,
            max_height_mm=max_height_mm,
            scale_mm_per_px=scale_mm_per_px,
            downsample=downsample,
            invert=DEFAULT_INVERT,
            binary=DEFAULT_BINARY,
            smooth_radius=DEFAULT_SMOOTH_RADIUS,
            open_size=DEFAULT_OPEN_SIZE,
            close_size=DEFAULT_CLOSE_SIZE,
            dilate_size=DEFAULT_DILATE_SIZE,
            median_size=DEFAULT_MEDIAN_SIZE,
            keep_largest=DEFAULT_KEEP_LARGEST,
            min_area_ratio=DEFAULT_MIN_AREA_RATIO,
        )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"floorplan_3d_model_{file_id}.stl"
        
        # Upload to cloud storage
        upload_result = _upload_to_cloud_storage(stl_bytes, filename)
        
        if upload_result['success']:
            # Return cloud storage URL
            return {
                "success": True,
                "file_data": {
                    "displayName": filename,
                    "fileUri": upload_result['url'],
                    "mimeType": "application/sla"
                },
                "cloud_storage": {
                    "service": upload_result['service'],
                    "expires": upload_result['expires'],
                    "download_url": upload_result['url']
                },
                "metadata": {
                    "max_height_mm": max_height_mm,
                    "scale_mm_per_px": scale_mm_per_px,
                    "downsample": downsample,
                    "file_size_bytes": len(stl_bytes),
                    "source_url": image_url
                },
                "instructions": {
                    "usage": f"STL file uploaded to {upload_result['service']} and expires in {upload_result['expires']}",
                    "download": f"Download from: {upload_result['url']}",
                    "mime_type": "application/sla",
                    "filename": filename
                }
            }
        else:
            # Fallback to local file if cloud upload fails
            output_dir = os.path.join(tempfile.gettempdir(), "ai_orchestration_stl_files")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            
            # Save STL file locally
            with open(file_path, 'wb') as f:
                f.write(stl_bytes)
            
            # Convert to file URI
            file_uri = f"file:///{file_path.replace(os.sep, '/')}"
            
            return {
                "success": True,
                "file_data": {
                    "displayName": filename,
                    "fileUri": file_uri,
                    "mimeType": "application/sla"
                },
                "cloud_storage": {
                    "error": upload_result['error'],
                    "fallback": "local_file"
                },
                "metadata": {
                    "max_height_mm": max_height_mm,
                    "scale_mm_per_px": scale_mm_per_px,
                    "downsample": downsample,
                    "file_size_bytes": len(stl_bytes),
                    "source_url": image_url,
                    "local_path": file_path
                },
                "instructions": {
                    "usage": "Cloud upload failed, STL file saved locally",
                    "warning": "File is only accessible on the local machine",
                    "mime_type": "application/sla",
                    "filename": filename,
                    "local_path": file_path
                }
            }
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to download image from URL: {str(e)}",
            "file_data": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to process image: {str(e)}",
            "file_data": None
        }