import os
import requests
import tempfile
import uuid
import base64
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def _upload_to_imgbb(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to imgbb.com (image hosting service)"""
    try:
        api_key = os.getenv('IMGBB_API_KEY')
        if not api_key:
            return {'success': False, 'error': 'IMGBB_API_KEY not found in environment variables'}
        
        # Convert image bytes to base64
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
        # Prepare the form data
        data = {
            'key': api_key,
            'image': image_base64,
            'name': filename
        }
        
        response = requests.post('https://api.imgbb.com/1/upload', data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return {
                    'success': True,
                    'url': result['data']['url'],
                    'service': 'imgbb.com',
                    'expires': 'permanent',
                    'delete_url': result['data']['delete_url'],
                    'thumb_url': result['data']['thumb']['url']
                }
        
        return {'success': False, 'error': f'Upload failed: {response.text}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _upload_to_catbox(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to catbox.moe (temporary file sharing service)"""
    try:
        files = {
            'fileToUpload': (filename, file_bytes, 'image/png')
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

def _upload_to_transfer_sh(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload file to transfer.sh (temporary file sharing service)"""
    try:
        response = requests.put(
            f'https://transfer.sh/{filename}',
            data=file_bytes,
            headers={'Content-Type': 'image/png'},
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
        _upload_to_imgbb,
        _upload_to_transfer_sh,
        _upload_to_catbox
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

def _generate_layout_drawing(image_url: str, layout_type: str, api_base_url: str = "http://127.0.0.1:8001") -> Dict[str, Any]:
    """
    Generate electrical or HVAC layout drawing from floorplan image URL
    
    Args:
        image_url: URL of the floorplan image
        layout_type: 'electrical' or 'hvac'
        api_base_url: Base URL of the layout generation API
    
    Returns:
        Dictionary with generated image bytes or error
    """
    
    try:
        # Download the source image
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        
        # Prepare the multipart form data
        files = {
            'image': ('floorplan.png', image_response.content, 'image/png')
        }
        
        # Call the layout generation API
        api_url = f"{api_base_url}/api/layout/{layout_type}"
        layout_response = requests.post(api_url, files=files, timeout=60)
        
        if layout_response.status_code == 200:
            return {
                'success': True,
                'image_bytes': layout_response.content,
                'content_type': layout_response.headers.get('content-type', 'image/png')
            }
        else:
            return {
                'success': False,
                'error': f"API returned status {layout_response.status_code}: {layout_response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }

def get_hvac_and_electrical_drawings(
    floorplan_image_url: str,
    api_base_url: str = "http://127.0.0.1:8001"
) -> Dict[str, Any]:
    """
    Generate HVAC and electrical layout drawings from a floorplan image URL.
    
    This tool takes a floorplan image, calls external APIs to generate HVAC and electrical 
    layout drawings, uploads the results to cloud storage, and returns download links.
    
    Args:
        floorplan_image_url: URL of the floorplan image to process
        api_base_url: Base URL of the layout generation API (default: http://127.0.0.1:8001)
    
    Returns:
        Dictionary containing download links for HVAC and electrical drawings
    """
    
    # Validate input
    if not floorplan_image_url or not floorplan_image_url.strip():
        return {
            "success": False,
            "error": "Floorplan image URL cannot be empty",
            "hvac_drawing": None,
            "electrical_drawing": None
        }
    
    results = {
        "success": True,
        "hvac_drawing": None,
        "electrical_drawing": None,
        "errors": []
    }
    
    # Generate unique IDs for filenames
    file_id = str(uuid.uuid4())
    
    # Generate HVAC drawing
    print("Generating HVAC layout drawing...")
    hvac_result = _generate_layout_drawing(floorplan_image_url, 'hvac', api_base_url)
    
    if hvac_result['success']:
        # Upload HVAC drawing to cloud storage
        hvac_filename = f"hvac_layout_{file_id}.png"
        hvac_upload = _upload_to_cloud_storage(hvac_result['image_bytes'], hvac_filename)
        
        if hvac_upload['success']:
            results["hvac_drawing"] = {
                "displayName": hvac_filename,
                "fileUri": hvac_upload['url'],
                "mimeType": "image/png",
                "service": hvac_upload['service'],
                "expires": hvac_upload['expires'],
                "download_url": hvac_upload['url']
            }
        else:
            # Fallback to local file
            temp_dir = os.path.join(tempfile.gettempdir(), "ai_orchestration_drawings")
            os.makedirs(temp_dir, exist_ok=True)
            hvac_path = os.path.join(temp_dir, hvac_filename)
            
            with open(hvac_path, 'wb') as f:
                f.write(hvac_result['image_bytes'])
            
            results["hvac_drawing"] = {
                "displayName": hvac_filename,
                "fileUri": f"file:///{hvac_path.replace(os.sep, '/')}",
                "mimeType": "image/png",
                "local_path": hvac_path,
                "upload_error": hvac_upload['error']
            }
    else:
        results["errors"].append(f"HVAC generation failed: {hvac_result['error']}")
    
    # Generate Electrical drawing
    print("Generating electrical layout drawing...")
    electrical_result = _generate_layout_drawing(floorplan_image_url, 'electrical', api_base_url)
    
    if electrical_result['success']:
        # Upload electrical drawing to cloud storage
        electrical_filename = f"electrical_layout_{file_id}.png"
        electrical_upload = _upload_to_cloud_storage(electrical_result['image_bytes'], electrical_filename)
        
        if electrical_upload['success']:
            results["electrical_drawing"] = {
                "displayName": electrical_filename,
                "fileUri": electrical_upload['url'],
                "mimeType": "image/png",
                "service": electrical_upload['service'],
                "expires": electrical_upload['expires'],
                "download_url": electrical_upload['url']
            }
        else:
            # Fallback to local file
            temp_dir = os.path.join(tempfile.gettempdir(), "ai_orchestration_drawings")
            os.makedirs(temp_dir, exist_ok=True)
            electrical_path = os.path.join(temp_dir, electrical_filename)
            
            with open(electrical_path, 'wb') as f:
                f.write(electrical_result['image_bytes'])
            
            results["electrical_drawing"] = {
                "displayName": electrical_filename,
                "fileUri": f"file:///{electrical_path.replace(os.sep, '/')}",
                "mimeType": "image/png",
                "local_path": electrical_path,
                "upload_error": electrical_upload['error']
            }
    else:
        results["errors"].append(f"Electrical generation failed: {electrical_result['error']}")
    
    # Determine overall success
    if results["hvac_drawing"] is None and results["electrical_drawing"] is None:
        results["success"] = False
        results["error"] = "Both HVAC and electrical drawing generation failed"
    elif len(results["errors"]) > 0:
        results["success"] = True  # Partial success
        results["warning"] = f"Some drawings failed: {'; '.join(results['errors'])}"
    
    # Add metadata
    results["metadata"] = {
        "source_url": floorplan_image_url,
        "api_base_url": api_base_url,
        "file_id": file_id
    }
    
    # Add instructions
    results["instructions"] = {
        "usage": "Download the generated HVAC and electrical layout drawings using the provided URLs",
        "note": "Files are uploaded to cloud storage for easy sharing and access"
    }
    
    return results