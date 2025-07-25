#!/usr/bin/env python3
"""
Example demonstrating how to use the paragraph_enabled parameter in OmniParser API.

The paragraph_enabled parameter controls how EasyOCR processes text:
- paragraph_enabled=True (default): Groups text into paragraphs
- paragraph_enabled=False: Treats each text element separately
"""

import requests
import base64
import json

def encode_image_to_base64(image_path):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_omniparser_api(base64_image, paragraph_enabled=True, api_url="http://localhost:8000/parse/"):
    """
    Call the OmniParser API with the paragraph_enabled parameter.
    
    Args:
        base64_image (str): Base64 encoded image
        paragraph_enabled (bool): Whether to enable paragraph grouping for OCR
        api_url (str): API endpoint URL
    
    Returns:
        dict: API response
    """
    payload = {
        "base64_image": base64_image,
        "paragraph_enabled": paragraph_enabled
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Example image path (replace with your actual image path)
    image_path = "imgs/demo_image.jpg"
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        print("Testing with paragraph_enabled=True (default):")
        result_paragraph_true = call_omniparser_api(base64_image, paragraph_enabled=True)
        print(f"Latency: {result_paragraph_true.get('latency', 'N/A')}s")
        print(f"Found {len(result_paragraph_true.get('parsed_content_list', []))} elements")
        print()
        
        print("Testing with paragraph_enabled=False:")
        result_paragraph_false = call_omniparser_api(base64_image, paragraph_enabled=False)
        print(f"Latency: {result_paragraph_false.get('latency', 'N/A')}s")
        print(f"Found {len(result_paragraph_false.get('parsed_content_list', []))} elements")
        
        # The difference will be in how text elements are grouped and processed
        # paragraph_enabled=False may result in more granular text detection
        
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please replace 'image_path' with a valid image file path.")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Make sure the OmniParser server is running.")
    except Exception as e:
        print(f"Error: {e}")
