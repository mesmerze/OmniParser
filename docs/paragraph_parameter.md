# OmniParser API Parameter Documentation

## Paragraph Enabled Parameter

The OmniParser API now supports a `paragraph_enabled` parameter that controls how OCR (Optical Character Recognition) processes text in images.

### Overview

The `paragraph_enabled` parameter is a boolean value that determines the text grouping behavior of the EasyOCR component:

- **`paragraph_enabled: true`** (default): Groups detected text into paragraphs/blocks
- **`paragraph_enabled: false`**: Treats each text element separately without grouping

### API Usage

#### Server API Endpoint

**POST** `/parse/`

**Request Body:**
```json
{
  "base64_image": "base64_encoded_image_string",
  "paragraph_enabled": true  // Optional, defaults to true
}
```

**Response:**
```json
{
  "som_image_base64": "base64_encoded_result_image",
  "parsed_content_list": [...],
  "latency": 1.23,
  "image_size": [width, height]
}
```

#### Python API

```python
from util.omniparser import Omniparser

# Initialize OmniParser
config = {
    'som_model_path': 'weights/icon_detect/model.pt',
    'caption_model_name': 'florence2',
    'caption_model_path': 'weights/icon_caption_florence',
    'BOX_TRESHOLD': 0.05
}
omniparser = Omniparser(config)

# Parse with paragraph grouping enabled (default)
result1 = omniparser.parse(base64_image, paragraph_enabled=True)

# Parse with paragraph grouping disabled
result2 = omniparser.parse(base64_image, paragraph_enabled=False)
```

### When to Use Each Setting

#### Use `paragraph_enabled: true` when:
- You want coherent text blocks (paragraphs, sentences)
- Processing documents, articles, or UI with structured text
- Need better readability of continuous text content
- Working with forms, dialogs, or text-heavy interfaces

#### Use `paragraph_enabled: false` when:
- You need granular text detection (individual words/phrases)
- Processing interfaces with scattered text elements
- Working with button labels, menu items, or isolated text
- Need precise positioning of individual text components
- Processing technical interfaces with short labels and values

### Examples

#### Example 1: Document Processing
```python
# For documents with paragraphs, enable paragraph grouping
dino_img, content_list, size = omniparser.parse(document_image, paragraph_enabled=True)
# Result: Text grouped into coherent paragraphs
```

#### Example 2: UI Element Detection
```python
# For UI with buttons and labels, disable paragraph grouping
dino_img, content_list, size = omniparser.parse(ui_image, paragraph_enabled=False)
# Result: Individual buttons, labels, and text elements detected separately
```

### Backward Compatibility

The parameter is optional and defaults to `True` to maintain backward compatibility with existing implementations. All existing API calls will continue to work without modification.

### HTTP Request Examples

#### cURL Example
```bash
# With paragraph grouping enabled
curl -X POST "http://localhost:8000/parse/" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_image": "your_base64_image_here",
    "paragraph_enabled": true
  }'

# With paragraph grouping disabled
curl -X POST "http://localhost:8000/parse/" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_image": "your_base64_image_here",
    "paragraph_enabled": false
  }'
```

#### Python Requests Example
```python
import requests
import json

payload = {
    "base64_image": base64_image_string,
    "paragraph_enabled": False  # Disable for granular text detection
}

response = requests.post(
    "http://localhost:8000/parse/",
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)

result = response.json()
```

### Impact on Performance

The `paragraph_enabled` parameter affects the OCR processing phase:
- **`paragraph_enabled: true`**: May be slightly slower due to text grouping algorithms
- **`paragraph_enabled: false`**: Potentially faster with more individual text elements

The difference in performance is typically minimal, so choose based on your use case requirements rather than performance considerations.
