# GUI-Actor Grounding API

## Endpoint

```
POST https://aucvu2pq4vswcj-8888.proxy.runpod.net/ground/
```

## Authentication

All requests (except `/probe/`) require an API key header:

```
X-API-Key: JIAYI
```

## Request

```json
{
  "base64_image": "<base64-encoded PNG or JPEG>",
  "instruction": "click on Elementary checkbox",
  "topk": 3,
  "use_verifier": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `base64_image` | string | yes | Screenshot as base64 |
| `instruction` | string | yes | Natural language description of the element to find |
| `topk` | int | no | Max candidates to return (default: 3) |
| `use_verifier` | bool | no | Override server default for verifier (default: server setting) |

## Response

```json
{
  "points": [
    {
      "x": 523.14,
      "y": 360.62,
      "x_norm": 0.4087,
      "y_norm": 0.5009
    }
  ],
  "topk_values": [0.994],
  "image_size": [1280, 720],
  "latency": 1.46,
  "verified": true
}
```

| Field | Description |
|---|---|
| `points` | Ranked list of click targets |
| `points[].x`, `y` | Pixel coordinates on the screenshot |
| `points[].x_norm`, `y_norm` | Normalized coordinates [0,1] — multiply by actual screen size if different from screenshot |
| `topk_values` | Confidence scores for each point. When `verified: true` these are verifier scores (0-1, higher = more confident). When `verified: false` these are raw attention scores (typically 0-0.2) |
| `image_size` | Screenshot dimensions `[width, height]` in pixels |
| `latency` | Server-side processing time in seconds |
| `verified` | Whether the verifier model was used to re-rank candidates |

## Confidence interpretation

| Score | Meaning |
|---|---|
| > 0.9 | High confidence — click it |
| 0.8 - 0.9 | Likely correct |
| < 0.8 | Low confidence — element may not exist or instruction is ambiguous |

## Example

```bash
IMAGE_B64=$(base64 -i screenshot.png | tr -d '\n')

curl -X POST https://aucvu2pq4vswcj-8888.proxy.runpod.net/ground/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: JIAYI" \
  -d "{\"base64_image\": \"$IMAGE_B64\", \"instruction\": \"click the Submit button\"}"
```

## Health check

```bash
curl https://aucvu2pq4vswcj-8888.proxy.runpod.net/probe/
# {"message": "Omniparser API ready"}
```
