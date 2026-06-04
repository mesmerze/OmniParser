# UI-TARS Grounding API

Backed by `ByteDance-Seed/UI-TARS-1.5-7B` (a Qwen2.5-VL-7B vision-language-action fine-tune trained for GUI grounding).

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
  "instruction": "Click on the Sicher zur Kasse button.",
  "topk": 1
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `base64_image` | string | yes | Screenshot as base64 |
| `instruction` | string | yes | Natural language description of the element to find. Phrasing as a click command (e.g. "Click on …") tends to work best. |
| `topk` | int | no | Number of candidates to return (default: 1). For `topk > 1` the model is sampled with mild temperature for diversity; for `topk = 1` decoding is deterministic. |

## Response

```json
{
  "points": [
    {
      "x": 1258.0,
      "y": 680.0,
      "x_norm": 0.9828,
      "y_norm": 0.9444,
      "bbox": null
    }
  ],
  "topk_values": [1.0],
  "image_size": [1280, 720],
  "latency": 1.84,
  "verified": false,
  "raw_outputs": ["(1258,680)"]
}
```

| Field | Description |
|---|---|
| `points[].x`, `y` | Pixel coordinates on the screenshot |
| `points[].x_norm`, `y_norm` | Normalized `[0, 1]` coordinates — multiply by your actual screen size if it differs from the screenshot |
| `points[].bbox` | `[x1, y1, x2, y2]` in pixel coords when the model emitted an action-format bbox; `null` when it returned a bare point |
| `topk_values` | Placeholder confidences (always `1.0`). UI-TARS does not expose a calibrated confidence — treat all returned candidates as "model thinks this is the answer." Use repeated sampling with topk>1 if you need an agreement signal. |
| `image_size` | Screenshot dimensions `[width, height]` in pixels |
| `latency` | Server-side processing time in seconds |
| `verified` | Always `false`. The previous GUI-Actor + Verifier re-ranking has been removed. |
| `raw_outputs` | The model's raw decoded output(s), one per returned sequence. Useful for debugging when no point was parsed. |

## What changed from the GUI-Actor backend

This endpoint previously ran `microsoft/GUI-Actor-7B-Qwen2.5-VL` with an attention-pointer head and an optional `GUI-Actor-Verifier-2B` re-ranker. The grounding backbone has been fully replaced with UI-TARS-1.5-7B. Behavioral differences callers may notice:

- **`use_verifier` request field removed.** It will be silently ignored if sent.
- **`topk` semantics shifted.** Previously topk candidates came from attention-map peaks in one forward pass. Now topk>1 means the model is sampled multiple times — slower than before, candidates may converge to the same point and get deduped.
- **`topk_values` are placeholders.** Old values were attention scores or verifier probabilities; UI-TARS doesn't surface a comparable confidence.
- **`bbox` field is new** — populated when UI-TARS produces an action-format response with a bounding box. Older callers can keep using the existing `x`/`y` point.

## Confidence guidance

Since `topk_values` is no longer informative, use these heuristics:

- Send `topk=3` and check if all candidates agree to within a small radius. Agreement → high confidence. Wide spread → element is ambiguous or not present.
- Inspect `raw_outputs` for hesitation language or empty-string responses; both are warning signs.

## Example

```bash
IMAGE_B64=$(base64 -i screenshot.png | tr -d '\n')

curl -X POST https://aucvu2pq4vswcj-8888.proxy.runpod.net/ground/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: JIAYI" \
  -d "{\"base64_image\": \"$IMAGE_B64\", \"instruction\": \"Click on the Submit button.\"}"
```

## Health check

```bash
curl https://aucvu2pq4vswcj-8888.proxy.runpod.net/probe/
# {"message": "Omniparser API ready"}
```
