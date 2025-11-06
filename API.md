# RTMDetSwift C API Documentation

RTMDetSwift provides a C API compatible with Unity, matching the YOLOUnity interface design.

## Overview

The API uses a callback-based pattern where you:
1. Initialize the model
2. Register a callback to receive results
3. Send images for inference
4. Results are delivered asynchronously via the callback

## Key Differences from YOLOUnity

- **No class names**: RTMDetSwift returns only class IDs (integers), not class name strings
  - `namesData` is always empty pointer, `namesLength` is always 0
  - Use a lookup table in Unity to convert class IDs to names
- **Same callback signature**: Fully compatible with YOLOUnity Unity delegate
- **Same coordinate system**: All coordinates are returned in the **original image space**
- **Same image format**: RGBA interleaved with Y-axis flip for Unity compatibility

## API Functions

### 1. RegisterRTMDetCallback

Register a callback function to receive detection results.

```c
void RegisterRTMDetCallback(RTMDetCallback callback);
```

**Callback Signature:**
```c
typedef void (*RTMDetCallback)(
    int32_t numDetections,          // Number of detections
    const int32_t* classIndices,    // Class IDs [numDetections]
    const uint8_t* namesData,       // UNUSED - always empty (for YOLOUnity compatibility)
    int32_t namesLength,            // UNUSED - always 0 (for YOLOUnity compatibility)
    const float* scores,            // Confidence scores [numDetections]
    const int32_t* boxes,           // Bounding boxes [numDetections * 4] as [x1,y1,x2,y2,...]
    const int32_t* contourPoints,   // Flattened contour points [contourPointsCount]
    int32_t contourPointsCount,     // Total number of contour coordinate values
    const int32_t* contourIndices,  // Indices to separate contours [contourIndicesCount]
    int32_t contourIndicesCount,    // Number of contour indices
    const int32_t* centroids,       // Mask centroids [numDetections * 2] as [x1,y1,x2,y2,...]
    uint64_t timestamp              // Timestamp in milliseconds
);
```

**Note**: The `namesData` and `namesLength` parameters are **unused** and kept only for Unity/YOLOUnity signature compatibility. RTMDet returns class IDs only - use a lookup table in Unity to get class names from IDs.

**Coordinate System:**
- All coordinates (boxes, contours, centroids) are in **original image space**
- Y-axis follows Unity convention (flipped)
- Coordinates are scaled by the `scaleX` and `scaleY` parameters from RunRTMDet

**Contour Format:**
- `contourPoints`: Flattened array of (x,y) pairs: `[x1,y1,x2,y2,x3,y3,...]`
- `contourIndices`: Structure `[startIdx, endIdx1, endIdx2, ..., -1, nextStartIdx, ...]`
  - First index: where this detection's contours start
  - Following indices: where each contour ends
  - `-1`: separator between detections

Example:
```
Detection 0 has 2 contours with 10 and 5 points
Detection 1 has 1 contour with 8 points

contourIndices = [0, 10, 15, -1, 15, 23, -1]
                  ^   ^   ^   ^   ^   ^   ^
                  |   |   |   |   |   |   separator
                  |   |   |   sep |   det1 contour0 end
                  |   |   det0 contour1 end
                  |   det0 contour0 end
                  det0 starts at 0

contourPoints has 23 values (10 + 5 + 8 points = 23 points * 2 coords = 46 values)
```

### 2. InitializeRTMDet

Initialize the RTMDet model from an ONNX file.

```c
bool InitializeRTMDet(
    const char* modelPath,          // Full path to .onnx model file
    float confidenceThreshold,      // Confidence threshold (e.g., 0.5)
    float iouThreshold              // IoU threshold - IGNORED (kept for API compatibility)
);
```

**Returns:** `true` if initialization succeeded, `false` otherwise.

**Note**: The `iouThreshold` parameter is **ignored** because RTMDet has built-in NMS in the ONNX model. It's kept for API compatibility with YOLOUnity.

**Example:**
```c
bool success = InitializeRTMDet("/path/to/rtmdet-m.onnx", 0.5, 0.5);
// Note: The second 0.5 (iouThreshold) is ignored - RTMDet uses built-in NMS
```

### 3. RunRTMDet

Run inference on a float image (RGBA format, 0-1 range).

```c
void RunRTMDet(
    const float* imageData,         // RGBA image [height * width * 4]
    int width,                      // Image width
    int height,                     // Image height
    uint64_t timestamp,             // Optional timestamp (0 = auto)
    float scaleX,                   // Additional X scaling (default: 1.0)
    float scaleY                    // Additional Y scaling (default: 1.0)
);
```

**Image Format:**
- Interleaved RGBA: `[r,g,b,a,r,g,b,a,...]`
- Values in range `[0.0, 1.0]`
- Y-axis flipped (Unity convention)
- Total size: `width * height * 4` floats

**Scaling:**
- Results are automatically scaled from 640×640 model space to original image size
- `scaleX` and `scaleY` apply additional scaling on top of this

### 4. RunRTMDet_Byte

Run inference on a byte image (RGBA format, 0-255 range).

```c
void RunRTMDet_Byte(
    const uint8_t* imageData,       // RGBA image [height * width * 4]
    int width,                      // Image width
    int height,                     // Image height
    uint64_t timestamp,             // Optional timestamp (0 = auto)
    float scaleX,                   // Additional X scaling (default: 1.0)
    float scaleY                    // Additional Y scaling (default: 1.0)
);
```

**Image Format:**
- Interleaved RGBA: `[r,g,b,a,r,g,b,a,...]`
- Values in range `[0, 255]`
- Y-axis flipped (Unity convention)
- Total size: `width * height * 4` bytes

## Usage Example

```c
#include <RTMDetSwift/RTMDetSwift.h>

// Callback to handle results - matches YOLOUnity signature
void OnDetection(
    int32_t numDetections,
    const int32_t* classIndices,
    const uint8_t* namesData,       // UNUSED - always empty
    int32_t namesLength,            // UNUSED - always 0
    const float* scores,
    const int32_t* boxes,
    const int32_t* contourPoints,
    int32_t contourPointsCount,
    const int32_t* contourIndices,
    int32_t contourIndicesCount,
    const int32_t* centroids,
    uint64_t timestamp
) {
    printf("Detected %d objects\n", numDetections);
    // Note: namesLength will always be 0 - use classIndices with your own lookup table

    for (int i = 0; i < numDetections; i++) {
        printf("Object %d: class=%d, score=%.2f, bbox=[%d,%d,%d,%d], centroid=[%d,%d]\n",
            i,
            classIndices[i],  // Use this ID with your class name lookup table
            scores[i],
            boxes[i*4], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3],
            centroids[i*2], centroids[i*2+1]
        );
    }
}

int main() {
    // Initialize
    if (!InitializeRTMDet("/path/to/rtmdet-m.onnx", 0.5, 0.5)) {
        fprintf(stderr, "Failed to initialize RTMDet\n");
        return 1;
    }

    // Register callback
    RegisterRTMDetCallback(OnDetection);

    // Prepare image (1920x1080 RGBA)
    int width = 1920;
    int height = 1080;
    uint8_t* imageData = malloc(width * height * 4);
    // ... fill imageData ...

    // Run inference
    RunRTMDet_Byte(imageData, width, height, 0, 1.0, 1.0);

    // Results will be delivered via OnDetection callback

    free(imageData);
    return 0;
}
```

## COCO Class IDs

RTMDet returns class IDs from 0-79 for COCO dataset classes:

| ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|
| 0 | person | 1 | bicycle | 2 | car |
| 3 | motorcycle | 4 | airplane | 5 | bus |
| ... | ... | ... | ... | ... | ... |

See `RTMDetSwift/COCO_LABELS.swift` for the complete mapping.

## Thread Safety

- Inference runs asynchronously on a background queue
- Callbacks are invoked on a background thread (not main thread)
- Only one model instance is supported (global state)

## Performance Considerations

- Input images are automatically resized to 640×640
- The model uses CoreML with Neural Engine acceleration
- Mask processing uses SIMD optimization (NEON on ARM)
- Y-flip conversion adds overhead; pre-flip images if possible
- **No NMS overhead**: RTMDet has built-in NMS, so no additional processing needed
