//
//  RTMDetBridge.swift
//  RTMDetSwift
//
//  C API bridge for Unity integration - matches YOLOUnity interface
//

import Foundation
import UIKit

// Callback type - matches YOLOUnity signature for Unity compatibility
public typealias RTMDetCallback = @convention(c) (
    Int32,                          // number of detections
    UnsafePointer<Int32>,           // classIndex (length = numDetections)
    UnsafePointer<UInt8>, Int32,    // names data, names length (UNUSED - always empty for RTMDet)
    UnsafePointer<Float>,           // scores (length = numDetections)
    UnsafePointer<Int32>,           // boxes (length = numDetections * 4)
    UnsafePointer<Int32>, Int32,    // contour points, count
    UnsafePointer<Int32>, Int32,    // contour indices, count
    UnsafePointer<Int32>,           // centroids (length = numDetections * 2)
    UInt64                          // timestamp
) -> Void

// Global variables
private var inferencer: RTMDetInferencer? = nil
private var rtmdetCallback: RTMDetCallback? = nil
private var confidenceThreshold: Float = 0.5
private var iouThreshold: Float = 0.5

// Serial queue to prevent concurrent inference overload (memory spike prevention)
private let inferenceQueue = DispatchQueue(label: "com.rtmdet.inference", qos: .userInitiated)

// Register the callback
@_cdecl("RegisterRTMDetCallback")
public func RegisterRTMDetCallback(callback: @escaping RTMDetCallback) {
    rtmdetCallback = callback
}

// Initialize RTMDet
@_cdecl("InitializeRTMDet")
public func InitializeRTMDet(
    modelPath: UnsafePointer<CChar>,
    confidenceThreshold: Float,
    iouThreshold: Float
) -> Bool {
    let pathOrName = String(cString: modelPath)

    // Resolve model path: check if it's a full path or just a model name
    let fullPath: String
    if pathOrName.hasPrefix("/") || pathOrName.hasPrefix("file://") {
        // Already a full path
        fullPath = pathOrName.replacingOccurrences(of: "file://", with: "")
    } else {
        // Model name - look in framework bundle first, then main bundle
        let modelName = pathOrName.replacingOccurrences(of: ".onnx", with: "")

        // Get the framework bundle (RTMDetSwift.framework)
        let frameworkBundle = Bundle(for: RTMDetInferencer.self)

        // Try framework bundle first (where the model actually is)
        if let bundlePath = frameworkBundle.path(forResource: modelName, ofType: "onnx") {
            fullPath = bundlePath
            NSLog("Found model in framework bundle: \(bundlePath)")
        } else if let bundlePath = frameworkBundle.path(forResource: modelName, ofType: "mlpackage") {
            fullPath = bundlePath
            NSLog("Found model in framework bundle: \(bundlePath)")
        }
        // Then try main app bundle
        else if let bundlePath = Bundle.main.path(forResource: modelName, ofType: "onnx") {
            fullPath = bundlePath
            NSLog("Found model in main bundle: \(bundlePath)")
        } else if let bundlePath = Bundle.main.path(forResource: modelName, ofType: "mlpackage") {
            fullPath = bundlePath
            NSLog("Found model in main bundle: \(bundlePath)")
        } else {
            // Try looking in Documents directory
            let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
            let documentsModelPath = (documentsPath as NSString).appendingPathComponent(pathOrName)

            if FileManager.default.fileExists(atPath: documentsModelPath) {
                fullPath = documentsModelPath
                NSLog("Found model in Documents: \(documentsModelPath)")
            } else {
                NSLog("Error: Model not found. Tried:")
                NSLog("  - Framework bundle (\(frameworkBundle.bundlePath)): \(modelName).onnx")
                NSLog("  - Main bundle: \(modelName).onnx")
                NSLog("  - Documents: \(documentsModelPath)")
                return false
            }
        }
    }

    guard let newInferencer = RTMDetInferencer(modelPath: fullPath, inputWidth: 640, inputHeight: 640) else {
        NSLog("Error: Failed to initialize RTMDetInferencer with model at \(fullPath)")
        return false
    }

    inferencer = newInferencer
    RTMDetBridge.confidenceThreshold = confidenceThreshold
    RTMDetBridge.iouThreshold = iouThreshold

    NSLog("Initialized RTMDet with model=\(fullPath), confidence=\(confidenceThreshold), iou=\(iouThreshold)")
    return true
}

// Run RTMDet on float image data
@_cdecl("RunRTMDet")
public func RunRTMDet(
    imageData: UnsafePointer<Float>,
    width: Int,
    height: Int,
    timestamp: UInt64 = 0,
    scaleX: Float = 1.0,
    scaleY: Float = 1.0
) {
    guard let inferencer = inferencer else {
        NSLog("Error: RTMDetInferencer not initialized.")
        return
    }

    // Optimized path: Skip UIImage creation entirely
    autoreleasepool {
        // Resize directly (RGBA interleaved format)
        guard let resizedRGBA = resizeFloatImage(
            data: imageData,
            sourceWidth: width,
            sourceHeight: height,
            targetWidth: 640,
            targetHeight: 640
        ) else {
            NSLog("Error: Failed to resize float image data.")
            return
        }

        // Convert RGBA interleaved → RGB CHW (model input format)
        let inputCHW = convertRGBAToCHW(rgba: resizedRGBA, width: 640, height: 640)

        // Run inference directly on float array (skip UIImage creation!)
        runDetectionFromFloatArray(
            inferencer: inferencer,
            inputData: inputCHW,
            originalWidth: width,
            originalHeight: height,
            timestamp: timestamp,
            scaleX: scaleX,
            scaleY: scaleY
        )
    }
}

// Run RTMDet on byte image data (RGB format)
@_cdecl("RunRTMDet_Byte")
public func RunRTMDet_Byte(
    imageData: UnsafePointer<UInt8>,
    width: Int,
    height: Int,
    timestamp: UInt64 = 0,
    scaleX: Float = 1.0,
    scaleY: Float = 1.0
) {
    guard let inferencer = inferencer else {
        NSLog("Error: RTMDetInferencer not initialized.")
        return
    }

    // Optimized path: Skip UIImage creation entirely
    autoreleasepool {
        // Resize directly (RGB bytes)
        guard let resizedBytes = resizeByteImage(
            data: imageData,
            sourceWidth: width,
            sourceHeight: height,
            targetWidth: 640,
            targetHeight: 640
        ) else {
            NSLog("Error: Failed to resize byte image data.")
            return
        }

        // Convert bytes to normalized floats in CHW format
        let inputCHW = convertBytesToCHW(bytes: resizedBytes, width: 640, height: 640)

        // Run inference directly on float array (skip UIImage creation!)
        runDetectionFromFloatArray(
            inferencer: inferencer,
            inputData: inputCHW,
            originalWidth: width,
            originalHeight: height,
            timestamp: timestamp,
            scaleX: scaleX,
            scaleY: scaleY
        )
    }
}

// MARK: - Helper Functions

/// Optimized detection path: run inference directly on preprocessed float array (skip UIImage)
private func runDetectionFromFloatArray(
    inferencer: RTMDetInferencer,
    inputData: [Float],
    originalWidth: Int,
    originalHeight: Int,
    timestamp: UInt64,
    scaleX: Float,
    scaleY: Float
) {
    // Use serial queue to prevent concurrent inference memory spikes
    inferenceQueue.async {
        autoreleasepool {  // Critical: matches YOLOUnity memory management
            let ts = timestamp == 0 ? getCurrentTimestamp() : timestamp

            guard let detections = inferencer.detectFromFloatArray(
                inputData: inputData,
                confidenceThreshold: RTMDetBridge.confidenceThreshold,
                iouThreshold: RTMDetBridge.iouThreshold
            ) else {
                NSLog("Error: Detection failed.")
                return
            }

            processAndSendDetections(
                detections: detections,
                originalWidth: originalWidth,
                originalHeight: originalHeight,
                timestamp: ts,
                scaleX: scaleX,
                scaleY: scaleY
            )
        }
    }
}

/// Legacy detection path: for UIImage-based inference
private func runDetection(
    inferencer: RTMDetInferencer,
    image: UIImage,
    originalWidth: Int,
    originalHeight: Int,
    timestamp: UInt64,
    scaleX: Float,
    scaleY: Float
) {
    // Use serial queue to prevent concurrent inference memory spikes
    inferenceQueue.async {
        autoreleasepool {  // Critical: matches YOLOUnity memory management
            let ts = timestamp == 0 ? getCurrentTimestamp() : timestamp

            guard let detections = inferencer.detect(
                image: image,
                confidenceThreshold: RTMDetBridge.confidenceThreshold,
                iouThreshold: RTMDetBridge.iouThreshold
            ) else {
                NSLog("Error: Detection failed.")
                return
            }

            processAndSendDetections(
                detections: detections,
                originalWidth: originalWidth,
                originalHeight: originalHeight,
                timestamp: ts,
                scaleX: scaleX,
                scaleY: scaleY
            )
        }
    }
}

/// Common code to process detections and send to Unity callback
private func processAndSendDetections(
    detections: [Detection],
    originalWidth: Int,
    originalHeight: Int,
    timestamp: UInt64,
    scaleX: Float,
    scaleY: Float
) {
    guard let callback = rtmdetCallback else {
        NSLog("Warning: No callback registered.")
        return
    }

    // Convert detections to callback format
    // All coordinates are in 640x640 model space, need to scale to ACTUAL original image space
    // CRITICAL: Use the original dimensions passed from Unity, NOT image.size (which is 640x640)
    let modelSize: Float = 640.0

    // Scale factors: from 640x640 model space to original image space
    let toOriginalX = Float(originalWidth) / modelSize
    let toOriginalY = Float(originalHeight) / modelSize

    // Apply additional user-specified scaling
    let finalScaleX = toOriginalX * scaleX
    let finalScaleY = toOriginalY * scaleY

    // Prepare data arrays
    let classIndices = detections.map { Int32($0.classId) }
    let scores = detections.map { $0.confidence }

    // Image boundaries for clamping
    let maxX = Int32(originalWidth - 1)
    let maxY = Int32(originalHeight - 1)

    // Boxes: [x1, y1, x2, y2] for each detection in original image space
    // Clamp to valid image coordinates [0, width-1] and [0, height-1]
    let boxes = detections.flatMap { detection -> [Int32] in
        let x1 = max(0, min(maxX, Int32(detection.bbox.x1 * finalScaleX)))
        let y1 = max(0, min(maxY, Int32(detection.bbox.y1 * finalScaleY)))
        let x2 = max(0, min(maxX, Int32(detection.bbox.x2 * finalScaleX)))
        let y2 = max(0, min(maxY, Int32(detection.bbox.y2 * finalScaleY)))
        return [x1, y1, x2, y2]
    }

    // Flatten all contour points and scale to original image space
    // Clamp to valid image coordinates
    let contourPoints = detections.flatMap { detection -> [Int32] in
        detection.contours.flatMap { contour -> [Int32] in
            stride(from: 0, to: contour.count, by: 2).flatMap { i -> [Int32] in
                let x = max(0, min(maxX, Int32(contour[i] * finalScaleX)))
                let y = max(0, min(maxY, Int32(contour[i + 1] * finalScaleY)))
                return [x, y]
            }
        }
    }

    // Contour indices: [startIdx, endIdx, endIdx, ..., -1] for each detection
    // Format: [det0_start, det0_contour1_end, det0_contour2_end, ..., -1, det1_start, ...]
    var contourIndices: [Int32] = []
    var currentIndex: Int32 = 0
    for detection in detections {
        contourIndices.append(currentIndex)
        for contour in detection.contours {
            let pointCount = Int32(contour.count / 2)  // x,y pairs
            currentIndex += pointCount
            contourIndices.append(currentIndex)
        }
        contourIndices.append(-1)  // Separator between detections
    }

    // Centroids: [x1, y1, x2, y2, ...] in original image space
    // Clamp to valid image coordinates
    let centroids = detections.flatMap { detection -> [Int32] in
        let x = max(0, min(maxX, Int32(detection.centroid.0 * finalScaleX)))
        let y = max(0, min(maxY, Int32(detection.centroid.1 * finalScaleY)))
        return [x, y]
    }

    // Empty name data for Unity signature compatibility
    // RTMDet returns class IDs only - Unity should use a lookup table for names
    let emptyNames: [UInt8] = []

    // Call the callback with all the data
    classIndices.withUnsafeBufferPointer { classPtr in
        emptyNames.withUnsafeBufferPointer { namesPtr in
            scores.withUnsafeBufferPointer { scoresPtr in
                boxes.withUnsafeBufferPointer { boxesPtr in
                    contourPoints.withUnsafeBufferPointer { pointsPtr in
                        contourIndices.withUnsafeBufferPointer { indicesPtr in
                            centroids.withUnsafeBufferPointer { centroidPtr in
                                // Pass empty pointer for names to maintain Unity compatibility
                                let namesPointer = namesPtr.baseAddress ?? UnsafePointer<UInt8>(bitPattern: 1)!
                                callback(
                                    Int32(detections.count),
                                    classPtr.baseAddress!,
                                    namesPointer,
                                    Int32(0),  // names length = 0
                                    scoresPtr.baseAddress!,
                                    boxesPtr.baseAddress!,
                                    pointsPtr.baseAddress!,
                                    Int32(contourPoints.count),
                                    indicesPtr.baseAddress!,
                                            Int32(contourIndices.count),
                                            centroidPtr.baseAddress!,
                                            ts
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }  // autoreleasepool
    }
}

// Convert float array (RGBA interleaved, 0-1 range) to UIImage
// Matches YOLOUnity format: data is [r,g,b,a,r,g,b,a,...] with Y-flip
// Optimized with parallel processing like YOLOUnity
private func floatArrayToUIImage(data: UnsafePointer<Float>, width: Int, height: Int) -> UIImage? {
    let bytesPerPixel = 4
    let byteCount = width * height * bytesPerPixel
    var pixels = [UInt8](repeating: 0, count: byteCount)

    let width4 = width * 4
    let lookup = (0...255).map { UInt8($0) }  // Pre-calculate UInt8 conversions (YOLOUnity optimization)

    // Parallel processing across rows (matches YOLOUnity approach)
    DispatchQueue.concurrentPerform(iterations: height) { y in
        let sourceY = height - 1 - y  // Flip Y axis
        let sourceStart = sourceY * width4
        let destStart = y * width4

        for x in stride(from: 0, to: width4, by: 4) {
            let sourceIndex = sourceStart + x
            let destIndex = destStart + x

            // Use lookup table for faster conversion (YOLOUnity technique)
            let r = Int(data[sourceIndex] * 255.0)
            let g = Int(data[sourceIndex + 1] * 255.0)
            let b = Int(data[sourceIndex + 2] * 255.0)
            let a = Int(data[sourceIndex + 3] * 255.0)

            pixels[destIndex] = lookup[max(0, min(255, r))]
            pixels[destIndex + 1] = lookup[max(0, min(255, g))]
            pixels[destIndex + 2] = lookup[max(0, min(255, b))]
            pixels[destIndex + 3] = lookup[max(0, min(255, a))]
        }
    }

    return createUIImage(from: pixels, width: width, height: height)
}

// Convert byte array (RGBA interleaved) to UIImage
// Matches YOLOUnity format: data is [r,g,b,a,r,g,b,a,...] with Y-flip
// Optimized with parallel processing and memcpy like YOLOUnity
private func byteArrayToUIImage(data: UnsafePointer<UInt8>, width: Int, height: Int) -> UIImage? {
    let bytesPerPixel = 4
    let byteCount = width * height * bytesPerPixel
    var pixels = [UInt8](repeating: 0, count: byteCount)

    let rowBytes = width * bytesPerPixel

    // Parallel processing across rows with memcpy (matches YOLOUnity approach)
    DispatchQueue.concurrentPerform(iterations: height) { y in
        let sourceY = height - 1 - y  // Flip Y axis
        let sourceStart = sourceY * rowBytes
        let destStart = y * rowBytes

        // Direct memory copy for each row (fastest method)
        memcpy(&pixels[destStart], data + sourceStart, rowBytes)
    }

    return createUIImage(from: pixels, width: width, height: height)
}

// Create UIImage from RGBA byte array
private func createUIImage(from pixels: [UInt8], width: Int, height: Int) -> UIImage? {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let context = CGContext(
        data: UnsafeMutableRawPointer(mutating: pixels),
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
    ) else {
        return nil
    }

    guard let cgImage = context.makeImage() else {
        return nil
    }

    return UIImage(cgImage: cgImage)
}

// Get current timestamp in milliseconds (matches YOLOUnity)
private func getCurrentTimestamp() -> UInt64 {
    return UInt64(Date().timeIntervalSince1970 * 1000)
}

// MARK: - Memory-Optimized Image Conversion

/// Convert RGBA interleaved float array to RGB CHW (planar) format for model input
/// Input: [R0, G0, B0, A0, R1, G1, B1, A1, ...] - RGBA interleaved
/// Output: [R0, R1, R2, ..., G0, G1, G2, ..., B0, B1, B2, ...] - RGB planar (CHW)
private func convertRGBAToCHW(rgba: [Float], width: Int, height: Int) -> [Float] {
    let pixelCount = width * height
    var chw = [Float](repeating: 0, count: 3 * pixelCount)

    // Parallel conversion for performance
    DispatchQueue.concurrentPerform(iterations: height) { y in
        for x in 0..<width {
            let pixelIndex = y * width + x
            let rgbaIndex = pixelIndex * 4

            // R channel
            chw[pixelIndex] = rgba[rgbaIndex]
            // G channel
            chw[pixelCount + pixelIndex] = rgba[rgbaIndex + 1]
            // B channel
            chw[2 * pixelCount + pixelIndex] = rgba[rgbaIndex + 2]
            // Skip alpha channel
        }
    }

    return chw
}

/// Convert RGB byte array to RGB CHW float array (normalized to [0, 1])
/// Input: [R0, G0, B0, R1, G1, B1, ...] - RGB bytes [0-255]
/// Output: [R0, R1, R2, ..., G0, G1, G2, ..., B0, B1, B2, ...] - RGB planar floats [0-1]
private func convertBytesToCHW(bytes: [UInt8], width: Int, height: Int) -> [Float] {
    let pixelCount = width * height
    var chw = [Float](repeating: 0, count: 3 * pixelCount)

    // Parallel conversion with normalization
    DispatchQueue.concurrentPerform(iterations: height) { y in
        for x in 0..<width {
            let pixelIndex = y * width + x
            let rgbIndex = pixelIndex * 3

            // Normalize from [0, 255] to [0, 1]
            chw[pixelIndex] = Float(bytes[rgbIndex]) / 255.0  // R
            chw[pixelCount + pixelIndex] = Float(bytes[rgbIndex + 1]) / 255.0  // G
            chw[2 * pixelCount + pixelIndex] = Float(bytes[rgbIndex + 2]) / 255.0  // B
        }
    }

    return chw
}

/// Resize float image data directly without creating intermediate UIImages
/// This saves significant memory by avoiding large UIImage allocations
private func resizeFloatImage(
    data: UnsafePointer<Float>,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int
) -> [Float]? {
    let sourceSize = sourceWidth * sourceHeight * 4  // RGBA
    let targetSize = targetWidth * targetHeight * 4

    var resized = [Float](repeating: 0, count: targetSize)

    // Calculate scale factors
    let scaleX = Float(sourceWidth) / Float(targetWidth)
    let scaleY = Float(sourceHeight) / Float(targetHeight)

    // Bilinear interpolation resize (parallel processing)
    DispatchQueue.concurrentPerform(iterations: targetHeight) { y in
        let sourceY = targetHeight - 1 - y  // Y-flip for Unity
        let sy = Float(sourceY) * scaleY
        let sy1 = Int(sy)
        let sy2 = min(sy1 + 1, sourceHeight - 1)
        let fy = sy - Float(sy1)

        for x in 0..<targetWidth {
            let sx = Float(x) * scaleX
            let sx1 = Int(sx)
            let sx2 = min(sx1 + 1, sourceWidth - 1)
            let fx = sx - Float(sx1)

            let destIdx = y * targetWidth * 4 + x * 4

            // Bilinear interpolation for each channel
            for c in 0..<4 {
                let p1 = data[sy1 * sourceWidth * 4 + sx1 * 4 + c]
                let p2 = data[sy1 * sourceWidth * 4 + sx2 * 4 + c]
                let p3 = data[sy2 * sourceWidth * 4 + sx1 * 4 + c]
                let p4 = data[sy2 * sourceWidth * 4 + sx2 * 4 + c]

                let top = p1 * (1 - fx) + p2 * fx
                let bottom = p3 * (1 - fx) + p4 * fx
                resized[destIdx + c] = top * (1 - fy) + bottom * fy
            }
        }
    }

    return resized
}

/// Convert float array directly to UIImage (no intermediate conversions)
/// Only used for already-resized 640x640 data
private func floatArrayToUIImageDirect(data: [Float], width: Int, height: Int) -> UIImage? {
    let pixelCount = width * height * 4
    var pixels = [UInt8](repeating: 0, count: pixelCount)

    // Fast conversion without lookup table (data is already small)
    for i in 0..<pixelCount {
        pixels[i] = UInt8(max(0, min(255, data[i] * 255)))
    }

    return createUIImage(from: pixels, width: width, height: height)
}

/// Resize byte image data directly without creating intermediate UIImages
private func resizeByteImage(
    data: UnsafePointer<UInt8>,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int
) -> [UInt8]? {
    let targetSize = targetWidth * targetHeight * 4
    var resized = [UInt8](repeating: 0, count: targetSize)

    // Calculate scale factors
    let scaleX = Float(sourceWidth) / Float(targetWidth)
    let scaleY = Float(sourceHeight) / Float(targetHeight)

    // Bilinear interpolation resize (parallel processing)
    DispatchQueue.concurrentPerform(iterations: targetHeight) { y in
        let sourceY = targetHeight - 1 - y  // Y-flip for Unity
        let sy = Float(sourceY) * scaleY
        let sy1 = Int(sy)
        let sy2 = min(sy1 + 1, sourceHeight - 1)
        let fy = sy - Float(sy1)

        for x in 0..<targetWidth {
            let sx = Float(x) * scaleX
            let sx1 = Int(sx)
            let sx2 = min(sx1 + 1, sourceWidth - 1)
            let fx = sx - Float(sx1)

            let destIdx = y * targetWidth * 4 + x * 4

            // Bilinear interpolation for each channel
            for c in 0..<4 {
                let p1 = Float(data[sy1 * sourceWidth * 4 + sx1 * 4 + c])
                let p2 = Float(data[sy1 * sourceWidth * 4 + sx2 * 4 + c])
                let p3 = Float(data[sy2 * sourceWidth * 4 + sx1 * 4 + c])
                let p4 = Float(data[sy2 * sourceWidth * 4 + sx2 * 4 + c])

                let top = p1 * (1 - fx) + p2 * fx
                let bottom = p3 * (1 - fx) + p4 * fx
                resized[destIdx + c] = UInt8(top * (1 - fy) + bottom * fy)
            }
        }
    }

    return resized
}

/// Convert byte array directly to UIImage (no intermediate conversions)
/// Only used for already-resized 640x640 data
private func byteArrayToUIImageDirect(data: [UInt8], width: Int, height: Int) -> UIImage? {
    return createUIImage(from: data, width: width, height: height)
}

// Namespace for static variables
private enum RTMDetBridge {
    static var confidenceThreshold: Float = 0.5
    static var iouThreshold: Float = 0.5
}
