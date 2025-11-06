//
//  RTMDetBridge.swift
//  RTMDetSwift
//
//  C API bridge for Unity integration - matches YOLOUnity interface
//

import Foundation
import UIKit

// Callback type - matches YOLOUnity but WITHOUT class names (only IDs)
public typealias RTMDetCallback = @convention(c) (
    Int32,                          // number of detections
    UnsafePointer<Int32>,           // classIndex (length = numDetections)
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
    let path = String(cString: modelPath)

    guard let newInferencer = RTMDetInferencer(modelPath: path, inputWidth: 640, inputHeight: 640) else {
        NSLog("Error: Failed to initialize RTMDetInferencer with model at \(path)")
        return false
    }

    inferencer = newInferencer
    RTMDetBridge.confidenceThreshold = confidenceThreshold
    RTMDetBridge.iouThreshold = iouThreshold

    NSLog("Initialized RTMDet with model=\(path), confidence=\(confidenceThreshold), iou=\(iouThreshold)")
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

    guard let image = floatArrayToUIImage(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert float array to UIImage.")
        return
    }

    runDetection(inferencer: inferencer, image: image, timestamp: timestamp, scaleX: scaleX, scaleY: scaleY)
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

    guard let image = byteArrayToUIImage(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert byte array to UIImage.")
        return
    }

    runDetection(inferencer: inferencer, image: image, timestamp: timestamp, scaleX: scaleX, scaleY: scaleY)
}

// MARK: - Helper Functions

private func runDetection(
    inferencer: RTMDetInferencer,
    image: UIImage,
    timestamp: UInt64,
    scaleX: Float,
    scaleY: Float
) {
    DispatchQueue.global(qos: .userInitiated).async {
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

            guard let callback = rtmdetCallback else {
                NSLog("Warning: No callback registered.")
                return
            }

                // Convert detections to callback format
            // All coordinates are in 640x640 space, need to scale to original image space
            let originalWidth = Float(image.size.width)
            let originalHeight = Float(image.size.height)
            let modelSize: Float = 640.0

            // Scale factors: from 640x640 model space to original image space
            let toOriginalX = originalWidth / modelSize
            let toOriginalY = originalHeight / modelSize

            // Apply additional user-specified scaling
            let finalScaleX = toOriginalX * scaleX
            let finalScaleY = toOriginalY * scaleY

            // Prepare data arrays
            let classIndices = detections.map { Int32($0.classId) }
            let scores = detections.map { $0.confidence }

            // Boxes: [x1, y1, x2, y2] for each detection in original image space
            let boxes = detections.flatMap { detection -> [Int32] in
                let x1 = Int32(detection.bbox.x1 * finalScaleX)
                let y1 = Int32(detection.bbox.y1 * finalScaleY)
                let x2 = Int32(detection.bbox.x2 * finalScaleX)
                let y2 = Int32(detection.bbox.y2 * finalScaleY)
                return [x1, y1, x2, y2]
            }

            // Flatten all contour points and scale to original image space
            let contourPoints = detections.flatMap { detection -> [Int32] in
                detection.contours.flatMap { contour -> [Int32] in
                    stride(from: 0, to: contour.count, by: 2).flatMap { i -> [Int32] in
                        let x = Int32(contour[i] * finalScaleX)
                        let y = Int32(contour[i + 1] * finalScaleY)
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
            let centroids = detections.flatMap { detection -> [Int32] in
                let x = Int32(detection.centroid.0 * finalScaleX)
                let y = Int32(detection.centroid.1 * finalScaleY)
                return [x, y]
            }

            // Call the callback with all the data
            classIndices.withUnsafeBufferPointer { classPtr in
                scores.withUnsafeBufferPointer { scoresPtr in
                    boxes.withUnsafeBufferPointer { boxesPtr in
                        contourPoints.withUnsafeBufferPointer { pointsPtr in
                            contourIndices.withUnsafeBufferPointer { indicesPtr in
                                centroids.withUnsafeBufferPointer { centroidPtr in
                                    callback(
                                        Int32(detections.count),
                                        classPtr.baseAddress!,
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

// Namespace for static variables
private enum RTMDetBridge {
    static var confidenceThreshold: Float = 0.5
    static var iouThreshold: Float = 0.5
}
