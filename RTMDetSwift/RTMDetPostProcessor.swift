//
//  RTMDetPostProcessor.swift
//  RTMDetSwift
//
//  SIMD-optimized post-processing for RTMDet outputs
//

import Foundation
import onnxruntime_objc
import Accelerate

public class RTMDetPostProcessor {
    private let confidenceThreshold: Float
    private let iouThreshold: Float  // Unused - RTMDet has built-in NMS

    public init(confidenceThreshold: Float = 0.5, iouThreshold: Float = 0.5) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold  // Kept for API compatibility but unused
    }

    /// Post-process RTMDet outputs with SIMD optimization
    /// - Parameters:
    ///   - outputs: Dictionary of output names to ORTValue
    /// - Returns: Array of Detection objects after confidence filtering
    /// - Note: RTMDet model has built-in NMS, so no additional NMS is applied
    public func process(outputs: [String: ORTValue]) throws -> [Detection] {
        // RTMDet outputs by name:
        // "labels": [1, 100] - INT64 class IDs
        // "dets": [1, 100, 5] - FLOAT32 [x1, y1, x2, y2, confidence]
        // "masks": [1, 100, 640, 640] - FLOAT32 segmentation masks

        guard let labelsOutput = outputs["labels"] else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing 'labels' output"])
        }

        guard let detsOutput = outputs["dets"] else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing 'dets' output"])
        }

        guard let masksOutput = outputs["masks"] else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing 'masks' output"])
        }

        // Get labels data (INT64)
        guard let labelsData = try? labelsOutput.tensorData() as Data else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get labels data"])
        }

        // Get detections data (FLOAT32)
        guard let detsData = try? detsOutput.tensorData() as Data else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get detections data"])
        }

        // Get masks data (FLOAT32)
        guard let masksData = try? masksOutput.tensorData() as Data else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get masks data"])
        }

        // Parse data using SIMD-friendly operations
        let numDetections = 100
        // Parse labels - correctly read as INT64
        let labels = labelsData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Int] in
            let typedPointer = ptr.bindMemory(to: Int64.self)
            return (0..<numDetections).map { i in Int(typedPointer[i]) }
        }

        let dets = detsData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> UnsafePointer<Float> in
            ptr.bindMemory(to: Float.self).baseAddress!
        }

        let masks = masksData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> UnsafePointer<Float> in
            ptr.bindMemory(to: Float.self).baseAddress!
        }

        // Filter detections by confidence threshold using vectorized operations
        var detections: [Detection] = []
        detections.reserveCapacity(numDetections)

        // Extract confidence scores for vectorized comparison
        var confidences = [Float](repeating: 0, count: numDetections)
        for i in 0..<numDetections {
            confidences[i] = dets[i * 5 + 4]
        }

        // Vectorized threshold comparison
        var threshold = confidenceThreshold
        var thresholdArray = [Float](repeating: threshold, count: numDetections)
        var comparisonResult = [Float](repeating: 0, count: numDetections)

        // Compare: result[i] = (confidence[i] >= threshold) ? 1.0 : 0.0
        vDSP_vthres(confidences, 1, &threshold, &comparisonResult, 1, vDSP_Length(numDetections))

        // Build detections for valid boxes with masks
        let maskSize = 640
        let maskArea = maskSize * maskSize

        for i in 0..<numDetections {
            if confidences[i] >= confidenceThreshold {
                let offset = i * 5
                let x1 = dets[offset + 0]
                let y1 = dets[offset + 1]
                let x2 = dets[offset + 2]
                let y2 = dets[offset + 3]
                let confidence = dets[offset + 4]

                let classId = labels[i]
                let bbox = BoundingBox(x1: x1, y1: y1, x2: x2, y2: y2)

                // Extract mask for this detection (640x640)
                let maskOffset = i * maskArea
                let maskPointer = masks.advanced(by: maskOffset)

                // Convert mask to Array for storage
                let maskArray = Array(UnsafeBufferPointer(start: maskPointer, count: maskArea))

                // Extract contours and centroid using OpenCV
                let contoursDict = OpenCVWrapper.findContours(
                    maskPointer,
                    width: Int32(maskSize),
                    height: Int32(maskSize)
                )

                // Parse contours from NSDictionary
                let contoursNS = contoursDict["contours"] as? [[NSNumber]] ?? []
                let contours = contoursNS.map { contour in
                    contour.map { $0.floatValue }
                }

                // Parse centroid from NSDictionary
                let centroidNS = contoursDict["centroid"] as? [NSNumber] ?? [0, 0]
                let centroid = (centroidNS[0].floatValue, centroidNS[1].floatValue)

                detections.append(Detection(
                    classId: classId,
                    confidence: confidence,
                    bbox: bbox,
                    mask: maskArray,
                    contours: contours,
                    centroid: centroid
                ))
            }
        }

        // RTMDet model has built-in NMS, so we just return filtered detections
        // No additional NMS needed - confidence filtering doesn't create new overlaps
        return detections
    }
}
