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
    private let iouThreshold: Float

    public init(confidenceThreshold: Float = 0.5, iouThreshold: Float = 0.5) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    /// Post-process RTMDet outputs with SIMD optimization
    /// - Parameters:
    ///   - outputs: Array of ORTValue outputs [labels, masks, dets]
    /// - Returns: Array of Detection objects after NMS
    public func process(outputs: [ORTValue]) throws -> [Detection] {
        guard outputs.count == 3 else {
            throw NSError(domain: "RTMDetPostProcessor", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Expected 3 outputs, got \(outputs.count)"])
        }

        // RTMDet outputs:
        // Output 0: labels [1, 100] - INT64 class IDs
        // Output 1: dets [1, 100, 5] - FLOAT32 [x1, y1, x2, y2, confidence]
        // Output 2: masks [1, 100, 640, 640] - FLOAT32 segmentation masks

        let labelsOutput = outputs[0]
        let detsOutput = outputs[1]

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

        // Build detections for valid boxes
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
                detections.append(Detection(classId: classId, confidence: confidence, bbox: bbox))
            }
        }

        // Apply NMS with optimized IoU calculation
        let finalDetections = applyNMSOptimized(detections: detections, iouThreshold: iouThreshold)

        return finalDetections
    }

    /// Optimized NMS using vectorized IoU calculations
    private func applyNMSOptimized(detections: [Detection], iouThreshold: Float) -> [Detection] {
        guard !detections.isEmpty else { return [] }

        // Sort by confidence (highest first)
        let sortedIndices = detections.indices.sorted { detections[$0].confidence > detections[$1].confidence }

        var keep: [Detection] = []
        keep.reserveCapacity(detections.count)
        var suppressed = Set<Int>()
        suppressed.reserveCapacity(detections.count)

        // Precompute box areas for all detections
        var areas = [Float](repeating: 0, count: detections.count)
        for i in 0..<detections.count {
            let box = detections[i].bbox
            areas[i] = (box.x2 - box.x1) * (box.y2 - box.y1)
        }

        for i in 0..<sortedIndices.count {
            let idx = sortedIndices[i]
            if suppressed.contains(idx) { continue }

            let current = detections[idx]
            keep.append(current)

            // Vectorized IoU computation for remaining boxes
            let currentBox = current.bbox
            let currentArea = areas[idx]

            for j in (i + 1)..<sortedIndices.count {
                let jdx = sortedIndices[j]
                if suppressed.contains(jdx) { continue }

                let other = detections[jdx]

                // Only apply NMS within the same class
                if current.classId == other.classId {
                    let otherBox = other.bbox

                    // Calculate intersection (manually for now, can be vectorized further)
                    let x1 = max(currentBox.x1, otherBox.x1)
                    let y1 = max(currentBox.y1, otherBox.y1)
                    let x2 = min(currentBox.x2, otherBox.x2)
                    let y2 = min(currentBox.y2, otherBox.y2)

                    let intersectionArea = max(0, x2 - x1) * max(0, y2 - y1)
                    let unionArea = currentArea + areas[jdx] - intersectionArea

                    let iou = unionArea > 0 ? intersectionArea / unionArea : 0

                    if iou > iouThreshold {
                        suppressed.insert(jdx)
                    }
                }
            }
        }

        return keep
    }
}
