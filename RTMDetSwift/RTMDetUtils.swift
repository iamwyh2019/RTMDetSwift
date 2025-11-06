//
//  RTMDetUtils.swift
//  RTMDetSwift
//
//  Utility functions for RTMDet post-processing
//

import Foundation
import Accelerate

public typealias BoundingBox = (x1: Float, y1: Float, x2: Float, y2: Float)

public struct Detection {
    public let classId: Int
    public let confidence: Float
    public let bbox: BoundingBox        // Bounding box in 640x640 coordinate space
    public let mask: [Float]            // 640x640 segmentation mask (values in [0,1])
    public let contours: [[Float]]      // Array of contours in 640x640 space, each contour is [x1,y1,x2,y2,...]
    public let centroid: (Float, Float) // Mask centroid (x, y) in 640x640 space

    public init(classId: Int, confidence: Float, bbox: BoundingBox, mask: [Float], contours: [[Float]], centroid: (Float, Float)) {
        self.classId = classId
        self.confidence = confidence
        self.bbox = bbox
        self.mask = mask
        self.contours = contours
        self.centroid = centroid
    }
}

// Calculate IoU (Intersection over Union) for two bounding boxes
func calculateIoU(box1: BoundingBox, box2: BoundingBox) -> Float {
    let x1 = max(box1.x1, box2.x1)
    let y1 = max(box1.y1, box2.y1)
    let x2 = min(box1.x2, box2.x2)
    let y2 = min(box1.y2, box2.y2)

    let intersectionArea = max(0, x2 - x1) * max(0, y2 - y1)

    let box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    let box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

    let unionArea = box1Area + box2Area - intersectionArea

    return unionArea > 0 ? intersectionArea / unionArea : 0
}

// Non-Maximum Suppression
public func applyNMS(detections: [Detection], iouThreshold: Float) -> [Detection] {
    guard !detections.isEmpty else { return [] }

    // Sort by confidence (highest first)
    let sorted = detections.sorted { $0.confidence > $1.confidence }

    var keep: [Detection] = []
    var suppressed = Set<Int>()

    for i in 0..<sorted.count {
        if suppressed.contains(i) { continue }

        let current = sorted[i]
        keep.append(current)

        // Suppress overlapping boxes
        for j in (i + 1)..<sorted.count {
            if suppressed.contains(j) { continue }

            let other = sorted[j]

            // Only apply NMS within the same class
            if current.classId == other.classId {
                let iou = calculateIoU(box1: current.bbox, box2: other.bbox)
                if iou > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }
    }

    return keep
}
