//
//  RTMDetSwiftTests.swift
//  RTMDetSwiftTests
//
//  Created by 吴宇恒 on 11/6/25.
//

import Testing
import UIKit
@testable import RTMDetSwift
import onnxruntime_objc

struct RTMDetSwiftTests {

    /* Commented out raw inference test - keeping detection test only
    @Test func testRawInference() async throws {
        // Get the model path from the framework bundle
        let frameworkBundle = Bundle(for: RTMDetInferencer.self)
        guard let modelPath = frameworkBundle.path(forResource: "rtmdet-m", ofType: "onnx") else {
            print("Model file not found in framework bundle")
            print("Make sure rtmdet-m.onnx is added to the RTMDetSwift target (not test target)")
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Model file not found"])
        }

        print("Loading model from: \(modelPath)")

        // Initialize inferencer
        guard let inferencer = RTMDetInferencer(modelPath: modelPath, inputWidth: 640, inputHeight: 640) else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to initialize inferencer"])
        }

        print("Inferencer initialized successfully")

        // Load demo image from test bundle
        // Try multiple bundle locations
        var imagePath: String?

        // Try test bundle first
        for bundle in Bundle.allBundles {
            if let path = bundle.path(forResource: "demo", ofType: "jpg") {
                imagePath = path
                break
            }
        }

        // If not found, try direct path in test bundle
        if imagePath == nil {
            let testBundlePath = Bundle.main.bundlePath
            let directPath = testBundlePath + "/demo.jpg"
            if FileManager.default.fileExists(atPath: directPath) {
                imagePath = directPath
            }
        }

        guard let finalPath = imagePath else {
            print("Failed to find demo.jpg in any bundle")
            print("Searched in \(Bundle.allBundles.count) bundles")
            print("Test bundle path: \(Bundle.main.bundlePath)")
            print("Make sure demo.jpg is added to the RTMDetSwiftTests target in Xcode")
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Demo image not found"])
        }

        print("Found demo image at: \(finalPath)")

        guard let image = UIImage(contentsOfFile: finalPath) else {
            print("Failed to load image from: \(finalPath)")
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load image"])
        }

        print("Loaded demo image: \(image.size.width)x\(image.size.height)")

        // Run inference
        print("\nRunning inference...")
        guard let outputs = inferencer.infer(image: image) else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Inference failed"])
        }

        // Print raw outputs
        print("\n=== Inference Results ===")
        print("Number of outputs: \(outputs.count)")

        for (index, output) in outputs.enumerated() {
            if let tensorInfo = try? output.tensorTypeAndShapeInfo() {
                print("\nOutput \(index):")
                print("  Shape: \(tensorInfo.shape)")
                print("  Element type: \(tensorInfo.elementType.rawValue)")

                // Calculate element count from shape
                let elementCount = tensorInfo.shape.reduce(1) { $0 * $1.intValue }
                print("  Element count: \(elementCount)")

                // Print first few values
                if let tensorData = try? output.tensorData() as Data {
                    let floatCount = min(10, tensorData.count / MemoryLayout<Float>.size)
                    let floatArray = tensorData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
                        let typedPointer = ptr.bindMemory(to: Float.self)
                        return Array(UnsafeBufferPointer(start: typedPointer.baseAddress, count: floatCount))
                    }
                    print("  First \(floatCount) values: \(floatArray)")
                }
            }
        }

        print("\n=== Test completed successfully ===")
    } */

    @Test func testDetection() async throws {
        // Get the model path from the framework bundle
        let frameworkBundle = Bundle(for: RTMDetInferencer.self)
        guard let modelPath = frameworkBundle.path(forResource: "rtmdet-m", ofType: "onnx") else {
            print("Model file not found in framework bundle")
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Model file not found"])
        }

        // Initialize inferencer
        guard let inferencer = RTMDetInferencer(modelPath: modelPath, inputWidth: 640, inputHeight: 640) else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to initialize inferencer"])
        }

        print("Inferencer initialized successfully")

        // Load demo image
        var imagePath: String?
        for bundle in Bundle.allBundles {
            if let path = bundle.path(forResource: "demo", ofType: "jpg") {
                imagePath = path
                break
            }
        }

        if imagePath == nil {
            let testBundlePath = Bundle.main.bundlePath
            let directPath = testBundlePath + "/demo.jpg"
            if FileManager.default.fileExists(atPath: directPath) {
                imagePath = directPath
            }
        }

        guard let finalPath = imagePath else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Demo image not found"])
        }

        guard let image = UIImage(contentsOfFile: finalPath) else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load image"])
        }

        print("Loaded demo image: \(image.size.width)x\(image.size.height)")

        // Run detection with post-processing
        print("\nRunning detection with post-processing...")
        guard let detections = inferencer.detect(image: image, confidenceThreshold: 0.5, iouThreshold: 0.5) else {
            throw NSError(domain: "RTMDetSwiftTests", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Detection failed"])
        }

        // Print results
        print("\n=== Detection Results ===")
        print("Found \(detections.count) objects:")

        for (index, detection) in detections.enumerated() {
            let className = COCO_LABELS[detection.classId] ?? "unknown"
            print("\nDetection \(index + 1):")
            print("  Class: \(className) (ID: \(detection.classId))")
            print("  Confidence: \(String(format: "%.2f%%", detection.confidence * 100))")
            print("  Bounding Box: [\(String(format: "%.1f", detection.bbox.x1)), \(String(format: "%.1f", detection.bbox.y1)), \(String(format: "%.1f", detection.bbox.x2)), \(String(format: "%.1f", detection.bbox.y2))]")
            print("  Mask size: \(detection.mask.count) pixels (640x640)")
            print("  Centroid: (\(String(format: "%.1f", detection.centroid.0)), \(String(format: "%.1f", detection.centroid.1)))")
            print("  Number of contours: \(detection.contours.count)")

            // Print first contour details (if any)
            if let firstContour = detection.contours.first {
                let numPoints = firstContour.count / 2
                print("  First contour has \(numPoints) points")

                // Print first few points
                if numPoints > 0 {
                    let pointsToPrint = min(3, numPoints)
                    var pointsStr = "["
                    for i in 0..<pointsToPrint {
                        let x = firstContour[i * 2]
                        let y = firstContour[i * 2 + 1]
                        pointsStr += "(\(String(format: "%.1f", x)), \(String(format: "%.1f", y)))"
                        if i < pointsToPrint - 1 { pointsStr += ", " }
                    }
                    if numPoints > pointsToPrint {
                        pointsStr += ", ..."
                    }
                    pointsStr += "]"
                    print("  First contour points: \(pointsStr)")
                }
            }
        }

        // Draw detections on image
        print("\n=== Generating output image ===")
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(at: .zero)

        guard let context = UIGraphicsGetCurrentContext() else {
            print("Failed to get graphics context")
            UIGraphicsEndImageContext()
            return
        }

        // Scale factor to convert from 640x640 to original image size
        let scaleX = image.size.width / 640.0
        let scaleY = image.size.height / 640.0

        for detection in detections {
            let className = COCO_LABELS[detection.classId] ?? "unknown"

            // Random color for this detection
            let hue = CGFloat(detection.classId % 20) / 20.0
            let color = UIColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1.0)

            // Draw bounding box
            context.setStrokeColor(color.cgColor)
            context.setLineWidth(3.0)
            let bbox = CGRect(
                x: CGFloat(detection.bbox.x1) * scaleX,
                y: CGFloat(detection.bbox.y1) * scaleY,
                width: CGFloat(detection.bbox.x2 - detection.bbox.x1) * scaleX,
                height: CGFloat(detection.bbox.y2 - detection.bbox.y1) * scaleY
            )
            context.stroke(bbox)

            // Draw contours
            context.setStrokeColor(color.withAlphaComponent(0.7).cgColor)
            context.setLineWidth(2.0)
            for contour in detection.contours {
                guard contour.count >= 4 else { continue }

                context.beginPath()
                let firstX = CGFloat(contour[0]) * scaleX
                let firstY = CGFloat(contour[1]) * scaleY
                context.move(to: CGPoint(x: firstX, y: firstY))

                for i in stride(from: 2, to: contour.count, by: 2) {
                    let x = CGFloat(contour[i]) * scaleX
                    let y = CGFloat(contour[i + 1]) * scaleY
                    context.addLine(to: CGPoint(x: x, y: y))
                }
                context.closePath()
                context.strokePath()
            }

            // Draw centroid
            let centroidX = CGFloat(detection.centroid.0) * scaleX
            let centroidY = CGFloat(detection.centroid.1) * scaleY
            context.setFillColor(color.cgColor)
            context.fillEllipse(in: CGRect(x: centroidX - 5, y: centroidY - 5, width: 10, height: 10))

            // Draw label
            let label = "\(className) \(String(format: "%.0f%%", detection.confidence * 100))"
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 16),
                .foregroundColor: UIColor.white,
                .backgroundColor: color
            ]
            let labelSize = (label as NSString).size(withAttributes: attrs)
            (label as NSString).draw(
                at: CGPoint(x: CGFloat(detection.bbox.x1) * scaleX, y: CGFloat(detection.bbox.y1) * scaleY - labelSize.height - 2),
                withAttributes: attrs
            )
        }

        guard let outputImage = UIGraphicsGetImageFromCurrentImageContext() else {
            print("Failed to create output image")
            UIGraphicsEndImageContext()
            return
        }
        UIGraphicsEndImageContext()

        // Save to file
        if let imageData = outputImage.pngData() {
            let outputPath = NSTemporaryDirectory() + "rtmdet_output.png"
            try? imageData.write(to: URL(fileURLWithPath: outputPath))
            print("Output image saved to: \(outputPath)")
        }

        print("\n=== Test completed successfully ===")
    }

}
