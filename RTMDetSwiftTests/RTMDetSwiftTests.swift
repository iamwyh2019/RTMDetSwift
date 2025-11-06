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
    }

}
