//
//  RTMDetInferencer.swift
//  RTMDetSwift
//
//  Raw ONNX Runtime inferencer for RTMDet
//

import Foundation
import UIKit
import onnxruntime_objc

public class RTMDetInferencer {
    private var session: ORTSession?
    private var env: ORTEnv?
    private let inputWidth: Int
    private let inputHeight: Int

    public init?(modelPath: String, inputWidth: Int = 640, inputHeight: Int = 640) {
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight

        do {
            // Create ONNX Runtime environment
            env = try ORTEnv(loggingLevel: .warning)

            // Create session options
            let options = try ORTSessionOptions()

            // CRITICAL: Disable CPU memory arena to prevent memory buildup
            // Memory arena pre-allocates and never returns memory to system
            // This alone can save GBs of memory during continuous inference
            try options.addConfigEntry(withKey: "session.use_arena_allocators", value: "0")

            // Use basic graph optimization for best CoreML compatibility
            try options.setGraphOptimizationLevel(.basic)

            // Enable CoreML with Neural Engine
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            coreMLOptions.useCPUOnly = false
            coreMLOptions.enableOnSubgraphs = true
            coreMLOptions.onlyEnableForDevicesWithANE = false
            coreMLOptions.onlyAllowStaticInputShapes = true  // Static shapes = less memory
            coreMLOptions.createMLProgram = true  // Use MLProgram format (iOS 15+) for better memory

            try options.appendCoreMLExecutionProvider(with: coreMLOptions)

            // Create session
            session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: options)

            print("Successfully loaded ONNX model from: \(modelPath)")

            // Print model input/output info
            try printModelInfo()

        } catch {
            print("Error initializing ONNX Runtime session: \(error)")
            return nil
        }
    }

    private func printModelInfo() throws {
        guard let session = session else { return }

        // Get input names
        let inputNames = try session.inputNames()
        print("\n=== Model Input Info ===")
        for name in inputNames {
            print("Input: \(name)")
        }

        // Get output names
        let outputNames = try session.outputNames()
        print("\n=== Model Output Info ===")
        for name in outputNames {
            print("Output: \(name)")
        }
        print("")
    }

    public func preprocess(image: UIImage) -> [Float]? {
        // Only resize if image is not already the correct size
        let targetSize = CGSize(width: inputWidth, height: inputHeight)
        let needsResize = image.size.width != targetSize.width || image.size.height != targetSize.height

        let imageToProcess: UIImage
        if needsResize {
            guard let resized = image.resize(to: targetSize) else {
                print("Failed to resize image")
                return nil
            }
            imageToProcess = resized
        } else {
            imageToProcess = image
        }

        guard let cgImage = imageToProcess.cgImage else {
            print("Failed to get CGImage")
            return nil
        }

        // Convert to RGB float array (CHW format: [1, 3, H, W])
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )

        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to CHW format and normalize to [0, 1]
        var floatArray = [Float](repeating: 0, count: 3 * height * width)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * width + x
                let offset = (y * width + x) * bytesPerPixel

                // R channel
                floatArray[pixelIndex] = Float(pixelData[offset]) / 255.0
                // G channel
                floatArray[height * width + pixelIndex] = Float(pixelData[offset + 1]) / 255.0
                // B channel
                floatArray[2 * height * width + pixelIndex] = Float(pixelData[offset + 2]) / 255.0
            }
        }

        return floatArray
    }

    public func runInference(inputData: [Float]) throws -> [String: ORTValue] {
        guard let session = session, let env = env else {
            throw NSError(domain: "RTMDetInferencer", code: -1, userInfo: [NSLocalizedDescriptionKey: "Session not initialized"])
        }

        // Create input tensor
        let inputShape: [NSNumber] = [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)]

        let inputData = inputData.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }

        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: inputShape
        )

        // Get input and output names
        let inputNames = try session.inputNames()
        guard let inputName = inputNames.first else {
            throw NSError(domain: "RTMDetInferencer", code: -1, userInfo: [NSLocalizedDescriptionKey: "No input name found"])
        }

        let outputNames = try session.outputNames()
        let outputNameSet = Set(outputNames)

        // Run inference - returns dictionary with output names as keys
        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: outputNameSet,
            runOptions: nil
        )

        return outputs
    }

    public func infer(image: UIImage) -> [String: ORTValue]? {
        guard let preprocessedData = preprocess(image: image) else {
            print("Preprocessing failed")
            return nil
        }

        do {
            let outputs = try runInference(inputData: preprocessedData)
            return outputs
        } catch {
            print("Inference failed: \(error)")
            return nil
        }
    }

    /// Run inference and post-processing in one call
    /// - Parameters:
    ///   - image: Input image
    ///   - confidenceThreshold: Confidence threshold for detections
    ///   - iouThreshold: IoU threshold for NMS
    /// - Returns: Array of Detection objects
    /// Run inference directly on preprocessed float array (CHW format)
    /// - Parameters:
    ///   - inputData: Float array in CHW format [3, H, W], normalized to [0, 1]
    ///   - confidenceThreshold: Confidence threshold for detections
    ///   - iouThreshold: IoU threshold (unused, kept for compatibility)
    /// - Returns: Array of Detection objects
    public func detectFromFloatArray(inputData: [Float], confidenceThreshold: Float = 0.5, iouThreshold: Float = 0.5) -> [Detection]? {
        do {
            let outputs = try runInference(inputData: inputData)
            let postProcessor = RTMDetPostProcessor(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)
            let detections = try postProcessor.process(outputs: outputs)
            return detections
        } catch {
            print("Inference or post-processing failed: \(error)")
            return nil
        }
    }

    public func detect(image: UIImage, confidenceThreshold: Float = 0.5, iouThreshold: Float = 0.5) -> [Detection]? {
        guard let outputs = infer(image: image) else {
            return nil
        }

        let postProcessor = RTMDetPostProcessor(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)

        do {
            let detections = try postProcessor.process(outputs: outputs)
            return detections
        } catch {
            print("Post-processing failed: \(error)")
            return nil
        }
    }
}

// Extension to resize UIImage efficiently using CGContext
extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        // Use autoreleasepool to prevent memory buildup
        return autoreleasepool {
            guard let cgImage = self.cgImage else { return nil }

            let width = Int(size.width)
            let height = Int(size.height)
            let bitsPerComponent = 8
            let bytesPerRow = width * 4
            let colorSpace = CGColorSpaceCreateDeviceRGB()

            guard let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return nil
            }

            // Draw the image
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            guard let scaledImage = context.makeImage() else {
                return nil
            }

            return UIImage(cgImage: scaledImage)
        }
    }
}
