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
        // Resize image to model input size
        guard let resizedImage = image.resize(to: CGSize(width: inputWidth, height: inputHeight)),
              let cgImage = resizedImage.cgImage else {
            print("Failed to resize image")
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

    public func runInference(inputData: [Float]) throws -> [ORTValue] {
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

        // Run inference
        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: outputNameSet,
            runOptions: nil
        )

        return Array(outputs.values)
    }

    public func infer(image: UIImage) -> [ORTValue]? {
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
}

// Extension to resize UIImage
extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
