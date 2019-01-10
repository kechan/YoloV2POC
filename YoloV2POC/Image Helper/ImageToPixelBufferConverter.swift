//
//  ImageToPixelBufferConverter.swift
//  SmartGroceryList
//
//  Created by Meghan Kane on 7/31/17.
//  Copyright © 2017 Meghan Kane. All rights reserved.
//

import AVFoundation
import UIKit

// MARK: - ImageToPixelBufferConverter

class ImageToPixelBufferConverter {

    // TODO: Do not use in production, this method is involved with a bad memory leak, need more investigation.
    static func convertToPixelBuffer(image: UIImage, newSize size: (CGFloat, CGFloat)) -> CVPixelBuffer? {

        // resize image
        let newSize = CGSize(width: size.0, height: size.1)
        UIGraphicsBeginImageContext(newSize)
        image.draw(in: CGRect(origin: CGPoint.zero, size: newSize))

        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            return nil
        }

        UIGraphicsEndImageContext()

        // convert to pixel buffer

        let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(newSize.width),
                                         Int(newSize.height),
                                         kCVPixelFormatType_32ARGB,
                                         attributes,
                                         &pixelBuffer)

        guard let createdPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }

        CVPixelBufferLockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(createdPixelBuffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(newSize.width),
                                      height: Int(newSize.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(createdPixelBuffer),
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }

        context.translateBy(x: 0, y: newSize.height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        resizedImage.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return createdPixelBuffer
    }
}

func sampleBufferToUIImage(sampleBuffer: CMSampleBuffer) -> UIImage {
    let imageBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
    return cvPixelBufferToUIImage(cvPixelBuffer: imageBuffer)
}

func cvPixelBufferToUIImage(cvPixelBuffer: CVPixelBuffer) -> UIImage {
    let ciimage : CIImage = CIImage(cvPixelBuffer: cvPixelBuffer)
    return convert(cmage: ciimage)
}

func convert(cmage: CIImage) -> UIImage
{
    let context:CIContext = CIContext.init(options: nil)
    let cgImage:CGImage = context.createCGImage(cmage, from: cmage.extent)!
    let image:UIImage = UIImage.init(cgImage: cgImage)
    return image
}

