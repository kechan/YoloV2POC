//
//  MainViewController.swift
//  YoloV2POC
//
//  Created by Kelvin C on 1/2/19.
//  Copyright © 2019 Kelvin Chan. All rights reserved.
//

import UIKit
import AVFoundation

class MainViewController: UIViewController {
    
    @IBOutlet weak var capturePreviewView: UIView!
    var rootLayer: CALayer?
    private var detectionOverlay: CALayer?
    
    // Visual Recognition and Prediction
    var yoloDetectionController: YoloDetectionController?
    private var _boundingBoxPredictionAdjustmentRatio: CGFloat?
    var boundingBoxPredictionAdjustmentRatio: CGFloat? {
        
        get {
            // calculate bounding box adjustment
            
            // The AVCaptureVideoPreviewLayer has videoGravity of .resizeAspectFill and
            // Preset of .hd1280x720
            // Because of this resizing, the normalizing will need adjustment
            
            if _boundingBoxPredictionAdjustmentRatio == nil {
                let r1 = CGFloat(720.0/1280.0)
                if let height = capturePreviewView?.bounds.size.height,
                    let width = capturePreviewView?.bounds.size.width {
                    
                    let r2 = height/width
                    _boundingBoxPredictionAdjustmentRatio = r2/r1
                }
            }
            return _boundingBoxPredictionAdjustmentRatio
        }
        
        set(newValue) {
            _boundingBoxPredictionAdjustmentRatio = newValue
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        
        yoloDetectionController = YoloDetectionController(quality: .hd1280x720) { [unowned self] (identifiers, scores, boxes, message) in
            
//            if let boxes = boxes {
//                for k in 0..<boxes.count {
//                    print("identifiers: \(identifiers[k])")
//                    print("scores: \(scores[k])")
//                    print("boxes: \(boxes[k])")
//                }
//            }
            
            DispatchQueue.main.async {
                // debug by hardcoding some boxes, scores, and identifiers
//                self.draw(boundingBoxes: [CGRect(x: 0.8, y: 0.8, width: 0.2, height: 0.2)],
//                          scores: [0.99],
//                          identifiers: ["car"])
                
                guard identifiers.count > 0 else {
                    self.clearAllBoundingBoxes()
                    return
                }
                
                if let boxes = boxes {
                    self.draw(boundingBoxes: boxes, scores: scores, identifiers: identifiers)
                }
            }
        }
        
        yoloDetectionController?.prepare { [weak self] error in
            if let error = error as? YoloDetectionController.YoloDetectionControllerError {
                if error == .cameraNotAuthorized {
                    let changePrivacySetting = "Yolo doesn't have permission to use the camera, please change privacy settings"

                    let message = changePrivacySetting
                    
                    let alertController = UIAlertController(title: "Yolo", message: message, preferredStyle: .alert)
                    
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Alert OK button"),
                                                            style: .cancel,
                                                            handler: nil))
                    
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("Settings", comment: "Alert button to open Settings"),
                                                            style: .`default`,
                                                            handler: { _ in
                                                                UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
                    }))
                    
                    self?.present(alertController, animated: true, completion: nil)
                    
//                    self?.statusViewController.showMessage(changePrivacySetting)
                    
//                    _ = self?.speak(words: "Please let me use your camera.", rate: 0.5)
                    
                }
                else {
                    print(error)   // dev debug
                }
            } else {
//                self?.statusViewController.showMessage("Welcome!", autoHide: true)
//                _ = self?.speak(words: "Welcome!", rate: 0.5)
            }
            
            try? self?.yoloDetectionController?.displayPreview(on: self!.capturePreviewView)
            
            self?.rootLayer = self?.capturePreviewView.layer
            
            self?.setupDetectionLayer()
        }
    }

}

// MARK: - Visualizing Predictions
extension MainViewController {
    private func setupDetectionLayer() {
        
        let midX = capturePreviewView.bounds.midX
        let midY = capturePreviewView.bounds.midY
//        let side = min(midX, midY) * 2.0
        
        let half_h = 720.0/1280.0 * midX   // half the height of the detectionOverlay
        let delta_h = midY - half_h
        
        detectionOverlay = CALayer()    // container layer that has all the renderings of the observations
        if let detectionOverlay = detectionOverlay, let rootLayer = rootLayer {
            detectionOverlay.name = "DetectionOverlay"
            
            detectionOverlay.frame = CGRect(x: 0.0,
                                            y: delta_h,
                                            width: midX * 2.0,
                                            height: half_h * 2.0)
            
            detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
            rootLayer.addSublayer(detectionOverlay)
        }
    }
    
    private func clearAllBoundingBoxes() {
        detectionOverlay?.sublayers = nil
    }
    
    private func draw(boundingBoxes: [CGRect], scores: [Double], identifiers: [String]) {
        //*
        // draw multiple bounding boxes for detection model
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        clearAllBoundingBoxes()
        
        for (k, boundingBox) in boundingBoxes.enumerated() {
            let scaledBoundingBox = normalizeRect(boundingBox)
            
            let shapeLayer = createRoundedRectLayerWithBounds(scaledBoundingBox, identifier: identifiers[k], score: scores[k])
            let textLayer = createTextSubLayerInBounds(scaledBoundingBox, identifier: identifiers[k], confidence: round(100*scores[k])/100)
            
            shapeLayer.addSublayer(textLayer)
            detectionOverlay?.addSublayer(shapeLayer)
        }
        
        CATransaction.commit()
        // */
    }
    
    private func normalizeRect(_ boundingBox: CGRect) -> CGRect {
        guard let detectionOverlay = detectionOverlay else { return CGRect.zero}
        
//        let adj = boundingBoxPredictionAdjustmentRatio ?? 1.0
        let adj = CGFloat(1.0)
        
        var x = boundingBox.origin.x
        var y = boundingBox.origin.y
        var width = boundingBox.size.width
        var height = boundingBox.size.height
        
        x = (x - 0.5) * adj + 0.5   // adjust for videogravity resizing
        y = (y - 0.5) * adj + 0.5
        width = width * adj
        height = height * adj
        
        let h = detectionOverlay.bounds.height
        let w = detectionOverlay.bounds.width
        
        x = x * w
        y = y * h
        
        width = width * w
        height = height * h
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    private func createRoundedRectLayerWithBounds(_ bounds: CGRect, identifier: String, score: Double) -> CALayer {
        let squarePath = UIBezierPath(roundedRect: bounds, cornerRadius: 1.0)
        
        let shapeLayer = CAShapeLayer()
        shapeLayer.path = squarePath.cgPath
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.strokeColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        let idx = (yoloDetectionController?.labels.firstIndex(of: identifier)!)!
        let color_idx = idx % 6
        switch color_idx {
        case 0:
            shapeLayer.strokeColor  = UIColor.yellow.cgColor
        case 1:
            shapeLayer.strokeColor  = UIColor.orange.cgColor
        case 2:
            shapeLayer.strokeColor  = UIColor.red.cgColor
        case 3:
            shapeLayer.strokeColor  = UIColor.cyan.cgColor
        case 4:
            shapeLayer.strokeColor  = UIColor.green.cgColor
        case 5:
            shapeLayer.strokeColor = UIColor.magenta.cgColor
        default:
            shapeLayer.strokeColor  = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        }
        shapeLayer.lineWidth = 2.0
        
//        if score < boundingBoxAlphaThreshold {
//            shapeLayer.opacity = 0.5
//        }
        
        return shapeLayer
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: Double) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        
//        let displayText = String(format: "\(CoinSenseUIViewController.mlprediction_text[identifier] ?? ""):  %.2f", confidence)
        
        let idx = (yoloDetectionController?.labels.firstIndex(of: identifier)!)!
        
        let displayText = String(format: "\(identifier): %.2f", confidence)
        
        let formattedString = NSMutableAttributedString(string: displayText)
        let largeFont = UIFont(name: "Helvetica", size: 16.0)!
        
        var textColor = UIColor(cgColor: CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])!)
        
        let color_idx = idx % 6
        switch color_idx {
        case 0:
            textColor = UIColor.yellow
        case 1:
            textColor = UIColor.orange
        case 2:
            textColor = UIColor.red
        case 3:
            textColor = UIColor.cyan
        case 4:
            textColor = UIColor.green
        case 5:
            textColor = UIColor.magenta
        default:
            //            textColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
            textColor = UIColor.black
        }
        
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont, NSAttributedString.Key.foregroundColor: textColor], range: NSRange(location: 0, length: identifier.count))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.width - 10, height: bounds.size.height - 10)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        
        textLayer.contentsScale = 2.0 // retina rendering
        
        // rotate the layer into screen orientation and scale and mirror
        //        textLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: 1.0, y: -1.0))
        return textLayer
    }
}
