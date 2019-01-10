//
//  YoloDetectionController.swift
//  YoloV2POC
//
//  Created by Kelvin C on 1/2/19.
//  Copyright © 2019 Kelvin Chan. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import VideoToolbox
import Accelerate

class YoloDetectionController: NSObject {
    var quality: AVCaptureSession.Preset

    // AVFoundation stuff
    weak var capturePreviewView: UIView?
    
    private let sessionQueue = DispatchQueue(label: "session queue")
    
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput",     // Queue for video frame
        qos: .userInitiated,
        attributes: [],
        autoreleaseFrequency: .workItem)
    
    private var captureSession: AVCaptureSession?
    private var captureSessionFullyPrepared = false
    private var setupResult: AVSessionSetupResult = .success
    
    private var rearCamera: AVCaptureDevice?
    private var rearCameraInput: AVCaptureDeviceInput?
    
    private var photoOutput: AVCapturePhotoOutput?
    
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    // Vision Detection stuff
    let input_image_width = 1024
    let input_image_height = 720
    let num_anchors = 5
    let num_classes = 80
    let labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                          "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                          "teddy bear", "hair drier", "toothbrush"]
    
    private let predictionHandler: ([String], [Double], [CGRect]?, String?) -> Void
    
    private lazy var detectionRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: YoloV2Detection().model)
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.processDetectionFromLiveStream(for: request, error: error)
            }
            
            request.imageCropAndScaleOption = .scaleFill
            //        request.usesCPUOnly = true
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    } ()
    
    private let visionQueue = DispatchQueue(label: "net.kechan.YoloV2POC.serialVisionQueue", qos: .userInitiated)
    private var currentlyAnalyzedPixelBuffer: CVPixelBuffer?
    
    var tik = Date()
    
    init(quality: AVCaptureSession.Preset = .vga640x480, predictionHandler: @escaping ([String], [Double], [CGRect]?, String?) -> Void) {
        self.quality = quality        // .photo     // highest resultion at 4032 × 3024
                                      // .hd4K3840x2160
        self.predictionHandler = predictionHandler
    }
}

// MARK: - AVFoundation Setup
extension YoloDetectionController {
    func prepare(completionHandler: @escaping (Error?) -> Void) {
        
        // Check video authorization status
        func checkCameraAuthorization() {
            
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                // The user has previously granted access to the camera.

                break
            case .notDetermined:
                sessionQueue.suspend()
                AVCaptureDevice.requestAccess(for: .video, completionHandler: { granted in
                    if !granted {
                        self.setupResult = .notAuthorized
                    }
                    self.sessionQueue.resume()
                })
            default:
                // The user has previously denied access.
                setupResult = .notAuthorized
            }            
        }
        
        // create capture session
        func createCaptureSession() throws {
            if setupResult != .success {
                throw YoloDetectionControllerError.cameraNotAuthorized
            }
            
            print("createCaptureSession")
            captureSession = AVCaptureSession()
        }
        
        // Set sessionPreset quality
        func configureSessionPreset() {
            captureSession?.sessionPreset = quality
        }
        
        // configure capture device
        func configureCaptureDevices() throws {
            let session = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back)
            
            //            let session = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInDualCamera], mediaType: .video, position: .back)
            
            let cameras = session.devices.compactMap { $0 }
            
            rearCamera = cameras.first
            
            try rearCamera?.lockForConfiguration()
            rearCamera?.focusMode = .continuousAutoFocus
            rearCamera?.unlockForConfiguration()
        }
        
        // configure capture device input
        func configureCaptureDeviceInputs() throws {
            guard let captureSession = captureSession else { throw YoloDetectionControllerError.captureSessionIsMissing}
            
            if let rearCamera = rearCamera {
                rearCameraInput = try AVCaptureDeviceInput(device: rearCamera)
                
                if captureSession.canAddInput(rearCameraInput!) {
                    captureSession.addInput(rearCameraInput!)
                }
            }
            else {
                throw YoloDetectionControllerError.noCamerasAvailable
            }
        }
        
        // configure capture device output
        func configurePhotoOutput() throws {
            guard let captureSession = captureSession else {
                throw YoloDetectionControllerError.captureSessionIsMissing
            }
            
            photoOutput = AVCapturePhotoOutput()
            
            photoOutput?.setPreparedPhotoSettingsArray([AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])], completionHandler: nil)
            photoOutput?.isHighResolutionCaptureEnabled = true
            
            
            if captureSession.canAddOutput(photoOutput!) {
                captureSession.addOutput(photoOutput!)
            }
            
            if let photoOutput = photoOutput, photoOutput.isDualCameraDualPhotoDeliverySupported {
                photoOutput.isDepthDataDeliveryEnabled = false
                photoOutput.isDualCameraDualPhotoDeliveryEnabled = true
            }
            
            guard let connection = photoOutput?.connection(with: .video) else {
                return
            }
            connection.isEnabled = true
            
            guard connection.isVideoOrientationSupported else { return }
            guard connection.isVideoMirroringSupported else { return }
            connection.videoOrientation = .landscapeRight
        }
        
        func configureVideoOutput() throws {
            guard let captureSession = captureSession else {
                throw YoloDetectionControllerError.captureSessionIsMissing
            }
            
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            
            videoOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            else
            {
                throw YoloDetectionControllerError.outputsAreInvalid
            }
            
            guard let connection = videoOutput.connection(with: .video) else {
                return
            }
            connection.isEnabled = true
            
            guard connection.isVideoOrientationSupported else { return }
            guard connection.isVideoMirroringSupported else { return }
            connection.videoOrientation = .landscapeRight
            connection.isVideoMirrored = false
            connection.preferredVideoStabilizationMode = .auto
        }
        
        func startRunningCaptureSession() throws {
            guard let captureSession = captureSession else { throw YoloDetectionControllerError.captureSessionIsMissing }
            captureSession.startRunning()
        }
        
        checkCameraAuthorization()
        
        sessionQueue.async {
            do {
                defer {                     // ensure we commit config to session up to that point.
                    self.captureSessionFullyPrepared = true
                    self.captureSession?.commitConfiguration()
                    print("commitConfiguration")
                }
                
                try createCaptureSession()
                
                self.captureSession?.beginConfiguration()           // see defer {} for .commitConfiguration
                
                configureSessionPreset()
                try configureCaptureDevices()
                try configureCaptureDeviceInputs()
                try configurePhotoOutput()
                try configureVideoOutput()
            }
            catch {
                DispatchQueue.main.async {
                    completionHandler(error)
                }
                return
            }
            
            self.setupVision()
            
            // After commiting session config, start the session. Make this isnt done before commit.
            do {
                try startRunningCaptureSession()
            }
            catch {
                DispatchQueue.main.async {
                    completionHandler(error)
                }
                return
            }
            
            DispatchQueue.main.async {
                completionHandler(nil)   // completed prepare() without error
            }
        }
        
    }
    
    func displayPreview(on view: UIView) throws {
        guard let captureSession = self.captureSession, captureSession.isRunning else { throw YoloDetectionControllerError.captureSessionIsMissing}
        
        self.capturePreviewView = view
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
//        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.videoGravity = .resizeAspect
        previewLayer?.connection?.videoOrientation = .landscapeLeft
        
        self.capturePreviewView!.layer.insertSublayer(self.previewLayer!, at: 0)
        previewLayer?.frame = self.capturePreviewView!.frame
    }
}

// MARK: - Capture Video AVCaptureVideoDataOutputSampleBufferDelegate
extension YoloDetectionController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
//        let theImage = cvPixelBufferToUIImage(cvPixelBuffer: pixelBuffer)
//        let testImage = UIImage(named: "test_image")
//        let pixelBuffer = ImageToPixelBufferConverter.convertToPixelBuffer(image: testImage!, newSize: (1080, 720))!
        
        if currentlyAnalyzedPixelBuffer == nil {
            // Retain the image buffer for Vision processing.
            currentlyAnalyzedPixelBuffer = pixelBuffer
            
            var requestOptions:[VNImageOption: Any] = [:]
            
            if let cameraInstrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) {
                requestOptions = [.cameraIntrinsics: cameraInstrinsicData]
            }
            
            let exifOrientation = exifOrientationFromDeviceOrientation()
            
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: exifOrientation,
                                                            options: requestOptions)
            
            visionQueue.async {
                do {
                    // Release the pixel buffer when done, allowing the next buffer to be processed.
                    defer { self.currentlyAnalyzedPixelBuffer = nil }
                    
                    self.tik = Date()
                    try imageRequestHandler.perform([self.detectionRequest])
                    let tok = Date()
                    let executionTime = tok.timeIntervalSince(self.tik)
                    
                    //                self.netThrottler.imageAnalysisExecutionTime = executionTime
                    print("Total Execution time: \(executionTime*1000) ms")
                } catch {
                    print(error)
                }
            }
        }
    }
    
    public func exifOrientationFromDeviceOrientation() -> CGImagePropertyOrientation {
        let curDeviceOrientation = UIDevice.current.orientation
        let exifOrientation: CGImagePropertyOrientation
        
        switch curDeviceOrientation {
        case UIDeviceOrientation.portraitUpsideDown:  // Device oriented vertically, Home button on the top
            exifOrientation = .left
        case UIDeviceOrientation.landscapeLeft:       // Device oriented horizontally, Home button on the right
            exifOrientation = .upMirrored
        case UIDeviceOrientation.landscapeRight:      // Device oriented horizontally, Home button on the left
            exifOrientation = .down
        case UIDeviceOrientation.portrait:            // Device oriented vertically, Home button on the bottom
            exifOrientation = .up
        default:
            exifOrientation = .up
        }
        return exifOrientation
    }
}
    
// MARK: - Vision & CoreML
extension YoloDetectionController {
    func setupVision() {
//        analysisRequests.append(detectionRequest)
    }
    
    private func processDetectionFromLiveStream(for request: VNRequest, error: Error?) {
        if let (out_identifierStrings, out_scores, out_boxes) = processDetection(for: request, error: error) {
            
            let message: String? = nil
            
            predictionHandler(out_identifierStrings, out_scores, out_boxes, message)
            
            if let buffer = self.currentlyAnalyzedPixelBuffer {
                let exifOrientation = exifOrientationFromDeviceOrientation()
                let ciImage = CIImage(cvPixelBuffer: buffer).oriented(exifOrientation)
                
                for (k, o) in out_identifierStrings.enumerated() {
                    if o == "bowl" {
                        print("Found bowl")
                        
                    }
                }
            }
        }
    }
    
    private func processDetection(for request: VNRequest, error: Error?) -> ([String], [Double], [CGRect]?)? {
        
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let output = observations.first?.featureValue.multiArrayValue {
            
            let optr = UnsafeMutablePointer<Double>(OpaquePointer(output.dataPointer))
            // print(#function, output.shape)    // (425, 19, 19)
            
            // channel:
            // x, y, w, h, confidence, 80 class probs, x, y, w, h, confidence, 80 class probs, and so on for 5 boxes
            var half = 0.5
            var oneAsDouble: Double = 1
            var zeroAsDouble: Double = 0
            
            let conv_height = output.shape[1].intValue          // 19
            let conv_width = output.shape[2].intValue           // 19
            let conv_h_x_w = conv_height * conv_width             // 19x19

            let x_offset = 0
            let y_offset = 1*conv_h_x_w
            let w_offset = 2*conv_h_x_w
            let h_offset = 3*conv_h_x_w
            let confidence_offset = 4*conv_h_x_w
            let class_prob_offset = 5*conv_h_x_w
            
            let boxes = UnsafeMutablePointer<Double>.allocate(capacity: num_anchors * 4 * conv_h_x_w)
            let box_scores = UnsafeMutablePointer<Double>.allocate(capacity: num_anchors * num_classes * conv_h_x_w)    // 80 classes
            let box_classes = UnsafeMutablePointer<UInt>.allocate(capacity: num_anchors * conv_h_x_w)
            let box_class_scores = UnsafeMutablePointer<Double>.allocate(capacity: num_anchors * conv_h_x_w)
            let prediction_mask = UnsafeMutablePointer<Double>.allocate(capacity: num_anchors * conv_h_x_w)
            
            for box_id in 0..<num_anchors {
                let optr_offset = box_id * (conv_h_x_w * (num_classes + 5))
                let box_offset = box_id * (4 * conv_h_x_w)
                
                // (1) compute yolo_boxes_to_corners
                // ymin
                vDSP_vsmulD(optr + h_offset + optr_offset, 1, &half, boxes + box_offset, 1, vDSP_Length(conv_h_x_w))   // h/2
                vDSP_vsubD(boxes + box_offset, 1, optr + y_offset + optr_offset, 1, boxes + box_offset, 1, vDSP_Length(conv_h_x_w)) // y - h/2
                
                // xmin
                vDSP_vsmulD(optr + w_offset + optr_offset, 1, &half, boxes + conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w))   // w/2
                vDSP_vsubD(boxes + conv_h_x_w + box_offset, 1, optr + x_offset + optr_offset, 1, boxes + conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w))  // x - w/2
                
                // ymax
                vDSP_vsmulD(optr + h_offset + optr_offset, 1, &half, boxes + 2*conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w))   // h/2
                vDSP_vaddD(optr + y_offset + optr_offset, 1, boxes + 2*conv_h_x_w + box_offset, 1, boxes + 2*conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w)) // y + h/2
                
                // xmax
                vDSP_vsmulD(optr + w_offset + optr_offset, 1, &half, boxes + 3*conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w))   // w/2
                vDSP_vaddD(optr + x_offset + optr_offset, 1, boxes + 3*conv_h_x_w + box_offset, 1, boxes + 3*conv_h_x_w + box_offset, 1, vDSP_Length(conv_h_x_w))  // x + w/2
                
                // (2) yolo_filter_boxes
//                var score_threshold = 0.3
                
                // compute box_scores
                let box_scores_offset = box_id * (num_classes * conv_h_x_w)
                
                for cls in 0..<num_classes {
                    vDSP_vmulD(optr + confidence_offset + optr_offset, 1,
                               optr + conv_h_x_w*cls + class_prob_offset + optr_offset, 1,
                               box_scores + conv_h_x_w*cls + box_scores_offset, 1,
                               vDSP_Length(conv_h_x_w))
                }
                
                // compute box_class_scores and box_classes (max and argmax channel-wise)
                let box_class_scores_offset = box_id * conv_h_x_w
                let box_classes_offset = box_id * conv_h_x_w
                
                for k in 0..<conv_h_x_w { // loop over conv spatial dim,
                    vDSP_maxviD(box_scores + k + box_scores_offset, conv_h_x_w,
                                box_class_scores + k + box_class_scores_offset, box_classes + k + box_classes_offset,
                                vDSP_Length(num_classes))
                    box_classes[k + box_classes_offset] = box_classes[k + box_classes_offset] / UInt(conv_h_x_w)
                }
            }
            
            // (2) yolo_filter_boxes
            // compute prediction_mask
            var score_threshold = 0.3
            vDSP_vlimD(box_class_scores, 1, &score_threshold, &oneAsDouble, prediction_mask, 1, vDSP_Length(num_anchors * conv_h_x_w))
            vDSP_vthresD(prediction_mask, 1, &zeroAsDouble, prediction_mask, 1, vDSP_Length(num_anchors * conv_h_x_w))
            
            // find indices to prediction_mask that are nonzero (aka 1.0)
            let nonzero_indice = UnsafeMutablePointer<sparse_index>.allocate(capacity: num_anchors * conv_h_x_w)
            nonzero_indice.initialize(repeating: -1, count: num_anchors * conv_h_x_w)
            
            let nonzero_mask_value = UnsafeMutablePointer<Double>.allocate(capacity: num_anchors * conv_h_x_w) // dummy placeholder
            
            let num_filtered = sparse_pack_vector_double(sparse_dimension(num_anchors*conv_h_x_w), sparse_dimension(num_anchors*conv_h_x_w), prediction_mask, 1, nonzero_mask_value, nonzero_indice)
            
            let _tmp_nonzero_indice_double = Array(UnsafeBufferPointer(start: nonzero_indice, count: num_filtered))
            var __tmp_nonzero_indice_double = _tmp_nonzero_indice_double.map { Double($0) }
            let nonzero_indice_double = UnsafeMutablePointer<Double>(&__tmp_nonzero_indice_double) // nonzero indices
            
            // Using nonzero_indice_double to retrieve the list of scores and classes from box_classes and box_class_scores
            let scores = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered)
            let classes = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered)
            
            // Convert box_classes which is UnsafeMutablePointer<UInt>
            // to box_classes_double which is UnsafeMutablePointer<Double>
            let _tmp_box_classes_double = Array(UnsafeBufferPointer(start: box_classes, count: num_anchors * conv_h_x_w))
            var __tmp_box_classes_double = _tmp_box_classes_double.map { Double($0) }
            let box_classes_double = UnsafeMutablePointer<Double>(&__tmp_box_classes_double)
            
            vDSP_vindexD(box_class_scores, nonzero_indice_double, 1, scores, 1, vDSP_Length(num_filtered))
            vDSP_vindexD(box_classes_double, nonzero_indice_double, 1, classes, 1, vDSP_Length(num_filtered))
            
            // using nonzero_indice_double to retrieve boxes (4 coords)
            
            let boxes_filtered = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered*4)
            // boxes_filtered data layouts:
            // [y_min_0, x_min_0, y_max_0, x_max_0, y_min_1, x_min_1, y_max_1, x_max_1, ..... ]
            // for box0, box1, etc.
            
            // try do this anchor by anchor
            // y_min_*
//            vDSP_vindexD(boxes, nonzero_indice_double, 1, boxes_filtered, 4, vDSP_Length(num_filtered))
            
            var i = 0
            for k in 0..<num_filtered {
                let idx = nonzero_indice_double[k]
                
                let anchor_id = Int(idx) / conv_h_x_w
                let row_col = Int(idx) - anchor_id * conv_h_x_w
                let row = row_col / conv_width
                let col = row_col - row * conv_width
//                print("\(row), \(col) anchor \(anchor_id)")
                
                // y_min, x_min, y_max, x_max
                for c in 0..<4 {
                    boxes_filtered[i+c] = boxes[(anchor_id*4 + c) * conv_h_x_w + row * conv_width + col ]
                }
                i += 4
            }
            
            // (3) scale boxes back to original image shape
            let scale = UnsafeMutablePointer<Double>.allocate(capacity: 4 * num_filtered)
            var dims: [Double] = [input_image_height, input_image_width, input_image_height, input_image_width].map { Double($0) }
            
            Geometry.tile(UnsafeMutablePointer<Double>(&dims), scale, 4, UInt(num_filtered))
            
            vDSP_vmulD(boxes_filtered, 1, scale, 1, boxes_filtered, 1, vDSP_Length(4*num_filtered))
            
            // (4) Non-max suppression
            let iou_threshold = 0.5
            
            // sort desc according to scores
            var boxes_indices: [UInt] = Array(0..<UInt(num_filtered))
            vDSP_vsortiD(scores, &boxes_indices, nil, vDSP_Length(num_filtered), -1)
            //            print(boxes_indices)
            
            // arrange the boxes in boxes_filtered in descending order of their scores.
            var boxes_indicesx1: [Double] = boxes_indices.map { Double($0) }
            var boxes_indicesx4: [Double] = boxes_indices.map { Double($0)*4 }
            var boxes_indicesx4_1: [Double] = boxes_indices.map { Double($0)*4 + 1}
            var boxes_indicesx4_2: [Double] = boxes_indices.map { Double($0)*4 + 2}
            var boxes_indicesx4_3: [Double] = boxes_indices.map { Double($0)*4 + 3}
            
            let boxes_filtered_sorted = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered*4)
            
            vDSP_vindexD(boxes_filtered, &boxes_indicesx4, 1, boxes_filtered_sorted, 4, vDSP_Length(num_filtered))
            vDSP_vindexD(boxes_filtered, &boxes_indicesx4_1, 1, boxes_filtered_sorted+1, 4, vDSP_Length(num_filtered))
            vDSP_vindexD(boxes_filtered, &boxes_indicesx4_2, 1, boxes_filtered_sorted+2, 4, vDSP_Length(num_filtered))
            vDSP_vindexD(boxes_filtered, &boxes_indicesx4_3, 1, boxes_filtered_sorted+3, 4, vDSP_Length(num_filtered))
            
            // testing(arr_of_double: boxes_filtered_sorted, length: 4*num_filtered, name: "boxes_filtered_sorted")
            
            // arrange the score in descending order
            let scores_sorted = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered)
            
            vDSP_vindexD(scores, &boxes_indicesx1, 1, scores_sorted, 1, vDSP_Length(num_filtered))
            //            testing(arr_of_double: scores_sorted, length: num_filtered, name: "scores_sorted")
            
            // arrange the class in descending score order
            let classes_sorted = UnsafeMutablePointer<Double>.allocate(capacity: num_filtered)
            vDSP_vindexD(classes, &boxes_indicesx1, 1, classes_sorted, 1, vDSP_Length(num_filtered))
            //            testing(arr_of_double: classes_sorted, length: num_filtered, name: "classes_sorted")
            
            var box_mask: [Double] = Array(repeating: 1.0, count: num_filtered)
            
            if num_filtered > 0 {
                for k in 0..<num_filtered-1 {
                    if box_mask[k] > 0.0 {
                        let ious = Geometry.iou(boxes_filtered_sorted + k*4, boxes_filtered_sorted + (k+1)*4, count: UInt(num_filtered-1-k))
                        //            print("ious:\n\(ious)")
                        let _box_mask = ious.map { $0 > iou_threshold ? 0.0 : 1.0 }
                        for m in 0..<_box_mask.count {
                            box_mask[m+k+1] *= _box_mask[m]
                        }
                    }
                    
                    //                print("box_mask \(k): \(box_mask)")
                }
            }
            
            // construct the final output
            // out_boxes, out_scores, out_classes
            var out_boxes: [CGRect] = []
            var out_scores: [Double] = []
            var out_classes: [Int] = []
            
            let imageWidth = CGFloat(input_image_width)
            let imageHeight = CGFloat(input_image_height)
            
            for k in 0..<num_filtered {
                if box_mask[k] > 0.0 {
                    out_boxes.append(
                        CGRect(x: CGFloat(boxes_filtered_sorted[k*4 + 1]) / imageWidth,
                               y: CGFloat(boxes_filtered_sorted[k*4 + 0]) / imageHeight,
                               width: CGFloat(boxes_filtered_sorted[k*4 + 3] - boxes_filtered_sorted[k*4 + 1]) / imageWidth,
                               height: CGFloat(boxes_filtered_sorted[k*4 + 2] - boxes_filtered_sorted[k*4 + 0]) / imageHeight
                        )
                    )
                    
                    out_scores.append(scores_sorted[k])
                    
                    out_classes.append(Int(classes_sorted[k]))
                }
            }
            
            classes_sorted.deallocate()
            scores_sorted.deallocate()
            boxes_filtered_sorted.deallocate()
            scale.deallocate()
            boxes_filtered.deallocate()
            classes.deallocate()
            scores.deallocate()
            nonzero_mask_value.deallocate()
            nonzero_indice.deallocate()
            prediction_mask.deallocate()
            box_class_scores.deallocate()
            box_classes.deallocate()
            box_scores.deallocate()
            boxes.deallocate()
            
            let out_identifierStrings = out_classes.map { labels[$0] }
            
            return (out_identifierStrings, out_scores, out_boxes)
            /*
            func test_boxes() {
                // 0 -> y_min, 1 -> x_min, 2 -> y_max, 3 -> x_max
                // Traffic light at box0, (4, 6)
                // Car at box1, (8, 8)
                // Car at box1, (9, 0)
                // Truck at box1, (7, 13)
                let row = 4
                let col = 6
                let box = 0
                
                print("test box y_min: \(boxes[box * (4 * conv_h_x_w) + 0 * conv_h_x_w + row*conv_width + col])")
                print("test box x_min: \(boxes[box * (4 * conv_h_x_w) + 1 * conv_h_x_w + row*conv_width + col])")
                print("test box y_max: \(boxes[box * (4 * conv_h_x_w) + 2 * conv_h_x_w + row*conv_width + col])")
                print("test box x_max: \(boxes[box * (4 * conv_h_x_w) + 3 * conv_h_x_w + row*conv_width + col])")
            }
            
            func test_box_classes() {
                let row = 4
                let col = 6
                let box = 0
                
                print("test box_class: \(box_classes[box*conv_h_x_w + row*conv_width + col])")
            }
            
            func test_box_class_scores() {
                let row = 4
                let col = 6
                let box = 0
                print("box_class_scores: \(box_class_scores[box*conv_h_x_w + row*conv_width + col])")
            }
            
            func test_nonzero_indice() {
                print("num_filtered: \(num_filtered)")    // 11
                
                // [82, 101, 151, 155, 157, 174, 507, 511, 512, 521.0, 532.0]
                // (4, 6) anchor 0
                // (5, 6) anchor 0
                // (7, 18) anchor 0
                // (8, 3) anchor 0
                // (8, 5) anchor 0
                // (9, 3) anchor 0
                // (7, 13) anchor 1
                // (7, 17) anchor 1
                // (7, 18) anchor 1
                // (8, 8) anchor 1
                // (9, 0) anchor 1
                
                print("__tmp_nonzero_indice_double: \(__tmp_nonzero_indice_double)")
                
            }
            
            func test_boxes_filtered_scores_class() {
                var i = 0
                for k in 0..<num_filtered {
                    for c in 0..<4 {
                        print(boxes_filtered[i+c], terminator: ", ")
                    }
                    print("score: \(scores[k]), class: \(classes[k])")
                    i += 4
                }
            }
            
            func test_out_boxes_scores_classes() {
                for k in 0..<out_boxes.count {
                    print("out_box: \(out_boxes[k])")
                    print("out_score: \(out_scores[k])")
                    print("out_class: \(out_classes[k])")
                }
            }
            
//            test_box_classes()
//            test_box_class_scores()
//            test_boxes()
//            test_boxes_filtered_scores_class()
//            test_nonzero_indice()
//            test_out_boxes_scores_classes()
            */
        }
        
        return nil
    }
    /*
    private func test(output: MLMultiArray) {
        
        // Testing model after final activation
        print("conf: \(output[[89, 8, 8]])")   // confidence at (8, 8) box 1, 0.77685546875
        
        print("conf: \(output[[89, 8, 9]])")   // confidence at (8, 9) box 1, 0.003551483154296875
        
        print("conf: \(output[[89, 9, 0]])")   // confidence at (9, 0) box 1, 0.65966796875
        
        print("conf: \(output[[89, 7, 13]])")   // confidence at (7, 13) box 1, 0.7890625
        
        // class prob at (7, 13) for box 1
        for i in 90..<170 {
            let index = NSNumber(value: i)
            print("prob: \(output[[index, 9, 0]])")
        }
        
        print("-----")
        
        print("x: \(output[[85, 8, 8]]), y: \(output[[86, 8, 8]])")  // x, y at (8, 8) box 1, 0.451416015625, y: 0.44873046875
        
        print("x: \(output[[85, 8, 9]]), y: \(output[[86, 8, 9]])")  // x, y at (8, 9) box 1, 0.49169921875, y: 0.4375
        
        print("x: \(output[[85, 9, 0]]), y: \(output[[86, 9, 0]])")  // x, y at (9, 0) box 1, 0.044677734375, y: 0.490478515625
        
        print("x: \(output[[85, 7, 13]]), y: \(output[[86, 7, 13]])")  // x, y at (7, 13) box 1, 0.720703125, y: 0.391357421875
        
        print("w: \(output[[87, 8, 8]]), h: \(output[[88, 8, 8]])")  // w, h at (8, 8) box 1, 0.1693115234375, h: 0.08544921875
        
        print("w: \(output[[87, 8, 9]]), h: \(output[[88, 8, 9]])")  // w, h at (8, 9) box 1, 0.1162109375, h: 0.06488037109375
        
        print("w: \(output[[87, 9, 0]]), h: \(output[[88, 9, 0]])")  // w, h at (9, 0) box 1, 0.07452392578125, h: 0.08245849609375
        
        print("w: \(output[[87, 7, 13]]), h: \(output[[88, 7, 13]])")  // w, h at (7, 13) box 1, 0.168701171875, h: 0.09814453125
        
        /*
         // Testing yolo_model before final activation layer
         var index: [NSNumber] = [89, 8, 8]
         print("output \(output[index])")   // 1.2490234375
         
         index = [89, 8, 9]
         print("output \(output[index])")   // -5.63671875
         
         index = [89, 9, 0]
         print("output \(output[index])")   // 0.66259765625
         
         index = [89, 7, 13]
         print("output \(output[index])")   // 1.3212890625
         
         print("-----")
         
         for row in 0..<19 {
         for col in 0..<19 {
         for box in 0..<5 {
         let b = NSNumber(value: 85 * box + 4)
         index = [b, row, col] as! [NSNumber]
         print("(\(row), \(col), \(b)) \(output[index])")
         }
         }
         } */
    }
    */
}

// MARK: - Errors

extension YoloDetectionController {
    enum YoloDetectionControllerError: Swift.Error {
        case captureSessionAlreadyRunning
        case captureSessionIsMissing
        case inputsAreInvalid
        case outputsAreInvalid
        case invalidOperation
        case noCamerasAvailable
        case cameraNotAuthorized
        case unknown
    }
    
    public enum PhotoCaptureMode {
        case fullscreen
        case square
    }
    
    // MARK: AVSession Management
    
    private enum AVSessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
}
