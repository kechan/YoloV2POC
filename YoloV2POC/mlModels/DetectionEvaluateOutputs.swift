//
//  DetectionEvaluateOutputs.swift
//  YoloV2POC
//
//  Created by Kelvin C on 1/3/19.
//  Copyright © 2019 Kelvin Chan. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(DetectionEvaluateOutputs) class DetectionEvaluateOutputs: NSObject, MLCustomLayer {
    
    var one: Float = 1
    let num_anchors = 5
    let num_classes = 80
    let anchors: [Float] = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    
    let conv_index_x: UnsafeMutablePointer<Float32>
    let conv_index_y: UnsafeMutablePointer<Float32>
    
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        
        // TODO: Can we do these initialization lazily?
        conv_index_x = UnsafeMutablePointer<Float32>.allocate(capacity: 19*19)      // TODO: don't hard code 19*19
        conv_index_y = UnsafeMutablePointer<Float32>.allocate(capacity: 19*19)
        
        var seq: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        Geometry.tile(UnsafeMutablePointer<Float>(&seq), conv_index_x, 19, 19)
        for k in 0..<19 {
            Geometry.tile(UnsafeMutablePointer<Float>(&seq) + k, conv_index_y + 19*k, 1, 19)
        }
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        print(#function, inputShapes)   // 1, 1, 425, 19, 19
        /*
        var outputShapes: [[NSNumber]] = []
        
        for i in 0..<inputShapes.count {
            let inputShape = inputShapes[i]
            
            let n_channel  = inputShape[2]
            let n_row = inputShape[3]
            let n_col = inputShape[4]
            
            let n_box_params = NSNumber(value: Int(truncating: n_channel)/self.num_anchors)
            
            let shape: [NSNumber] = [1, 1, NSNumber(value: self.num_anchors), n_box_params, n_row, n_col]
            
            outputShapes.append(shape)
        } */
        
//        print(#function, outputShapes)
//        return outputShapes
        return inputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        swift_activate(inputs: inputs, outputs: outputs)
    }
    
    private func swift_activate(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
        for i in 0..<inputs.count {
            let input = inputs[i]
            let output = outputs[i]
            
            assert(input.dataType == .float32)
            assert(output.dataType == .float32)

//            print(output.shape)    // [1, 1, 425, 19, 19]
//            assert(input.shape == output.shape)
            
            let nc = Int32(truncating: output.shape[2])        // 425
            let nrow = Int32(truncating: output.shape[3])      // 19
            let ncol = Int32(truncating: output.shape[4])      // 19
            var nrow_ncol = Int32(nrow * ncol)                 // 19x19
            var two_nrow_ncol = Int32(2 * nrow * ncol)         // 2x19x19
            
            var conv_dims_x = Float(ncol)
            var conv_dims_y = Float(nrow)
            
            let iptr = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
            let optr = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))

//            cblas_scopy(Int32(input.count), iptr, 1, optr, 1)
            
            // Loop thru each box/anchor
            
            for box_id in 0..<num_anchors {
                
                let box_offset = box_id * (Int(nrow_ncol) * (num_classes + 5))
                
                // apply sigmoid 1/(1 + exp^{-x}) elementwise to the 1st 2 channels, which is the
                // first 2x19x19 element of the flatten array iptr/optr, (box_xy)
                
                var offset = Int(0) + box_offset
                
                vDSP_vneg(iptr + offset, 1, optr + offset, 1, vDSP_Length(2 * nrow * ncol))   // -x
                vvexpf(optr + offset, optr + offset, &two_nrow_ncol)                          // exp
                vDSP_vsadd(optr + offset, 1, &one, optr + offset, 1, vDSP_Length(2 * nrow * ncol))  // 1 +
                vDSP_svdiv(&one, optr + offset, 1, optr + offset, 1, vDSP_Length(2 * nrow * ncol))  // 1 /
                
                // apply exp to 2-3 channel (box_wh)
                offset = Int(two_nrow_ncol) + box_offset
                vvexpf(optr + offset, iptr + offset, &two_nrow_ncol)
                
                // apply sigmoid to 4th channel (box_confidence)
                offset = Int(4*nrow_ncol) + box_offset
                vDSP_vneg(iptr + offset, 1, optr + offset, 1, vDSP_Length(nrow_ncol))         // -x
                vvexpf(optr + offset, optr + offset, &nrow_ncol)                              // exp
                vDSP_vsadd(optr + offset, 1, &one, optr + offset, 1, vDSP_Length(nrow_ncol))  // 1 +
                vDSP_svdiv(&one, optr + offset, 1, optr + offset, 1, vDSP_Length(nrow_ncol))  // 1 /
                
                // Apply softmax (stably) to obtain class scores to compute probabilities
                // Starting from 5*nrow*ncol element in the big array
                
                offset = Int(5*nrow_ncol) + box_offset
                // TODO: can this be achieved in a single matrix/vector op?
                
                // 1) find the max of class score within each spatial conv cell and subtract each from it
                for k in 0..<Int(nrow_ncol) {
                    var maxValue: Float = -Float.infinity
                    vDSP_maxv(iptr + offset + k, vDSP_Stride(nrow_ncol), &maxValue, vDSP_Length(num_classes))
                    
                    // x - max
                    maxValue = -maxValue
                    vDSP_vsadd(iptr + offset + k, vDSP_Stride(nrow_ncol), &maxValue, optr + offset + k, vDSP_Stride(nrow_ncol), vDSP_Length(num_classes))
                }
                
                // 2) exp(x), take the exponent of each element within the class score channel (and across spatial cells)
                var num_classes_nrow_ncol = Int32(num_classes) * nrow_ncol
                vvexpf(optr + offset, optr + offset, &num_classes_nrow_ncol)
                
                // 3) sum(exp(x)), sum is over channel direction only, store the sum in sum_exp
                // 4) divide by sum, done across channels direction only.
                
                for k in 0..<Int(nrow_ncol) {
                    var sum_exp: Float = 0
                    vDSP_sve(optr + offset + k, vDSP_Stride(nrow_ncol), &sum_exp, vDSP_Length(num_classes))
                    vDSP_vsdiv(optr + offset + k, vDSP_Stride(nrow_ncol), &sum_exp, optr + offset + k, vDSP_Stride(nrow_ncol), vDSP_Length(num_classes))
                }
                
                // Adjust prediction to each spatial grid point
                // for x
                offset = Int(0) + box_offset
                vDSP_vadd(optr + offset, 1, conv_index_x, 1, optr + offset, 1, vDSP_Length(nrow_ncol))
                vDSP_vsdiv(optr + offset, 1, &conv_dims_x, optr + offset, 1, vDSP_Length(nrow_ncol))
                
                // for y
                offset = Int(nrow_ncol) + box_offset
                vDSP_vadd(optr + offset, 1, conv_index_y, 1, optr + offset, 1, vDSP_Length(nrow_ncol))
                vDSP_vsdiv(optr + offset, 1, &conv_dims_y, optr + offset, 1, vDSP_Length(nrow_ncol))
                
                var anchors_w = anchors[2*box_id]
                var anchors_h = anchors[2*box_id + 1]
                // for w
                offset = Int(2*nrow_ncol) + box_offset
                vDSP_vsmul(optr + offset, 1, &anchors_w, optr + offset, 1, vDSP_Length(nrow_ncol))
                vDSP_vsdiv(optr + offset, 1, &conv_dims_x, optr + offset, 1, vDSP_Length(nrow_ncol))
                
                // for h
                offset = Int(3*nrow_ncol) + box_offset
                vDSP_vsmul(optr + offset, 1, &anchors_h, optr + offset, 1, vDSP_Length(nrow_ncol))
                vDSP_vsdiv(optr + offset, 1, &conv_dims_y, optr + offset, 1, vDSP_Length(nrow_ncol))
            }
        }
    }
}

