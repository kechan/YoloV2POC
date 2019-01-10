//
//  Space2Depthx2.swift
//  YoloV2POC
//
//  Created by Kelvin C on 1/3/19.
//  Copyright © 2019 Kelvin Chan. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

// Implementing tf.space_to_depth with block_size of 2 as a MLCustomLayer

@objc(Space2Depthx2) class Space2Depthx2: NSObject, MLCustomLayer {
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        print(#function, inputShapes)  // 1, 1, 64, 38, 38
        
        var outputShapes: [[NSNumber]] = []
        
        for i in 0..<inputShapes.count {
            let inputShape = inputShapes[i]
            
            let in_channel  = inputShape[2].intValue  // 64
            let in_height = inputShape[3].intValue    // 38
            let in_width = inputShape[4].intValue    // 38
            
            let out_channel = NSNumber(value: in_channel * 4)   // 256
            let out_height = NSNumber(value: in_height / 2)     // 19
            let out_width = NSNumber(value: in_width / 2)       // 19
            
            outputShapes.append([1, 1, out_channel, out_height, out_width])
        }
        
        print(#function, outputShapes)
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        accelerate_activate(inputs: inputs, outputs: outputs)
    }
    
    private func accelerate_activate(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
        for i in 0..<inputs.count {
            let input = inputs[i]
            let output = outputs[i]
            
//            print(#function, input.shape)    // [1, 1, 64, 38, 38]
//            print(#function, output.shape)   // [1, 1, 256, 19, 19]            
            
            let nc_input = Int32(truncating: input.shape[2])          // 64
            let nrow_input = Int32(truncating: input.shape[3])        // 38
            let ncol_input = Int32(truncating: input.shape[4])        // 38
            
            let nc_output = Int32(truncating: output.shape[2])        // 256
            let nrow_output = Int32(truncating: output.shape[3])      // 19
            let ncol_output = Int32(truncating: output.shape[4])      // 19
            
//            let tot_len = input.count
            
            let iptr = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
            let optr = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))
            
//            cblas_scopy(Int32(tot_len), iptr, 1, optr, 1)
            
            let b: Int32 = 2     // block_size
            
            /*
                S = will be placed at "space" level of output array
                D1 = will be placed at one depth level behind "space" (output array)
                D2 = will be placed at one depth level behind D1 (output array)
                D3 = will be placed at one depth level behind D2 (output array)
             
                    S | D1 | S | D1
                    --+----+---+---
                    D2| D3 | D2| D3
                    --+----+---+---
                    S | D1 | S | D1
                    --+----+---+---
                    D2| D3 | D2| D3
             
             */

//            var item_copied: Int32 = 0
            // (1) S
            
            var i_input_offset_for_channel = 0   // offset position for input array, for channel
            var i_output_offset_for_channel = 0  // offset position for output array, for channel
            
            var i_input: Int    // pointer to start position for input array for each cblas_scopy, which copies an entire row of data
            var i_output: Int   // pointer to start position for output array for each cblas_scopy
            
            for ic in 0..<Int(nc_input) {              // this clarifies we copy the correct number of channel from input
                
                i_input_offset_for_channel = ic * Int(nrow_input * ncol_input)
                i_output_offset_for_channel = ic * Int(nrow_output * ncol_output)
                
                for irow in 0..<nrow_output {     // this clarifies we copy the correct number of row for the output
                    i_input = Int(b * irow * ncol_input) + i_input_offset_for_channel
                    i_output = Int(irow * ncol_output) + i_output_offset_for_channel
                    cblas_scopy(ncol_output, iptr + i_input, b, optr + i_output, 1)
//                    item_copied += ncol_output
                }
            }
            
            // (2) D1
            for ic in 0..<Int(nc_input) {              // this clarifies we copy the correct number of channel from input
                
                i_input_offset_for_channel = ic * Int(nrow_input * ncol_input)
                i_output_offset_for_channel = ic * Int(nrow_output * ncol_output) + Int(nc_input * nrow_output * ncol_output)
                
                for irow in 0..<nrow_output {     // this clarifies we copy the correct number of row for the output
                    i_input = Int(b * irow * ncol_input) + i_input_offset_for_channel + 1
                    i_output = Int(irow * ncol_output) + i_output_offset_for_channel
                    cblas_scopy(ncol_output, iptr + i_input, b, optr + i_output, 1)
//                    item_copied += ncol_output
                }
            }
            
            // (3) D2
            for ic in 0..<Int(nc_input) {              // this clarifies we copy the correct number of channel from input
                
                i_input_offset_for_channel = ic * Int(nrow_input * ncol_input)
                i_output_offset_for_channel = ic * Int(nrow_output * ncol_output) + 2 * Int(nc_input * nrow_output * ncol_output)
                
                for irow in 0..<nrow_output {     // this clarifies we copy the correct number of row for the output
                    i_input = Int( (b * irow + 1) * ncol_input) + i_input_offset_for_channel
                    i_output = Int(irow * ncol_output) + i_output_offset_for_channel
                    cblas_scopy(ncol_output, iptr + i_input, b, optr + i_output, 1)
//                    item_copied += ncol_output
                }
            }
            
            // (4) D3
            for ic in 0..<Int(nc_input) {              // this clarifies we copy the correct number of channel from input
                
                i_input_offset_for_channel = ic * Int(nrow_input * ncol_input)
                i_output_offset_for_channel = ic * Int(nrow_output * ncol_output) + 3 * Int(nc_input * nrow_output * ncol_output)
                
                for irow in 0..<nrow_output {     // this clarifies we copy the correct number of row for the output
                    i_input = Int( (b * irow + 1) * ncol_input) + i_input_offset_for_channel + 1
                    i_output = Int(irow * ncol_output) + i_output_offset_for_channel
                    cblas_scopy(ncol_output, iptr + i_input, b, optr + i_output, 1)
//                    item_copied += ncol_output
                }
            }
            
//            print("item_copied: \(item_copied)")
            
        }
    }
}

