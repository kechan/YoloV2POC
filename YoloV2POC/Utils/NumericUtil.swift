//
//  Numeric.swift
//  CoinSense
//
//  Created by Kelvin C on 11/16/18.
//  Copyright © 2018 Kelvin Chan. All rights reserved.
//

import Foundation
import Accelerate
import Vision

enum Math {
    static func unsafePointer<T>(_ x: MLMultiArray, typeTemplate: T) -> UnsafeMutablePointer<T> {
        return UnsafeMutablePointer<T>(OpaquePointer(x.dataPointer))
    }
    
    static func unsafePointer(_ x: MLMultiArray) -> UnsafeMutablePointer<Float> {
        return UnsafeMutablePointer<Float>(OpaquePointer(x.dataPointer))
    }
    
    /*
     Matrix multiplication: C = A * B
     M: Number of rows in matrices A and C.
     N: Number of columns in matrices B and C.
     K: Number of columns in matrix A; number of rows in matrix B.
     */
    static func matmul(_ A: UnsafePointer<Float>, _ B: UnsafePointer<Float>, _ C: UnsafeMutablePointer<Float>, _ M: Int, _ N: Int, _ K: Int) {
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N),
                    Int32(K), 1, A, Int32(K), B, Int32(N), 0, C, Int32(N))
    }
    
    static func matmul(_ A: UnsafePointer<Double>, _ B: UnsafePointer<Double>, _ C: UnsafeMutablePointer<Double>, _ M: Int, _ N: Int, _ K: Int) {
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N),
                    Int32(K), 1, A, Int32(K), B, Int32(N), 0, C, Int32(N))
    }
    
    static func sum(_ x: UnsafeMutablePointer<Float>, _ count: Int) -> Float {
        var y: Float = 0
        vDSP_sve(x, 1, &y, vDSP_Length(count))
        return y
    }
    
    static func divide(_ x: UnsafeMutablePointer<Float>, _ y: inout Float, _ count: Int) {
        vDSP_vsdiv(x, 1, &y, x, 1, vDSP_Length(count))
    }
    
    static func divide(_ x: UnsafeMutablePointer<Double>, _ y: inout Double, _ count: Int) {
        vDSP_vsdivD(x, 1, &y, x, 1, vDSP_Length(count))
    }
    
    /*
     vector + scalar
     */
    static func add(_ x: UnsafeMutablePointer<Float>, _ y: inout Float, _ count: UInt) {
        vDSP_vsadd(x, 1, &y, x, 1, vDSP_Length(count))
    }
    
    static func add(_ x: UnsafeMutablePointer<Double>, _ y: inout Double, _ count: UInt) {
        vDSP_vsaddD(x, 1, &y, x, 1, vDSP_Length(count))
    }
    
    /*
     Element-wise maximum (np.maximum)
     */
    static func maximum(_ x: UnsafeMutablePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ z: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vmax(x, 1, y, 1, z, 1, count)
    }
    
    static func maximum(_ x: UnsafeMutablePointer<Double>, _ y: UnsafeMutablePointer<Double>, _ z: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vmaxD(x, 1, y, 1, z, 1, count)
    }
    
    /*
     Element-wise minimum (np.maximum)
     */
    static func minimum(_ x: UnsafeMutablePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ z: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vmin(x, 1, y, 1, z, 1, count)
    }
    
    static func minimum(_ x: UnsafeMutablePointer<Double>, _ y: UnsafeMutablePointer<Double>, _ z: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vminD(x, 1, y, 1, z, 1, count)
    }
    
    /*
     Element-wise multiply
     */
    static func multiply(_ a: UnsafeMutablePointer<Float>, _ b: UnsafeMutablePointer<Float>, _ c: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vmul(a, 1, b, 1, c, 1, count)
    }
    
    static func multiply(_ a: UnsafeMutablePointer<Double>, _ b: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vmulD(a, 1, b, 1, c, 1, count)
    }
    
    /*
     Element-wise add
     */
    static func add(_  a: UnsafeMutablePointer<Float>, _ b: UnsafeMutablePointer<Float>, _ c: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vadd(a, 1, b, 1, c, 1, count)
    }
    
    static func add(_  a: UnsafeMutablePointer<Double>, _ b: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vaddD(a, 1, b, 1, c, 1, count)
    }
    
    /*
     Element-wise subtract
     */
    static func minus(_ b: UnsafeMutablePointer<Float>, _ a: UnsafeMutablePointer<Float>, _ c: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vsub(b, 1, a, 1, c, 1, count)
    }
    
    static func minus(_ b: UnsafeMutablePointer<Double>, _ a: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vsubD(b, 1, a, 1, c, 1, count)
    }
    
    /*
     Element-wise divide
     */
    static func divide(_ b: UnsafeMutablePointer<Float>, _ a: UnsafeMutablePointer<Float>, _ c: UnsafeMutablePointer<Float>, _ count: UInt) {
        vDSP_vdiv(b, 1, a, 1, c, 1, count)
    }
    
    static func divide(_ b: UnsafeMutablePointer<Double>, _ a: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>, _ count: UInt) {
        vDSP_vdivD(b, 1, a, 1, c, 1, count)
    }
}

enum Geometry {
    static func tile(_ x: UnsafeMutablePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ x_count: UInt, _ tile_count: UInt) {
        // vDSP_mmov (copying)
        // Paramters
        // x: the vector/matrix copying from
        // y: the vector/matrix copying to
        // x_count: length of vector x (copying from)
        // tile_count: how many to repeat
        
        let n = x_count * tile_count
        
        for i in 0..<tile_count {
            vDSP_mmov(x, y + Int(i*x_count), x_count, 1, x_count, n)
        }
    }
    
    static func tile(_ x: UnsafeMutablePointer<Double>, _ y: UnsafeMutablePointer<Double>, _ x_count: UInt, _ tile_count: UInt) {
        // See comment for single precision version
        let n = x_count * tile_count
        
        for i in 0..<tile_count {
            vDSP_mmovD(x, y + Int(i*x_count), x_count, 1, x_count, n)
        }
    }
    
    static func iou(_ box1: UnsafeMutablePointer<Double>, _ box2: UnsafeMutablePointer<Double>, count n_boxes: UInt) -> [Double] {
        
        let box1_broadcasted = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes*4))
        
        tile(box1, box1_broadcasted, 4, n_boxes)
        
        let intersect_mins = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes * 4))
        
        Math.maximum(box1_broadcasted, box2, intersect_mins, 4 * n_boxes)
        
        var tmp: [Double] = [1, 1, 0, 0]
        let mask = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes * 4))
        
        tile(&tmp, mask, 4, n_boxes)
        
        Math.multiply(intersect_mins, mask, intersect_mins, 4 * n_boxes)
        
        let intersect_maxes = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes * 4))
        
        Math.minimum(box1_broadcasted, box2, intersect_maxes, 4 * n_boxes)
        
        var tmp2: [Double] = [0, 0, 1, 1]
        let mask2 = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes * 4))
        tile(&tmp2, mask2, 4, n_boxes)
        
        Math.multiply(intersect_maxes, mask2, intersect_maxes, 4 * n_boxes)
        
        vDSP_mmovD(intersect_maxes+2, intersect_maxes, 2, n_boxes, 4, 4)
        
        Math.multiply(intersect_maxes, mask, intersect_maxes, 4 * n_boxes)
        
        // intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        Math.minus(intersect_mins, intersect_maxes, intersect_maxes, 4 * n_boxes)
        
        var zero: Double = 0
        vDSP_vthresD(intersect_maxes, 1, &zero, intersect_maxes, 1, 4 * n_boxes)
        
        let intersect_wh = intersect_maxes
        
        // intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        let intersect_area = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        
        vDSP_vmulD(intersect_wh, 4, intersect_wh+1, 4, intersect_area, 1, n_boxes)
        
        //        var str="intersect_area: \n"
        //        for k in 0..<n_boxes {
        //            str += "\(intersect_area[Int(k)]), "
        //        }
        //        print(str)
        
        // calculate area of box1
        var area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        // calculate areas of box2
        
        let box2_height = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        let box2_width = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        let box2_area = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        
        vDSP_vsubD(box2, 4, box2+2, 4, box2_height, 1, n_boxes)
        vDSP_vsubD(box2+1, 4, box2+3, 4, box2_width, 1, n_boxes)
        
        Math.multiply(box2_height, box2_width, box2_area, n_boxes)
        
        // calculate: union_areas = areas1 + areas2 - intersect_areas
        //        let _union_area = try! MLMultiArray(shape: [n, 1], dataType: MLMultiArrayDataType.float32)
        //        let union_area = Math.unsafePointer(_union_area)
        let union_area = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        
        Math.minus(intersect_area, box2_area, union_area, n_boxes)
        Math.add(union_area, &area1, n_boxes)
        
        // calculate: iou_scores = intersect_areas / union_areas
        //        let _iou_scores = try! MLMultiArray(shape: [n, 1], dataType: MLMultiArrayDataType.float32)
        //        let iou_scores = Math.unsafePointer(_iou_scores)
        let iou_scores = UnsafeMutablePointer<Double>.allocate(capacity: Int(n_boxes))
        
        Math.divide(union_area, intersect_area, iou_scores, n_boxes)
        
        let results = Array(UnsafeBufferPointer(start: iou_scores, count: Int(n_boxes)))
        
        iou_scores.deallocate()
        
        return results
    }
}

enum UnitTest {
    static func testing() {
        // vDSP_mmov (copying)
        let _AA = try! MLMultiArray(shape: [1, 2], dataType: MLMultiArrayDataType.float32)
        _AA[0] = 1.0; _AA[1] = 1.0
        //var AA = UnsafeMutablePointer<Float32>(OpaquePointer(_AA.dataPointer))
        
        let AA = Math.unsafePointer(_AA, typeTemplate: Float(0))
        
        let _CC = try! MLMultiArray(shape: [1, 6], dataType: MLMultiArrayDataType.float32)
        
        let CC = Math.unsafePointer(_CC, typeTemplate: Float32(0))
        
        Geometry.tile(AA, CC, 2, 3)
        
        print(CC)
    }
}


extension UnsafeMutablePointer {
    func sum(count: Int) -> Pointee {
        if self.pointee is Float {
            var y: Float = 0
            let x = self as! UnsafeMutablePointer<Float>
            vDSP_sve(x, 1, &y, vDSP_Length(count))
            return y as! Pointee
        }
        else {
            return 0 as! Pointee
        }
    }
}
