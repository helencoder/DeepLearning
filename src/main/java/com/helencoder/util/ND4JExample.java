package com.helencoder.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ND4J example
 *
 * Created by helencoder on 2017/12/19.
 */
public class ND4JExample {

    /**
     *

     Creating NDArrays:
      •Create a zero-initialized array: Nd4j.zeros(nRows, nCols) or Nd4j.zeros(int...)
      •Create a one-initialized array: Nd4j.ones(nRows, nCols)
      •Create a copy (duplicate) of an NDArray: arr.dup()
      •Create a row/column vector from a double[]: myRow = Nd4j.create(myDoubleArr), myCol = Nd4j.create(myDoubleArr,new int[]{10,1})
      •Create a 2d NDArray from a double[][]: Nd4j.create(double[][])
      •Stacking a set of arrays to make a larger array: Nd4j.hstack(INDArray...), Nd4j.vstack(INDArray...) for horizontal and vertical respectively
      •Uniform random NDArrays: Nd4j.rand(int,int), Nd4j.rand(int[]) etc
      •Normal(0,1) random NDArrays: Nd4j.randn(int,int), Nd4j.randn(int[])

     Determining the Size/Dimensions of an INDArray:

     The following methods are defined by the INDArray interface:
      •Get the number of dimensions: rank()
      •For 2d NDArrays only: rows(), columns()
      •Size of the ith dimension: size(i)
      •Get the size of all dimensions, as an int[]: shape()
      •Determine the total number of elements in array: arr.length()
      •See also: isMatrix(), isVector(), isRowVector(), isColumnVector()

     Getting and Setting Single Values:
      •Get the value at row i, column j: arr.getDouble(i,j)
      •Getting a values from a 3+ dimenional array: arr.getDouble(int[])
      •Set a single value in an array: arr.putScalar(int[],double)

     Scalar operations: Scalar operations take a double/float/int value and do an operation for each As with element-wise operations, there are in-place and copy operations.
      •Add a scalar: arr1.add(myDouble)
      •Substract a scalar: arr1.sub(myDouble)
      •Multiply by a scalar: arr.mul(myDouble)
      •Divide by a scalar: arr.div(myDouble)
      •Reverse subtract (scalar - arr1): arr1.rsub(myDouble)
      •Reverse divide (scalar / arr1): arr1.rdiv(myDouble)

     Element-Wise Operations: Note: there are copy (add, mul, etc) and in-place (addi, muli) operations. The former: arr1 is not modified. In the latter: arr1 is modified
      •Adding: arr1.add(arr2)
      •Subtract: arr.sub(arr2)
      •Multiply: add1.mul(arr2)
      •Divide: arr1.div(arr2)
      •Assignment (set each value in arr1 to those in arr2): arr1.assign(arr2)

     Reduction Operations (sum, etc); Note that these operations operate on the entire array. Call .doubleValue() to get a double out of the returned Number.
      •Sum of all elements: arr.sumNumber()
      •Product of all elements: arr.prod()
      •L1 and L2 norms: arr.norm1() and arr.norm2()
      •Standard deviation of all elements: arr.stdNumber()

     Linear Algebra Operations:
      •Matrix multiplication: arr1.mmul(arr2)
      •Transpose a matrix: transpose()
      •Get the diagonal of a matrix: Nd4j.diag(INDArray)
      •Matrix inverse: InvertMatrix.invert(INDArray,boolean)

     Getting Parts of a Larger NDArray: Note: all of these methods return
      •Getting a row (2d NDArrays only): getRow(int)
      •Getting multiple rows as a matrix (2d only): getRows(int...)
      •Setting a row (2d NDArrays only): putRow(int,INDArray)
      •Getting the first 3 rows, all columns: Nd4j.create(0).get(NDArrayIndex.interval(0,3),NDArrayIndex.all());

     Element-Wise Transforms (Tanh, Sigmoid, Sin, Log etc):
      •Using Transforms: Transforms.sin(INDArray), Transforms.log(INDArray), Transforms.sigmoid(INDArray) etc
      •Directly (method 1): Nd4j.getExecutioner().execAndReturn(new Tanh(INDArray))
      •Directly (method 2) Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh",INDArray))

     */

    private static final Logger log = LoggerFactory.getLogger(ND4JExample.class);

    public static void main(String[] args) {
        // 创建2x2多维数组
        INDArray arr1 = Nd4j.create(new float[]{1,2,3,4}, new int[]{2,2});
        System.out.println(arr1);   // [[1.00,  2.00], [3.00,  4.00]]

        // 通过就地运算新增标量,各元素均加1
        arr1.addi(1);
        System.out.println(arr1);   // [[2.00,  3.00], [4.00,  5.00]]

        // 创建第二个数组（_arr2_）并将其加入第一个（_arr1_）
        INDArray arr2 = Nd4j.create(new float[]{5,6,7,8},new int[]{2,2});
        arr1.addi(arr2);
        System.out.println(arr1);   // [[7.00,  9.00], [11.00,  13.00]]

        // 创建一个3*5的元素均为10的矩阵
        INDArray tens = Nd4j.zeros(3,5).addi(10);
        //INDArray tens = Nd4j.ones(3,5).addi(9);
        System.out.println(tens);

    }

}
