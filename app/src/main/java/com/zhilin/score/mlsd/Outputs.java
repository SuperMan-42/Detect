package com.zhilin.score.mlsd;

import androidx.annotation.NonNull;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.util.HashMap;
import java.util.Map;

public class Outputs {
  private int[] shape0, shape1, shape2;
  private TensorBuffer outputFeature0;
  private TensorBuffer outputFeature1;
  private TensorBuffer outputFeature2;

  public Outputs(Interpreter interpreter) {
    shape0 = interpreter.getOutputTensor(0).shape();
    shape1 = interpreter.getOutputTensor(1).shape();
    shape2 = interpreter.getOutputTensor(2).shape();
    this.outputFeature0 = TensorBuffer.createFixedSize(shape0, DataType.FLOAT32);
    this.outputFeature1 = TensorBuffer.createFixedSize(shape1, DataType.FLOAT32);
    this.outputFeature2 = TensorBuffer.createFixedSize(shape2, DataType.FLOAT32);
  }

  @NonNull
  public TensorBuffer getOutputFeature0AsTensorBuffer() {
    return outputFeature0;
  }

  @NonNull
  public TensorBuffer getOutputFeature1AsTensorBuffer() {
    return outputFeature1;
  }

  @NonNull
  public TensorBuffer getOutputFeature2AsTensorBuffer() {
    return outputFeature2;
  }

  public int[] getFeature0Shape() {
    return shape0;
  }

  public int[] getFeature1Shape() {
    return shape1;
  }

  public int[] getFeature2Shape() {
    return shape2;
  }

  @NonNull
  public Map<Integer, Object> getBuffer() {
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, outputFeature0.getBuffer());
    outputs.put(1, outputFeature1.getBuffer());
    outputs.put(2, outputFeature2.getBuffer());
    return outputs;
  }
}