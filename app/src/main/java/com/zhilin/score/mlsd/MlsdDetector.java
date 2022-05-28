package com.zhilin.score.mlsd;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;
import android.widget.ImageView;

import androidx.annotation.NonNull;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.zhilin.score.App;
import com.zhilin.score.utils.ACache;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MlsdDetector {
  private GpuDelegate gpuDelegate = null;
  private NnApiDelegate nnapiDelegate = null;
  private Interpreter tfLite;
  private final int mode;
  private final float scoreThres;
  private final float distanceThres;
  private final MappedByteBuffer tfliteModel;
  private final Tensor input_tensor;
  private final TensorImage inputImageBuffer;
  private final Outputs outputs;

  private enum Device {
    CPU,
    GPU,
    NNAPI
  }

  public static MlsdDetector getDetector(final Context context, final String name) throws IOException {
    int num_threads = Integer.parseInt(ACache.get(context).getAsString("num", String.valueOf(Math.min(8, Math.max(1, Runtime.getRuntime().availableProcessors() - 1)))));
    Device device = Device.valueOf(ACache.get(context).getAsString("core", "CPU"));
    return new MlsdDetector(context, name + File.separator + name + ".tflite", num_threads, device);
  }

  protected MlsdDetector(Context context, String modelFilename, int num_threads, Device device) throws IOException {
    mode = (int) ACache.get(context).getAsObject("mode", 1);
    scoreThres = (float) ACache.get(context).getAsObject("scoreThres", 0.2f);
    distanceThres = (float) ACache.get(context).getAsObject("distanceThres", 20.0f);
    Log.i("hpw", "num_threads " + num_threads + " device " + device + " scoreThres " + scoreThres + " distanceThres " + distanceThres);
    Interpreter.Options options = new Interpreter.Options();
    File file = new File(App.rootDir, modelFilename);
    if (file.exists()) {
      FileInputStream inputStream = new FileInputStream(file);
      this.tfliteModel = inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, inputStream.available());
      inputStream.close();
    } else {
      this.tfliteModel = FileUtil.loadMappedFile(context, modelFilename);
    }
    switch (device) {
      case NNAPI:
        this.nnapiDelegate = new NnApiDelegate();
        options.addDelegate(this.nnapiDelegate);
        break;
      case GPU:
        CompatibilityList compatList = new CompatibilityList();
        if (compatList.isDelegateSupportedOnThisDevice()) {
          GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
          delegateOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
          this.gpuDelegate = new GpuDelegate(delegateOptions);
          options.addDelegate(this.gpuDelegate);
        } else {
          options.setUseXNNPACK(true);
        }
        break;
      case CPU:
        options.setUseXNNPACK(true);
        break;
    }
    options.setNumThreads(num_threads);
    this.tfLite = new Interpreter(this.tfliteModel, options);
    this.input_tensor = this.tfLite.getInputTensor(0);
    this.inputImageBuffer = new TensorImage(this.input_tensor.dataType());
    this.outputs = new Outputs(this.tfLite);
    //init python
    if (!Python.isStarted()) {
      Python.start(new AndroidPlatform(context));
    }
  }

  private ByteBuffer loadImage(Bitmap bitmap) {
    inputImageBuffer.load(bitmap);
    ImageProcessor.Builder builder = new ImageProcessor.Builder()
      .add(mode == 0 ? new ResizeWithCropOrPadOp(getInputSize(), getInputSize()) : new ResizeOp(getInputSize(), getInputSize(), ResizeOp.ResizeMethod.BILINEAR));
    Bitmap b = builder.build().process(inputImageBuffer).getBitmap();
    int w = b.getWidth();
    int h = b.getHeight();
    ByteBuffer imgData = ByteBuffer.allocateDirect(w * h * 4 * 4);
    imgData.order(ByteOrder.nativeOrder());
    int[] intValues = new int[w * h];
    b.getPixels(intValues, 0, w, 0, 0, w, h);
    imgData.rewind();
    for (int i = 0; i < w; ++i) {
      for (int j = 0; j < h; ++j) {
        int pixelValue = intValues[i * w + j];
        imgData.putFloat(Color.red(pixelValue));
        imgData.putFloat(Color.green(pixelValue));
        imgData.putFloat(Color.blue(pixelValue));
        imgData.putFloat(Color.alpha(pixelValue));
//        imgData.putFloat((pixelValue >> 16) & 0xFF);
//        imgData.putFloat((pixelValue >> 8) & 0xFF);
//        imgData.putFloat(pixelValue & 0xFF);
//        imgData.putFloat(pixelValue >>> 24);//or (pixelValue >> 24) & 0xFF
      }
    }
    return (ByteBuffer) imgData.rewind();
  }

  public Result recognizeImage(Bitmap bitmap, int origin_width, int origin_height, int dw, int dh, float r) {
    tfLite.runForMultipleInputsOutputs(new Object[]{loadImage(bitmap)}, outputs.getBuffer());
    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
    TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
    TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
    int[] feature0Shape = outputs.getFeature0Shape();
    int[] feature1Shape = outputs.getFeature1Shape();
    int[] feature2Shape = outputs.getFeature2Shape();
    ByteBuffer outputFeature0Buffer0 = outputFeature0.getBuffer();
    outputFeature0Buffer0.rewind();
    int[][] pts = new int[feature0Shape[1]][feature0Shape[2]];
    for (int i = 0; i < feature0Shape[1]; ++i) {
      for (int j = 0; j < feature0Shape[2]; ++j) {
        pts[i][j] = outputFeature0Buffer0.getInt();
      }
    }
    ByteBuffer outputFeature0Buffer1 = outputFeature1.getBuffer();
    outputFeature0Buffer1.rewind();
    float[] pts_score = new float[feature1Shape[1]];
    for (int i = 0; i < feature1Shape[1]; ++i) {
      pts_score[i] = outputFeature0Buffer1.getFloat();
    }
    ByteBuffer outputFeature0Buffer2 = outputFeature2.getBuffer();
    outputFeature0Buffer2.rewind();
    float[] start = new float[2];
    float[] end = new float[2];
    float[][][] vmap = new float[feature2Shape[1]][feature2Shape[2]][feature2Shape[3]];
    float[][] dist_map = new float[feature2Shape[1]][feature2Shape[2]];
    for (int i = 0; i < feature2Shape[1]; ++i) {
      for (int j = 0; j < feature2Shape[2]; ++j) {
        for (int k = 0; k < feature2Shape[3]; ++k) {
          vmap[i][j][k] = outputFeature0Buffer2.getFloat();
          if (k < 2) {
            start[k] = vmap[i][j][k];
          } else {
            end[k - 2] = vmap[i][j][k];
          }
        }
        dist_map[i][j] = (float) Math.sqrt((start[1] - end[1]) * (start[1] - end[1]) + (start[0] - end[0]) * (start[0] - end[0]));
      }
    }
    //处理所有线
    List<Result.Box> segments_list = new ArrayList<>();
    for (int i = 0; i < feature1Shape[1]; ++i) {
      int y = pts[i][0];
      int x = pts[i][1];
      float score = pts_score[i];
      float distance = dist_map[y][x];
      if (score > scoreThres && distance > distanceThres) {
        RectF rectF = new RectF(x + vmap[y][x][0], y + vmap[y][x][1], x + vmap[y][x][2], y + vmap[y][x][3]);
        float left, right, top, bootom;
        if (mode == 0) {
          left = Math.max(0, Math.min(origin_width, (rectF.left * 2 - dw) / r));
          right = Math.max(0, Math.min(origin_width, (rectF.right * 2 - dw) / r));
          top = Math.max(0, Math.min(origin_height, (rectF.top * 2 - dh) / r));
          bootom = Math.max(0, Math.min(origin_height, (rectF.bottom * 2 - dh) / r));
        } else {
          left = Math.max(0, Math.min(origin_width, rectF.left * 2 / getInputSize() * origin_width));
          right = Math.max(0, Math.min(origin_width, rectF.right * 2 / getInputSize() * origin_width));
          top = Math.max(0, Math.min(origin_height, rectF.top * 2 / getInputSize() * origin_height));
          bootom = Math.max(0, Math.min(origin_height, rectF.bottom * 2 / getInputSize() * origin_height));
        }
        RectF box_origin = new RectF(left, top, right, bootom);
        segments_list.add(new Result.Box(rectF, box_origin, score));
      }
    }
    //处理出现的矩形
    Python py = Python.getInstance();
    List<PyObject> squares = py.getModule("mlsd").callAttr("detect", segments_list).asList();
    if (squares.size() > 0) {
      List<float[][]> result = new ArrayList<>();
      for (PyObject pyObject : squares) {
        result.add(pyObject.toJava(float[][].class));
      }
      return new Result(segments_list, result);
    } else {
      return new Result(segments_list, null);
    }
  }

  public int getInputSize() {
    return input_tensor.shape()[1];
  }

  public void drawBox(Result result, Bitmap mutableBitmap, @NonNull ImageView imageView, ImageView imageView1) {
    if (result == null || result.getBoxList().size() <= 0) {
      imageView.setImageBitmap(mutableBitmap);
      imageView1.setImageBitmap(mutableBitmap);
      return;
    }
    float strokeWidth = ((mutableBitmap.getWidth() + mutableBitmap.getHeight() + 3) >> 1) * 0.004f;
    List<float[][]> points = result.getPointFList();
    if (points != null && points.size() > 0) {
      Bitmap bitmap = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true);
      Canvas canvas = new Canvas(bitmap);
      Paint mPaint = new Paint();
      mPaint.setAntiAlias(true);
      mPaint.setStrokeWidth(strokeWidth * 5);
      float[][] pointFList = points.get(0);
//      for (float[][] pointFList : points) {
        mPaint.setColor(Color.RED);
        float[] first = pointFList[0];
        float[] last = pointFList[0];
        for (int i = 1; i < pointFList.length; ++i) {
          canvas.drawLine(last[0], last[1], pointFList[i][0], pointFList[i][1], mPaint);
          last = pointFList[i];
        }
        canvas.drawLine(last[0], last[1], first[0], first[1], mPaint);
        mPaint.setColor(Color.GREEN);
        for (float[] pointF : pointFList) {
          canvas.drawCircle(pointF[0], pointF[1], strokeWidth * 5, mPaint);
        }
//      }
      imageView.setImageBitmap(bitmap);
    } else {
      imageView.setImageBitmap(mutableBitmap);
    }
    List<Result.Box> boxList = result.getBoxList();
    Bitmap bitmap1 = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true);
    Canvas canvas1 = new Canvas(bitmap1);
    Paint mPaint1 = new Paint();
    mPaint1.setColor(Color.RED);
    mPaint1.setAntiAlias(true);
    mPaint1.setStrokeWidth(strokeWidth);
    for (Result.Box box : boxList) {
      RectF rectF = box.getBox_origin();
      canvas1.drawLine(rectF.left, rectF.top, rectF.right, rectF.bottom, mPaint1);
    }
    imageView1.setImageBitmap(bitmap1);
  }

  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }
    if (nnapiDelegate != null) {
      nnapiDelegate.close();
      nnapiDelegate = null;
    }
  }
}