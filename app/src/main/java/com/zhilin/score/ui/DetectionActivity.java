package com.zhilin.score.ui;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;

import com.zhilin.score.App;
import com.zhilin.score.R;
import com.zhilin.score.mlsd.MlsdDetector;
import com.zhilin.score.mlsd.Result;
import com.zhilin.score.utils.Util;

import java.io.File;
import java.io.IOException;

public class DetectionActivity extends AppCompatActivity {
  private MlsdDetector detector;
  private ImageView imageView;
  private ImageView imageView1;
  private TextView textView;
  private ConstraintLayout layout;
  private ConstraintSet constraintSet;
  private File file;
  private Bitmap originBitmap;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_detection);
    imageView = findViewById(R.id.image);
    imageView1 = findViewById(R.id.image1);
    textView = findViewById(R.id.txt);
    layout = findViewById(R.id.layout);
    constraintSet = new ConstraintSet();
    constraintSet.clone(layout);
    textView.setMovementMethod(ScrollingMovementMethod.getInstance());
    try {
      detector = MlsdDetector.getDetector(this, App.name);
      file = new File(getIntent().getStringExtra("path"));
      if (file.exists()) {
        detect(file.getPath());
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void detect(String path) throws IOException {
    originBitmap = Util.decodeBitmapFromPath(path);
    int width = originBitmap.getWidth();
    int height = originBitmap.getHeight();
    constraintSet.setDimensionRatio(R.id.image, "h," + width + ":" + height);
    constraintSet.setDimensionRatio(R.id.image1, "h," + width + ":" + height);
    constraintSet.applyTo(layout);
    //picResize
    int new_width = detector.getInputSize();
    int new_height = detector.getInputSize();
    float r = Math.min((float) new_width / width, (float) new_height / height);
    int pad_height = Math.round(height * r);
    int pad_width = Math.round(width * r);
    int dw = (new_width - pad_width) >> 1;
    int dh = (new_height - pad_height) >> 1;
    Bitmap currentImage;
    if (width == new_width && height == new_height) {
      currentImage = originBitmap;
    } else {
      currentImage = Bitmap.createScaledBitmap(originBitmap, pad_width, pad_height, false);
    }
    Result results = detector.recognizeImage(currentImage, width, height, dw, dh, r);
    detector.drawBox(results, originBitmap, imageView, imageView1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    detector.close();
    detector = null;
    if (originBitmap != null && !originBitmap.isRecycled()) {
      originBitmap.recycle();
      originBitmap = null;
    }
  }
}