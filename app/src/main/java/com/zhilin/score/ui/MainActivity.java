package com.zhilin.score.ui;

import android.content.Intent;
import android.os.Bundle;
import android.provider.MediaStore;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.zhilin.score.R;
import com.zhilin.score.utils.Util;

public class MainActivity extends AppCompatActivity {

  private ActivityResultLauncher<String> mLauncherAlbum;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    findViewById(R.id.bt_detect).setOnClickListener(view -> mLauncherAlbum.launch("image/*"));
    mLauncherAlbum = registerForActivityResult(new ActivityResultContracts.GetContent(), result -> {
      if (result != null) {
        String imagePath = Util.uri2path(this, result, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivity(new Intent(this, DetectionActivity.class).putExtra("path", imagePath));
      }
    });
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
  }
}