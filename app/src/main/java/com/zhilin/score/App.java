package com.zhilin.score;

import android.content.Context;
import android.text.TextUtils;
import android.util.Log;

import androidx.multidex.MultiDexApplication;

import com.zhilin.score.utils.ACache;
import com.zhilin.score.utils.FileUtils;

import java.io.File;

public class App extends MultiDexApplication {

  public static String name;
  public static Context context;
  public static File rootDir;

  @Override
  public void onCreate() {
    super.onCreate();
    context = this;
    rootDir = getFilesDir();
    name = ACache.get(this).getAsString("name", "mlsd");
    setName(this, null);
  }

  public static void setName(Context context, String code) {
    if (!TextUtils.isEmpty(code)) {
      name = code;
      ACache.get(context).put("name", name);
    }
    Log.i("hpw", "当前实验 " + name);
    initRes();
  }

  public static void initRes() {
    FileUtils.getInstance(context).copyAssetsToSD(name, name).setFileOperateCallback(new FileUtils.FileOperateCallback() {
      @Override
      public void onSuccess() {
        Log.i("hpw", "复制成功");
      }

      @Override
      public void onFailed(String error) {
        Log.i("hpw", "复制失败");
      }
    });
  }
}