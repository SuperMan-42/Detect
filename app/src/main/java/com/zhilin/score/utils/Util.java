package com.zhilin.score.utils;

import android.annotation.SuppressLint;
import android.content.ContentUris;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.text.TextUtils;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class Util {
  public static String saveBitmap(Context context, Bitmap bitmap) {
    try {
      File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_PICTURES), System.currentTimeMillis() + ".png");
      FileOutputStream out = new FileOutputStream(file);
      if (bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)) {
        out.flush();
        out.close();
      }
      return file.getPath();
    } catch (IOException e) {
      return "";
    }
  }

  public static void clearBitmap(Context context) {
    File dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    if (!dir.exists()) {
      dir.mkdirs();
    } else {
      for (File file : dir.listFiles()) {
        if (file.isFile()) {
          file.delete();
        }
      }
    }
  }

  public static Bitmap decodeBitmapFromPath(String path) {
    try {
      if (TextUtils.isEmpty(path)) {
        return null;
      }
      final BitmapFactory.Options options = new BitmapFactory.Options();
      options.inMutable = true;
      FileInputStream fis = new FileInputStream(path);
      Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fis.getFD(), null, options);
      fis.close();
      return bitmap;
    } catch (Exception e) {
      return null;
    }
  }

  public static Bitmap decodeSampledBitmapFromPath(String path, int reqWidth, int reqHeight) {
    try {
      if (TextUtils.isEmpty(path)) {
        return null;
      }
      FileInputStream fis = new FileInputStream(path);
      final BitmapFactory.Options options = new BitmapFactory.Options();
      options.inJustDecodeBounds = true;
      BitmapFactory.decodeFileDescriptor(fis.getFD(), null, options);
      options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
      options.inJustDecodeBounds = false;
      options.inPreferredConfig = Bitmap.Config.RGB_565;
      options.inDither = true;
      Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fis.getFD(), null, options);
      fis.close();
      return bitmap;
    } catch (Exception e) {
      return null;
    }
  }

  private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
    final int width = options.outWidth;
    final int height = options.outHeight;
    int inSampleSize = 1;
    if (height > reqHeight || width > reqWidth) {
      final int suitedValue = Math.max(reqHeight, reqWidth);
      final int heightRatio = Math.round((float) height / (float) suitedValue);
      final int widthRatio = Math.round((float) width / (float) suitedValue);
      inSampleSize = Math.max(heightRatio, widthRatio);
    }
    return inSampleSize;
  }

  @Nullable
  public static String uri2path(Context context, Uri uri, Uri externalContentUri) {
    String path = null;
    if (DocumentsContract.isDocumentUri(context, uri)) {
      String docId = DocumentsContract.getDocumentId(uri);
      if ("com.android.providers.media.documents".equals(uri.getAuthority())) {
        String id = docId.split(":")[1];
        String selection = MediaStore.Images.Media._ID + "=" + id;
        path = getPath(context, externalContentUri, selection);
      } else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())) {
        Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(docId));
        path = getPath(context, contentUri, null);
      }
    } else if ("content".equalsIgnoreCase(uri.getScheme())) {
      path = getPath(context, uri, null);
    }
    return path;
  }

  @SuppressLint("Range")
  private static String getPath(Context context, Uri uri, String selection) {
    String path = null;
    Cursor cursor = context.getContentResolver().query(uri, null, selection, null, null);
    if (cursor != null) {
      if (cursor.moveToFirst()) {
        path = cursor.getString(cursor.getColumnIndex("_data"));
      }
      cursor.close();
    }
    return path;
  }
}