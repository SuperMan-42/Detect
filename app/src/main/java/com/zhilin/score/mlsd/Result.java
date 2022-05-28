package com.zhilin.score.mlsd;

import android.graphics.RectF;

import java.util.List;

public class Result {
  private List<Box> boxList;
  private List<float[][]> pointFList;

  public Result(List<Box> boxList, List<float[][]> pointFList) {
    this.boxList = boxList;
    this.pointFList = pointFList;
  }

  public List<Box> getBoxList() {
    return boxList;
  }

  public List<float[][]> getPointFList() {
    return pointFList;
  }

  public static class Box {
    private RectF box;
    private RectF box_origin;
    private float prob;

    public Box(final RectF box, RectF box_origin, final float prob) {
      this.prob = prob;
      this.box = box;
      this.box_origin = box_origin;
    }

    public RectF getBox() {
      return this.box;
    }

    public RectF getBox_origin() {
      return this.box_origin;
    }

    public float getProb() {
      return prob;
    }
  }
}
