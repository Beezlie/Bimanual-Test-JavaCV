package com.mdd.javacv_concussiontest.utils;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Scalar;

import static org.bytedeco.javacpp.opencv_core.CV_32SC4;
import static org.bytedeco.javacpp.opencv_core.inRange;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

/**
 * Created by matt on 11/10/17.
 */

public class ColorBlobDetector {
    String LOG_TAG = "ColorBlobDetection";
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);
    private Scalar mColorRadius = new Scalar(25,50,50,0);
    private int hueLower, hueUpper, satLower, satUpper, briLower, briUpper;
    private Mat threshed;

    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.get(0) >= mColorRadius.get(0)) ? hsvColor.get(0)-mColorRadius.get(0) : 0;
        double maxH = (hsvColor.get(0)+mColorRadius.get(0) <= 255) ? hsvColor.get(0)+mColorRadius.get(0) : 255;

        hueLower = (int)minH;
        hueUpper = (int)maxH;
        satLower = (int)(hsvColor.get(1) - mColorRadius.get(1));
        satUpper = (int)(hsvColor.get(1) + mColorRadius.get(1));
        briLower = (int)(hsvColor.get(2) - mColorRadius.get(2));
        briUpper = (int)(hsvColor.get(2) + mColorRadius.get(2));

        Scalar lowerBound = new Scalar(hueLower, satLower, briLower, 0);
        Scalar upperBound = new Scalar(hueUpper, satUpper, briUpper, 0);
        mLowerBound = lowerBound;
        mUpperBound = upperBound;
    }

    public void threshold(Mat mat) {
        Mat hsv = new Mat();
        Mat thresh = new Mat();

        GaussianBlur(mat, mat, new opencv_core.Size(3, 3), 1);

        cvtColor(mat, hsv, COLOR_RGB2HSV_FULL);

        inRange(hsv, new Mat(1, 1, CV_32SC4, new Scalar(hueLower, satLower, briLower, 0)),
                new Mat(1, 1, CV_32SC4, new Scalar(hueUpper, satUpper, briUpper, 0)), thresh);

        threshed = thresh;
    }

    public void setLowerBound(Scalar lowerBound) {
        mLowerBound = lowerBound;
        hueLower = (int)lowerBound.get(0);
        satLower = (int)lowerBound.get(1);
        briLower = (int)lowerBound.get(2);
    }

    public void setUpperBound(Scalar upperBound) {
        mUpperBound = upperBound;
        hueUpper = (int)upperBound.get(0);
        satUpper = (int)upperBound.get(1);
        briUpper = (int)upperBound.get(2);
    }

    public Mat getThreshold() { return threshed; }

    public Scalar getLowerBound() { return mLowerBound; }

    public Scalar getUpperBound() { return mUpperBound; }

}

