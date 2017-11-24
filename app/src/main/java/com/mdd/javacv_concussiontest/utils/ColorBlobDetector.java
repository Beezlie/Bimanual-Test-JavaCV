package com.mdd.javacv_concussiontest.utils;

import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by matt on 11/10/17.
 */

public class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);
    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;
    // Color radius for range checking in HSV color space
    private Scalar mColorRadius = new Scalar(25,50,50,0);
    private Mat mSpectrum = new Mat();
    //private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();

    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.get(0) >= mColorRadius.get(0)) ? hsvColor.get(0)-mColorRadius.get(0) : 0;
        double maxH = (hsvColor.get(0)+mColorRadius.get(0) <= 255) ? hsvColor.get(0)+mColorRadius.get(0) : 255;

        mLowerBound.put(0, minH);
        mUpperBound.put(0, maxH);

        mLowerBound.put(1, hsvColor.get(1) - mColorRadius.get(1));
        mUpperBound.put(1, hsvColor.get(1) + mColorRadius.get(1));

        mLowerBound.put(2, hsvColor.get(2) - mColorRadius.get(2));
        mUpperBound.put(2, hsvColor.get(2) + mColorRadius.get(2));

        mLowerBound.put(3, 0);
        mUpperBound.put(3, 255);

        Mat spectrumHsv = new Mat(1, (int)(maxH-minH), CV_8UC3);

        for (int j = 0; j < maxH-minH; j++) {
            byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
            //spectrumHsv.put(0, j, tmp);
            spectrumHsv.data().put(tmp);
        }

        cvtColor(spectrumHsv, mSpectrum, COLOR_HSV2RGB_FULL, 4);
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public Scalar getUpperBound() { return mUpperBound; }

    public Scalar getLowerBound() { return mLowerBound; }

}

