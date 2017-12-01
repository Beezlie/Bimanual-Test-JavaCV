package com.mdd.javacv_concussiontest.utils;

import android.util.Log;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.inRange;
import static org.bytedeco.javacpp.opencv_core.multiply;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;

import java.util.ArrayDeque;
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
    private MatVector mContours = new MatVector();

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

        //TODO - figure out why spectrumHsv null after this
        for (int j = 0; j < maxH-minH; j++) {
            byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
            //spectrumHsv.put(0, j, tmp);
            spectrumHsv.data().put(tmp);
        }

        cvtColor(spectrumHsv, mSpectrum, COLOR_HSV2RGB_FULL, 4);
    }

    public void process(Mat mRgba, double pxscale) throws FrameGrabber.Exception, InterruptedException {
        String LOG_TAG = "VideoProcessing";
        double mMinContourArea = 0.1;

        // Cache
        Mat mPyrDownMat = new Mat();
        Mat mHsvMat = new Mat();
        Mat mMask = new Mat();
        Mat mDilatedMask = new Mat();
        Mat mHierarchy = new Mat();

        for (int k = 0; k < 2; k++) {
            //Gaussian Blur
            GaussianBlur(mRgba, mRgba, new Size(3, 3), 1);

            //pyrdown x2
            //TODO - figure out why mPyrDownMat null after this
            pyrDown(mRgba, mPyrDownMat);
            pyrDown(mPyrDownMat, mPyrDownMat);

            //convert RBGA to HSV colorspace
            cvtColor(mPyrDownMat, mHsvMat, COLOR_RGB2HSV_FULL);

            //threshold hsv image for color range
            //inRange(intensity, new Mat(new Size(4, 1), CV_64FC1, zeroScalar), new Mat(new Size(4, 1), CV_64FC1, black_level), mask);
            //TODO - remove this (temporarily hard code thresholds)
            if (k == 0) {
                mLowerBound = new Scalar(135.26, 163.81, 49.42, 0);
                mUpperBound = new Scalar(185.26, 263.81, 149.42, 255);
            } else {
                mLowerBound = new Scalar(221.94, 177.92, 102.55, 0);
                mUpperBound = new Scalar(255, 277.92, 202.55, 255);
            }

            //TODO I'm setting these as CV_8UC4 but this may be incorrect. CONFIRM THIS
            DoublePointer dpl = new DoublePointer(mLowerBound.get(0), mLowerBound.get(1), mLowerBound.get(2), mLowerBound.get(3));
            Mat lowerBoundMat = new Mat(1, 1, CV_8UC4, dpl);
            DoublePointer dph = new DoublePointer(mLowerBound.get(0), mLowerBound.get(1), mLowerBound.get(2), mLowerBound.get(3));
            Mat upperBoundMat = new Mat(1, 1, CV_8UC4, dph);
            //TODO - figure out why mHsvMay and mMask are both null after this
            //inRange(src, low, high, dst)
            inRange(mHsvMat, lowerBoundMat, upperBoundMat, mMask);

            //dilate
            dilate(mMask, mDilatedMask, new Mat());

            //initialize list of contours
            MatVector contours = new MatVector();

            //find contours from the dilated matrix
            findContours(mDilatedMask, contours, mHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            //find max contour area
            double maxArea = 0;

            //iterate over list of contour vectors
            for (int i = 0; i < contours.size(); i++) {
                Mat contour = contours.get(i);
                //contour - Input vector of 2D points (contour vertices)
                double area = contourArea(contour);
                if (area > maxArea)
                    maxArea = area;
            }

            //filter contours by area and resize to fit original image size
            for (int i = 0; i < contours.size(); i++) {
                Mat contour = contours.get(i);
                if (contourArea(contour) > mMinContourArea*maxArea) {
                    //multiply - Calculates the per-element scaled product of two arrays.
                    //multiply(Mat src1, Scalar src2, Mat dst)
                    //src1 - First source array.
                    //src2 - Second source array of the same size and the same type as src1.
                    //dst - Destination array of the same size and type as src1.
                    multiply(4, contour);
                    mContours.put(contour);
                }
            }
        }
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public Scalar getUpperBound() { return mUpperBound; }

    public Scalar getLowerBound() { return mLowerBound; }

    public MatVector getContours() {
        return mContours;
    }

}

