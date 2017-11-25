package com.mdd.javacv_concussiontest;

import android.os.Environment;
import android.util.Log;

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.inRange;
import static org.bytedeco.javacpp.opencv_core.multiply;
import static org.bytedeco.javacpp.opencv_core.Moments;
import static org.bytedeco.javacpp.opencv_imgproc.moments;


import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

/**
 * The example for section "Reading video sequences" in Chapter 10, page 248.
 * <p>
 * This version of the example is implemented using JavaCV `FFmpegFrameGrabber`class.
 */
public class ProcessVideo {

    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException {

        String LOG_TAG = "ProcessVideo";
        String filename = Environment.getExternalStorageDirectory() + "/stream.mp4";

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(filename);
        OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

        //elements required for image processing
        Scalar mLowerBound = new Scalar(0);
        Scalar mUpperBound = new Scalar(0);
        double mMinContourArea = 0.1;
        int selector = 0;
        int[] numCycles = new int[2];
        int[] dirChange = new int[2];
        int[] minPeak = {10000, 10000};
        int[] maxPeak = {0, 0};

        // Cache
        Mat mPyrDownMat = new Mat();
        Mat mHsvMat = new Mat();
        Mat mMask = new Mat();
        Mat mDilatedMask = new Mat();
        Mat mHierarchy = new Mat();

        // Open video file
        grabber.start();

        long delay = Math.round(1000d / grabber.getFrameRate());

        // Read frame by frame
        Frame frame;
        while ((frame = grabber.grab()) != null) {
            // process the frame
            Mat mRgba = toMatConverter.convert(frame);

            MatVector mContours = new MatVector();
            int[] centroidPoints = new int[2];
            String[] dirYprev = new String[2];

            /*steps:
            1 - get mLowerBound and mUpperBound from shared prefs?
            2 - copy over the process() java code (need to find equivalent for List<MatOfPoint> for storing contours)
                check other javacv contour examples and see what they do
            3 - once contours found, get bounding rect, centroids and direction of motion
            */

            for (int k = 0; k < 2; k++) {
                //get the upper and lower bound for the kth blob

                //declare necessary elements (matrix, scalar, etc)

                //Gaussian Blur
                GaussianBlur(mRgba, mRgba, new Size(3, 3), 1);

                //pyrdown x2
                pyrDown(mRgba, mPyrDownMat);
                pyrDown(mPyrDownMat, mPyrDownMat);

                //convert RBGA to HSV colorspace
                cvtColor(mPyrDownMat, mHsvMat, COLOR_RGB2HSV_FULL);

                //threshold hsv image for color range
                //inRange(intensity, new Mat(new Size(4, 1), CV_64FC1, zeroScalar), new Mat(new Size(4, 1), CV_64FC1, black_level), mask);
                //TODO I'm setting these as CV_8UC4 but this may be incorrect. CONFIRM THIS
                Mat lowerBoundMat = new Mat(new Size(4, 1), CV_8UC4, mLowerBound);
                Mat upperBoundMat = new Mat(new Size(4, 1), CV_8UC4, mUpperBound);
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

                //get bounding rectangle
                RotatedRect rect = minAreaRect(contours.get(0));

                double boundWidth = rect.size().width();
                double boundHeight = rect.size().height();
                int boundPos = 0;

                //update the width and height for the bounding rectangle based on the area of each rectangle calculated from the contour list
                for (int i = 1; i < contours.size(); i++) {
                    rect = minAreaRect(contours.get(i));
                    if (rect.size().width() * rect.size().height() > boundWidth * boundHeight) {
                        boundWidth = rect.size().width();
                        boundHeight = rect.size().height();
                        //store the location in the contour list of the maximum area bounding rectangle
                        boundPos = i;
                    }
                }

                //create a new bounding rectangle from the largest contour area
                Rect boundRect = boundingRect(contours.get(boundPos));

                //get centroids of the bounding contour
                Moments mc = moments(contours.get(boundPos), false);
                int centroidx = (int) (mc.m10() / mc.m00());
                int centroidy = (int) (mc.m01() / mc.m00());
                centroidPoints[k] = centroidy;
            }

            // Delay
            Thread.sleep(delay);
        }

        // Close the video file
        grabber.release();

    }

}

