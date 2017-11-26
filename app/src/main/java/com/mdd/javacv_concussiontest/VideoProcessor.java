package com.mdd.javacv_concussiontest;

import android.content.Context;
import android.content.SharedPreferences;
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
import static org.bytedeco.javacpp.opencv_highgui.*;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * The example for section "Reading video sequences" in Chapter 10, page 248.
 * <p>
 * This version of the example is implemented using JavaCV `FFmpegFrameGrabber`class.
 */
public class VideoProcessor {
    private final String filename;
    private final double pxscale;

    public VideoProcessor(String filename, double pxscale) {
        this.filename = filename;
        this.pxscale = pxscale;
    }

    public void main(String[] args) throws FrameGrabber.Exception, InterruptedException {

        String LOG_TAG = "VideoProcessor";
        //String filename = Environment.getExternalStorageDirectory() + "/stream.mp4";
        //String filename = args[0];

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(filename);
        OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

        //elements required for image processing
        Scalar mLowerBound = new Scalar(0);
        Scalar mUpperBound = new Scalar(0);
        Scalar CONTOUR_COLOR_WHITE = new Scalar(255,255,255,255);
        double mMinContourArea = 0.1;
        int[] numCycles = new int[2];
        int[] dirChange = new int[2];
        int[] minPeak = {10000, 10000};
        int[] maxPeak = {0, 0};
        boolean finishedTest = false;
        String[] dirY = new String[2];
        boolean[] movingY = new boolean[2];
        ArrayDeque<int[]> movingWindow = new ArrayDeque<>();
        List<Integer> amplitudesL = new ArrayList<>();
        List<Integer> amplitudesR = new ArrayList<>();

        // Cache
        Mat mPyrDownMat = new Mat();
        Mat mHsvMat = new Mat();
        Mat mMask = new Mat();
        Mat mDilatedMask = new Mat();
        Mat mHierarchy = new Mat();

        //get the stored pixel scale from shared prefs

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
                rectangle(mRgba, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);

                //get centroids of the bounding contour
                Moments mc = moments(contours.get(boundPos), false);
                int centroidx = (int) (mc.m10() / mc.m00());
                int centroidy = (int) (mc.m01() / mc.m00());
                centroidPoints[k] = centroidy;
            }

            movingWindow.addFirst(centroidPoints);
            if (movingWindow.size() > 5) {
                movingWindow.removeLast();     //limit size of deque to 30
            }

            //track y-direction movement of centre of object
            if (movingWindow.size() >= 5) {
                for (int k = 0; k < 2; k++) {
                    int dY = movingWindow.getLast()[k] - movingWindow.getFirst()[k];
                    Log.i(TAG, "dY: " + dY);
                    if (abs(dY) > 100) {         //ensure significant movement
                        int sign = Integer.signum(dY);
                        if (sign == 1) {
                            dirYprev[k] = dirY[k];
                            dirY[k] = "Up";
                        } else {
                            dirYprev[k] = dirY[k];
                            dirY[k] = "Down";
                        }
                        if (dirY[k] != dirYprev[k]) {
                            dirChange[k]++;
                        }
                        movingY[k] = true;
                        Log.i(TAG, "Moving " + dirY[k]);
                    } else {
                        movingY[k] = false;
                        Log.i(TAG, "Not Moving");
                    }

                    //check if centroidy is a max or min peak of current cycle
                    if (centroidPoints[k] > maxPeak[k])
                        maxPeak[k] = centroidPoints[k];
                    if (centroidPoints[k] < minPeak[k])
                        minPeak[k] = centroidPoints[k];

                    //reset min/max peak after each cycle
                    if (dirChange[k] != 0 && dirChange[k] % 2 == 0) {
                        dirChange[k] = 0;
                        numCycles[k]++;
                        if (k == 0) {
                            amplitudesL.add(maxPeak[k] - minPeak[k]);
                        } else {
                            amplitudesR.add(maxPeak[k] - minPeak[k]);
                        }
                        minPeak[k] = 10000;
                        maxPeak[k] = 0;
                    }
                }
            }

            Log.i(TAG, "numCycles left: " + numCycles[0]);
            Log.i(TAG, "numCycles right: " + numCycles[1]);
            //display results once 10 cycles completed
            if (numCycles[0] >= 10 && numCycles[1] >= 10 && !finishedTest) {
                for (int amp : amplitudesR) {
                    Log.i(TAG, "Right hand amplitude: " + (amp / pxscale) + " mm" + " at cycle count " + numCycles[0]);
                }
                for (int amp : amplitudesL) {
                    Log.i(TAG, "Left hand amplitude: " + (amp / pxscale) + " mm" + " at cycle count " + numCycles[0]);
                }
                finishedTest = true;
            }

            imshow("bimanual_test", mRgba);

            // Delay
            Thread.sleep(delay);
        }

        // Close the video file
        grabber.release();

    }

}

