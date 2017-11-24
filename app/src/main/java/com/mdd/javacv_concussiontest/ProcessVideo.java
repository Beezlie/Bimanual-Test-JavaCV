package com.mdd.javacv_concussiontest;

import android.os.Environment;
import android.util.Log;

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.util.List;

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
        int selector = 0;
        int[] numCycles = new int[2];
        int[] dirChange = new int[2];
        int[] minPeak = {10000, 10000};
        int[] maxPeak = {0, 0};

        // Open video file
        grabber.start();

        long delay = Math.round(1000d / grabber.getFrameRate());

        // Read frame by frame
        Frame frame;
        while ((frame = grabber.grab()) != null) {
            // process the frame
            Mat mRgba = toMatConverter.convert(frame);

            int[] centroidPoints = new int[2];
            String[] dirYprev = new String[2];

            /*steps:
            1 - get mLowerBound and mUpperBound from shared prefs?
            2 - copy over the process() java code (need to find equivalent for List<MatOfPoint> for storing contours)
            3 - once contours found, get bounding rect, centroids and direction of motion
            */

            // Delay
            Thread.sleep(delay);
        }

        // Close the video file
        grabber.release();

    }

}

