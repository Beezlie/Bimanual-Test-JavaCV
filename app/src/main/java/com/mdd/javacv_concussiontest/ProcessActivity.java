package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.annotation.Nullable;
import android.util.Log;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_TREE;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.circle;
import static org.bytedeco.javacpp.opencv_imgproc.cvGetSpatialMoment;
import static org.bytedeco.javacpp.opencv_imgproc.cvMoments;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class ProcessActivity extends Activity implements CvCameraPreview.CvCameraViewListener {
    final String TAG = "ProcessActivity";

    private CvCameraPreview cameraView;
    private ColorBlobDetector mDetector = new ColorBlobDetector();
    private int centroidX, centroidY;
    private opencv_core.Scalar CONTOUR_COLOR_WHITE = new opencv_core.Scalar(255,255,255,255);
    private FFmpegFrameGrabber grabber;
    private Exception exception;
    private OpenCVFrameConverter.ToMat toMatConverter;
    private Frame frame;
    private long delay;
    private int frameCounter = 0;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_real_time);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        retrieveThreshColors();
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        File savePath = new File(Environment.getExternalStorageDirectory(), "bimanualtest.mp4");
        String filename = savePath.getAbsolutePath();
        grabber = new FFmpegFrameGrabber(filename);
        toMatConverter = new OpenCVFrameConverter.ToMat();

        // Open video file
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }

        delay = Math.round(1000d / grabber.getFrameRate());
    }

    @Override
    public void onCameraViewStopped() {
        // Close the video file
        try {
            grabber.release();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }
    }

    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        // Read frame by frame
        if (frameCounter < grabber.getLengthInFrames()) {
            try {
                frame = grabber.grabFrame();
                if (frame.image == null) {
                    Log.d(TAG, "EMPTY IMAGE: " + frameCounter);
                    return rgbaMat;
                }

                // process the frame
                Mat fMat = toMatConverter.convertToMat(frame);
                mDetector.threshold(fMat);
                Mat threshed = mDetector.getThreshold();

                opencv_core.Scalar test1 = mDetector.getLowerBound();
                opencv_core.Scalar test2 = mDetector.getUpperBound();

                opencv_core.MatVector contours = new opencv_core.MatVector();
                findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

                //TODO - figure out why no contours detected
                //could it be an issue with the color?
                //because some times there is a bounding box but it will be in a random location for a random color
                if (contours.size() <= 0) {
                    return fMat;
                }

                opencv_core.RotatedRect rect = minAreaRect(contours.get(0));

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
                opencv_core.Rect boundRect = boundingRect(contours.get(boundPos));
                rectangle(fMat, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);

                opencv_imgproc.CvMoments moments = new opencv_imgproc.CvMoments();
                cvMoments(new opencv_core.IplImage(contours.get(boundPos)), moments, 1);
                double m00 = cvGetSpatialMoment(moments, 0, 0);
                double m10 = cvGetSpatialMoment(moments, 1, 0);
                double m01 = cvGetSpatialMoment(moments, 0, 1);
                if (m00 != 0) {   // calculate center
                    centroidX = (int) Math.round(m10 / m00);
                    centroidY = (int) Math.round(m01 / m00);
                }
                circle(fMat, new opencv_core.Point(centroidX, centroidY), 4, new opencv_core.Scalar(255,255,255,255));

                fMat.copyTo(rgbaMat);

                // Delay
                try {
                    Thread.sleep(delay);
                } catch (InterruptedException e) {
                    exception = e;
                }
            } catch (FrameGrabber.Exception e) {
                exception = e;
            }
        }
        frameCounter++;

        return rgbaMat;
    }

    public void retrieveThreshColors() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);

        double hueLower = (double)prefs.getInt(getString(R.string.rightHueLower), 0);
        double satLower = (double)prefs.getInt(getString(R.string.rightSatLower), 0);
        double briLower = (double)prefs.getInt(getString(R.string.rightBriLower), 0);
        double hueUpper = (double)prefs.getInt(getString(R.string.rightHueUpper), 0);
        double satUpper = (double)prefs.getInt(getString(R.string.rightSatUpper), 0);
        double briUpper = (double)prefs.getInt(getString(R.string.rightBriUpper), 0);

        opencv_core.Scalar lowerBound = new opencv_core.Scalar(hueLower, satLower, briLower, 0);
        opencv_core.Scalar upperBound = new opencv_core.Scalar(hueUpper, satUpper, briUpper, 0);

        mDetector.setLowerBound(lowerBound);
        mDetector.setUpperBound(upperBound);
    }
}
