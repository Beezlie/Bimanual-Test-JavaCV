package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.annotation.Nullable;
import android.util.Log;
import android.widget.Toast;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.sumElems;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_TREE;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.circle;
import static org.bytedeco.javacpp.opencv_imgproc.cvGetSpatialMoment;
import static org.bytedeco.javacpp.opencv_imgproc.cvMoments;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class ProcessActivity extends Activity implements CvCameraPreview.CvCameraViewListener {
    private final String TAG = "ProcessActivity";
    private String measurement;
    private CvCameraPreview cameraView;
    private ColorBlobDetector mDetectorL = new ColorBlobDetector();
    private ColorBlobDetector mDetectorR = new ColorBlobDetector();
    private List<ColorBlobDetector> mDetectorList = new ArrayList<>();
    private int centroidX, centroidY;
    private opencv_core.Scalar CONTOUR_COLOR_WHITE = new opencv_core.Scalar(255,255,255,255);
    private FFmpegFrameGrabber grabber;
    private Exception exception;
    private OpenCVFrameConverter.ToMat toMatConverter;
    private Frame frame;
    private long delay;
    private float pxscale;
    private int frameCounter = 0;
    private int numFrames;
    private File root;
    private File amplitudeData;
    private File boundingData;
    private FileWriter amplitudeWriter;
    private FileWriter boundingWriter;
    private int numCycles;
    private int dirChange;
    private int minPeak = 1000;
    private int maxPeak = 0;
    private boolean movingY;
    private String dirY;
    private String dirYprev;
    private ArrayDeque<Integer> movingWindow = new ArrayDeque<>();
    private List<Integer> amplitudes = new ArrayList<>();

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_process);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        root = new File(Environment.getExternalStorageDirectory().toString());
        amplitudeData = new File(root, "amplitudes.txt");
        try {
            amplitudeWriter = new FileWriter(amplitudeData);
        } catch (IOException e) {
            exception = e;
        }
        boundingData = new File(root, "boundingbox.txt");
        try {
            boundingWriter = new FileWriter(boundingData);
        } catch (IOException e) {
            exception = e;
        }
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        File savePath = new File(Environment.getExternalStorageDirectory(), "bimanualtest.mp4");
        String filename = savePath.getAbsolutePath();
        grabber = new FFmpegFrameGrabber(filename);
        toMatConverter = new OpenCVFrameConverter.ToMat();
        mDetectorList.add(mDetectorL);
        mDetectorList.add(mDetectorR);

        getThreshColors();
        pxscale = getPixelScale();

        try {
            String s = "Beginning of Test - Measurements in " + measurement;
            boundingWriter.append(s);
            boundingWriter.append("\n\r");
            amplitudeWriter.append(s);
            amplitudeWriter.append("\n\r");
        } catch (IOException e) {
            exception = e;
        }

        // Open video file
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }

        delay = Math.round(1000d / grabber.getFrameRate());
        numFrames = grabber.getLengthInFrames();
    }

    @Override
    public void onCameraViewStopped() {
        try {
            boundingWriter.append("Test Completed");
            boundingWriter.flush();
            boundingWriter.close();
            amplitudeWriter.append("Test Completed");
            amplitudeWriter.flush();
            amplitudeWriter.close();
        } catch (IOException e) {
            exception = e;
        }

        // Close the video file
        try {
            grabber.release();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }
    }

    @Override
    public Mat onCameraFrame(Mat mRgba) {
        // Read frame by frame
        if (frameCounter < numFrames) {
            try {
                frame = grabber.grabFrame();
                if (frame.image == null) {
                    Log.d(TAG, "EMPTY IMAGE: " + frameCounter);
                    return mRgba;
                }
                frameCounter++;
                Mat fMat = toMatConverter.convertToMat(frame);

                //TODO - remove this test code later
                //************************************
                opencv_core.Rect touchedRect = new opencv_core.Rect();
                int x = 200;
                int y = 200;
                int tmpx = (x>5) ? x-5 : 0;
                int tmpy = (y>5) ? y-5 : 0;
                touchedRect.x(tmpx);// = (x>5) ? x-5 : 0;
                touchedRect.y(tmpy);// = (y>5) ? y-5 : 0;
                int cols = fMat.cols();
                int rows = fMat.rows();
                int tmpw = (x+5 < cols) ? x + 5 - touchedRect.x() : cols - touchedRect.x();
                int tmph = (y+5 < rows) ? y + 5 - touchedRect.y() : rows - touchedRect.y();
                touchedRect.width(tmpw);
                touchedRect.height(tmph);

                // Calculate average hsv color of touched region
                Mat touchedRegionRgba = new Mat(fMat, touchedRect);
                Mat touchedRegionHsv = new Mat();
                cvtColor(touchedRegionRgba, touchedRegionHsv, COLOR_RGB2HSV_FULL);
                Scalar mBlobColorHsv = sumElems(touchedRegionHsv);
                int pointCount = touchedRect.width()*touchedRect.height();
                for (int i = 0; i < 3; i++) {
                    mBlobColorHsv.put(i, mBlobColorHsv.get(i) / pointCount);
                }
                //******************************************************

                //process(fMat).copyTo(mRgba);
                fMat.copyTo(mRgba);
                circle(mRgba, new opencv_core.Point(x, y), 4, new opencv_core.Scalar(255, 255, 255, 255));
                fMat.release();
            } catch (FrameGrabber.Exception e) {
                exception = e;
            }
        }

        return mRgba;
    }

    private Mat process(Mat mRgba) {
        for (int k = 0; k < 2; k++) {
            mDetectorList.get(k).threshold(mRgba);
            Mat threshed = mDetectorList.get(k).getThreshold();

            MatVector contours = new MatVector();
            findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

            //TODO - figure out why no contours detected
            //problem is with the Hue value
            //there is a difference between hue value calculated in Record/Real-Time vs the one in Process
            //the process calculated hue is about 60 lower (for example - 240 vs. 185)
            //Why would this be???
            if (contours.size() > 0) {
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
                opencv_core.Rect boundRect = boundingRect(contours.get(boundPos));
                rectangle(mRgba, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);
                String w = Float.toString(boundRect.width() / pxscale);
                String h = Float.toString(boundRect.height() / pxscale);
                String br = "(" + (boundRect.br().x() / pxscale) + "," + (boundRect.br().y() / pxscale) + ")";
                String tl = "(" + (boundRect.tl().x() / pxscale) + "," + (boundRect.tl().y() / pxscale) + ")";
                String box = "Bounding box " + k + ": width = " + w + ", height = " + h +
                        ", bottom right = " + br + ", top left = " + tl;
                Log.i(TAG, box);
                try {
                    boundingWriter.append(box);
                    boundingWriter.append("\n\r");
                } catch (IOException e) {
                    exception = e;
                }

                //get centroid of object for y-direction motion tracking
                opencv_imgproc.CvMoments moments = new opencv_imgproc.CvMoments();
                cvMoments(new opencv_core.IplImage(contours.get(boundPos)), moments, 1);
                double m00 = cvGetSpatialMoment(moments, 0, 0);
                double m10 = cvGetSpatialMoment(moments, 1, 0);
                double m01 = cvGetSpatialMoment(moments, 0, 1);
                if (m00 != 0) {   // calculate center
                    centroidX = (int) Math.round(m10 / m00);
                    centroidY = (int) Math.round(m01 / m00);
                }
                circle(mRgba, new opencv_core.Point(centroidX, centroidY), 4, new opencv_core.Scalar(255, 255, 255, 255));

                trackMotion(contours, centroidY, k);
            }
        }

        return mRgba;
    }

    private void trackMotion(opencv_core.MatVector contours, int centroidY, int objectNum) {
        if (contours.size() > 0) {
            movingWindow.addFirst(centroidY);
            if (movingWindow.size() > 5) {
                movingWindow.removeLast();     //limit size of deque to 5
            }

            //track y-direction movement of centre of object
            if (movingWindow.size() >= 5) {
                int dY = movingWindow.getLast() - movingWindow.getFirst();
                Log.i(TAG, "dY: " + dY);
                if (abs(dY) > 100) {         //ensure significant movement
                    int sign = Integer.signum(dY);
                    if (sign == 1) {
                        dirYprev = dirY;
                        dirY = "Up";
                    } else {
                        dirYprev = dirY;
                        dirY = "Down";
                    }
                    if (dirY != dirYprev) {
                        dirChange++;
                    }
                    movingY = true;
                    Log.i(TAG, "Moving " + dirY);
                } else {
                    movingY = false;
                    Log.i(TAG, "Not Moving");
                }

                //check if centroidy is a max or min peak of current cycle
                if (centroidY > maxPeak)
                    maxPeak = centroidY;
                if (centroidY < minPeak)
                    minPeak = centroidY;

                //reset min/max peak after each cycle
                if (dirChange != 0 && dirChange % 2 == 0) {
                    int amp = maxPeak - minPeak;
                    amplitudes.add(amp);

                    minPeak = 10000;
                    maxPeak = 0;
                    dirChange = 0;
                    numCycles++;

                    String result = "Object " + objectNum + " Amplitude = " + (amp/pxscale) + ", at cycle number " + numCycles;
                    Log.i(TAG, result);
                    try {
                        amplitudeWriter.append(result);
                        amplitudeWriter.append("\n\r");
                    } catch (IOException e) {
                        exception = e;
                    }
                }

            }
        }
    }

    public void getThreshColors() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);

        double hueLower = (double)prefs.getInt(getString(R.string.leftHueLower), 0);
        double satLower = (double)prefs.getInt(getString(R.string.leftSatLower), 0);
        double briLower = (double)prefs.getInt(getString(R.string.leftBriLower), 0);
        double hueUpper = (double)prefs.getInt(getString(R.string.leftHueUpper), 0);
        double satUpper = (double)prefs.getInt(getString(R.string.leftSatUpper), 0);
        double briUpper = (double)prefs.getInt(getString(R.string.leftBriUpper), 0);
        Scalar lowerBound = new Scalar(hueLower, satLower, briLower, 0);
        Scalar upperBound = new Scalar(hueUpper, satUpper, briUpper, 0);
        mDetectorList.get(0).setLowerBound(lowerBound);
        mDetectorList.get(0).setUpperBound(upperBound);

        hueLower = (double)prefs.getInt(getString(R.string.rightHueLower), 0);
        satLower = (double)prefs.getInt(getString(R.string.rightSatLower), 0);
        briLower = (double)prefs.getInt(getString(R.string.rightBriLower), 0);
        hueUpper = (double)prefs.getInt(getString(R.string.rightHueUpper), 0);
        satUpper = (double)prefs.getInt(getString(R.string.rightSatUpper), 0);
        briUpper = (double)prefs.getInt(getString(R.string.rightBriUpper), 0);
        lowerBound = new Scalar(hueLower, satLower, briLower, 0);
        upperBound = new Scalar(hueUpper, satUpper, briUpper, 0);
        mDetectorList.get(1).setLowerBound(lowerBound);
        mDetectorList.get(1).setUpperBound(upperBound);
    }

    private float getPixelScale() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);

        float pxscale = prefs.getFloat(getString(R.string.pxscale), 1);
        if (pxscale == 1) {
            measurement = "pixels";
        } else {
            measurement = "mm";
        }

        return pxscale;
    }
}
