package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.preference.PreferenceManager;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

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
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.Scalar;
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

public class ProcessingActivity extends Activity implements View.OnClickListener, View.OnTouchListener, CvCameraPreview.CvCameraViewListener {
    private final String TAG = "ProcessingActivity";
    private String measurement;
    private CvCameraPreview cameraView;
    private Button startButton;
    private TextView text;
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
    private int frameCounter = 1;
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
    private Mat selectMat = new Mat();
    private boolean processing = false;
    private int selector;
    private boolean[] mIsColorSelected = new boolean[2];
    private boolean[] drawSelectionBox = new boolean[2];
    private final Handler mHandler = new Handler();

    final Runnable updateText = new Runnable() {
        public void run() {
            updateTextFinished();
        }
    };
    public void updateTextFinished() {
        if (processing) {
            text.setText("Processing");
        } else {
            text.setText("Processing complete.  Please check device storage for results file");
        }
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_processing);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        cameraView.setOnTouchListener(this);
        startButton = (Button)findViewById(R.id.start);
        startButton.setText("Start");
        startButton.setOnClickListener(this);
        text = (TextView)findViewById(R.id.instruction);

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
        if (frameCounter != 2 || processing) {
            if (frameCounter < numFrames) {
                try {
                    frame = grabber.grabFrame();
                    if (frame.image == null) {
                        Log.d(TAG, "EMPTY IMAGE: " + frameCounter);
                        return mRgba;
                    }
                    frameCounter++;
                } catch (FrameGrabber.Exception e) {
                    exception = e;
                }
            }
        }

        if (frameCounter == numFrames - 1) {
            processing = false;
            mHandler.post(updateText);
        }

        Mat fMat = toMatConverter.convertToMat(frame);
        fMat.copyTo(selectMat);
        process(fMat).copyTo(mRgba);
        fMat.release();

        return mRgba;
    }

    private Mat process(Mat mRgba) {
        for (int k = 0; k < 2; k++) {
            if (drawSelectionBox[k] || processing) {
                mDetectorList.get(k).threshold(mRgba);
                Mat threshed = mDetectorList.get(k).getThreshold();

                MatVector contours = new MatVector();
                findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

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
                    //Log.i(TAG, box);
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
                    drawSelectionBox[k] = false;
                }
            }
        }

        return mRgba;
    }

    private void trackMotion(MatVector contours, int centroidY, int objectNum) {
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

    @Override
    public void onClick(View v) {
        if (!processing && mIsColorSelected[0] && mIsColorSelected[1]) {
            processing = true;
            mHandler.post(updateText);
            startButton.setVisibility(View.GONE);
        }
    }

    public boolean onTouch(View v, MotionEvent event) {
        if (!processing) {
            int xTouch = (int) event.getX();
            int yTouch = (int) event.getY();
            getPixelColor(selectMat, xTouch, yTouch);
        }

        return false; // don't need subsequent touch events
    }

    public void getPixelColor(Mat mTouched, int xTouch, int yTouch) {
        int cols = mTouched.cols();
        int rows = mTouched.rows();

        int xOffset = (cameraView.getPreviewWidth() - cols) / 2;
        int yOffset = (cameraView.getPreviewHeight() - rows) / 2;

        //TODO fix incorrect calculation of x and y - need to scale xTouch and yTouch so x and y are within the matrix dimens
        int x = xTouch - xOffset;
        int y = yTouch - yOffset;

        Log.i(TAG, "Touched image coordinates: (" + x + ", " + y + ")");

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) {
            //screenTouched = false;
            return;
        }

        opencv_core.Rect touchedRect = new opencv_core.Rect();

        int tmpx = (x>5) ? x-5 : 0;
        int tmpy = (y>5) ? y-5 : 0;
        touchedRect.x(tmpx);// = (x>5) ? x-5 : 0;
        touchedRect.y(tmpy);// = (y>5) ? y-5 : 0;

        int tmpw = (x+5 < cols) ? x + 5 - touchedRect.x() : cols - touchedRect.x();
        int tmph = (y+5 < rows) ? y + 5 - touchedRect.y() : rows - touchedRect.y();
        touchedRect.width(tmpw);
        touchedRect.height(tmph);

        // Calculate average hsv color of touched region
        Mat touchedRegionRgba = new Mat(mTouched, touchedRect);
        Mat touchedRegionHsv = new Mat();
        cvtColor(touchedRegionRgba, touchedRegionHsv, COLOR_RGB2HSV_FULL);
        Scalar mBlobColorHsv = sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width()*touchedRect.height();
        for (int i = 0; i < 3; i++) {
            mBlobColorHsv.put(i, mBlobColorHsv.get(i) / pointCount);
        }

        mDetectorList.get(selector).setHsvColor(mBlobColorHsv);
        mIsColorSelected[selector] = true;
        drawSelectionBox[selector] = true;
        selector ^= 1;

        touchedRegionRgba.release();
        touchedRegionHsv.release();
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
