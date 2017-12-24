package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.Mat;
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

public class RealTimeActivity extends Activity implements View.OnTouchListener, CvCameraPreview.CvCameraViewListener {
    final String TAG = "OpenCvActivity";

    private CvCameraPreview cameraView;
    private ColorBlobDetector mDetector = new ColorBlobDetector();
    private int xTouch, yTouch;
    private int centroidX, centroidY;
    private boolean screenTouched = false;
    private boolean mIsColorSelected = false;
    private opencv_core.Scalar mBlobColorHsv = new opencv_core.Scalar(255);
    private opencv_core.Scalar CONTOUR_COLOR_WHITE = new opencv_core.Scalar(255,255,255,255);

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_real_time);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        initLayout();
    }

    private void initLayout() {
        cameraView.setCvCameraViewListener(this);
        cameraView.setOnTouchListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        if (screenTouched) {
            screenTouched = false;
            Mat mRgba = new Mat(rgbaMat.rows(), rgbaMat.cols(), CV_8UC4);
            rgbaMat.copyTo(mRgba);
            getPixelColor(mRgba, xTouch, yTouch);
            mRgba.release();
        }

        if (!mIsColorSelected)
            return rgbaMat;

        mDetector.threshold(rgbaMat);

        Mat threshed = mDetector.getThreshold();

        opencv_core.MatVector contours = new opencv_core.MatVector(); // MatVector is a JavaCV list of Mats
        findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        if (contours.size() <= 0) {
            return rgbaMat;
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
        rectangle(rgbaMat, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);

        opencv_imgproc.CvMoments moments = new opencv_imgproc.CvMoments();
        cvMoments(new opencv_core.IplImage(contours.get(boundPos)), moments, 1);
        double m00 = cvGetSpatialMoment(moments, 0, 0);
        double m10 = cvGetSpatialMoment(moments, 1, 0);
        double m01 = cvGetSpatialMoment(moments, 0, 1);
        if (m00 != 0) {   // calculate center
            centroidX = (int) Math.round(m10 / m00);
            centroidY = (int) Math.round(m01 / m00);
        }
        circle(rgbaMat, new opencv_core.Point(centroidX, centroidY), 4, new opencv_core.Scalar(255,255,255,255));

        return rgbaMat;
    }

    public boolean onTouch(View v, MotionEvent event) {
        xTouch = (int)event.getX();
        yTouch = (int)event.getY();
        screenTouched = true;

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
            screenTouched = false;
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
        mBlobColorHsv = sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width()*touchedRect.height();
        for (int i = 0; i < 3; i++) {
            mBlobColorHsv.put(i, mBlobColorHsv.get(i) / pointCount);
        }

        mDetector.setHsvColor(mBlobColorHsv);
        mIsColorSelected = true;

        touchedRegionRgba.release();
        touchedRegionHsv.release();
    }
}
