package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.opencv_core;

import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.sumElems;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_TREE;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class CalibrationActivity extends Activity implements View.OnTouchListener, CvCameraPreview.CvCameraViewListener {
    private final static String TAG = "CalibrationActivity";
    //width and height of 8.5 x 11 inch paper - in mm
    private final static float calibWidth = 279;
    private final static float calibHeight = 216;

    private CvCameraPreview cameraView;
    private ColorBlobDetector mDetector;
    private int xTouch, yTouch;
    private boolean screenTouched = false;
    private boolean mIsColorSelected = false;
    private opencv_core.Scalar mBlobColorHsv = new opencv_core.Scalar(255);
    private opencv_core.Scalar CONTOUR_COLOR_GREEN = new opencv_core.Scalar(16,238,8,255);
    private Rect boundRect;
    private EditText distanceField;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_calibration);

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
        mDetector = new ColorBlobDetector();
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

        MatVector contours = new MatVector(); // MatVector is a JavaCV list of Mats
        findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        if (contours.size() <= 0) {
            return rgbaMat;
        }

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
        boundRect = boundingRect(contours.get(boundPos));
        rectangle(rgbaMat, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_GREEN, 2, 8, 0);

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

    //get the pixel scale in pixels per mm
    public float getPixelScale(Rect boundRect) {
        float hscale = boundRect.height() / calibHeight;
        float wscale = boundRect.width() / calibWidth;

        //TODO - figure out if i should return average of the 2 numbers or if there is a better way
        return (hscale + wscale) / 2;
    }

    public void calibrate(View button) {
        float pxscale = getPixelScale(boundRect);

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putFloat(getString(R.string.pxscale), pxscale);
        editor.commit();

        Context context = getApplicationContext();
        CharSequence text = "Calibration completed";
        int duration = Toast.LENGTH_SHORT;

        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }
}

