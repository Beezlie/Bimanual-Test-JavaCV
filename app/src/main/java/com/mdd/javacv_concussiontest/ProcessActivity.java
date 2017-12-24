package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.PowerManager;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.avcodec.AV_CODEC_ID_MPEG4;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.CvScalar;
import static org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvGet2D;
import static org.bytedeco.javacpp.opencv_core.sumElems;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/*
TODO FOR PROJECT
1) fix the fact that touch doesn't work for the whole screen (there is a mismatch between matrix size and screen size) - may need to live with this though
2) debug the actual processing
3) implement the pixel scaling using some type of calibration
4) store pixel scaling in shared preferences
5) simplify app by keeping only RecordActivity and the calibration activity
6) fix view.WindowLeaked error
 */

public class ProcessActivity extends Activity implements View.OnTouchListener, CvCameraPreview.CvCameraViewListener {

    private final static String CLASS_LABEL = "ProcessActivity";
    private final static String LOG_TAG = CLASS_LABEL;
    private PowerManager.WakeLock wakeLock;
    private CvCameraPreview cameraView;
    private OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
    private final Object semaphore = new Object();
    private final Object colorSem = new Object();

    //Color Detection global variables
    private opencv_core.Size SPECTRUM_SIZE = new opencv_core.Size(200, 64);
    private Scalar CONTOUR_COLOR_WHITE = new Scalar(255,255,255,255);
    private Scalar mBlobColorHsv = new Scalar(255);
    private CvScalar mBlobColorRgba = new CvScalar(255);
    private boolean[] mIsColorSelected = new boolean[2];
    private boolean screenTouched = false;
    private ColorBlobDetector mDetectorL = new ColorBlobDetector();
    private ColorBlobDetector mDetectorR = new ColorBlobDetector();
    private List<ColorBlobDetector> mDetectorList = new ArrayList<>();
    private int selector = 0;
    private double pxscale = 1;
    private final Handler mHandler = new Handler();
    private Exception exception;
    private int xTouch, yTouch;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_process);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);

        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.SCREEN_BRIGHT_WAKE_LOCK, CLASS_LABEL);
        wakeLock.acquire();

        initLayout();
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (wakeLock == null) {
            PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
            wakeLock = pm.newWakeLock(PowerManager.SCREEN_BRIGHT_WAKE_LOCK, CLASS_LABEL);
            wakeLock.acquire();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (wakeLock != null) {
            wakeLock.release();
            wakeLock = null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (wakeLock != null) {
            wakeLock.release();
            wakeLock = null;
        }
    }

    private void initLayout() {
        cameraView.setCvCameraViewListener(this);
        cameraView.setOnTouchListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mDetectorList.add(mDetectorL);
        mDetectorList.add(mDetectorR);
    }

    @Override
    public void onCameraViewStopped() {
        return;
    }

    @Override
    public Mat onCameraFrame(Mat mat) {
        if (screenTouched) {
            screenTouched = false;
            Mat mRgba = new Mat(mat.rows(), mat.cols(), CV_8UC4);
            mat.copyTo(mRgba);
            getPixelColor(mRgba, xTouch, yTouch);
            mRgba.release();
        }

        for (int k = 0; k < 2; k++) {
            if (!mIsColorSelected[k])
                return mat;

            //recalculate and filter the contours of ColourBlob k
            Mat mRgba = new Mat(mat.rows(), mat.cols(), CV_8UC3);
            mat.copyTo(mRgba);
            IplImage img = new IplImage(mRgba);
            mRgba.release();
            mDetectorList.get(k).process3(img);
            mDetectorList.get(k).processMat2(mat);

            //get the list of contour vertices from ColourBlob k (0 = left, 1 = right)
            List<Mat> contours = mDetectorList.get(k).getContours();

            Log.d(LOG_TAG, "Contours count: " + contours.size());

            //if no contours found return unmodified rgba matrix
            if (contours.size() <= 0) {
                return mat;
            }

            Mat test = contours.get(0);
            //OpenCV Error: Assertion failed (total >= 0 && (depth == CV_32F || depth == CV_32S)) in void cv::convexHull
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
            Rect boundRect = boundingRect(contours.get(boundPos));

            //rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType, int shift)
            //draws a bounding rectangle on the image (tl = top-left, br = bottom-right)
            rectangle(mat, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);
        }

        return mat;
    }

    public boolean onTouch(View v, MotionEvent event) {
        xTouch = (int)event.getX();
        yTouch = (int)event.getY();
        screenTouched = true;

        return false; // don't need subsequent touch events
    }

    public int pxToDistance() {
        //f_x = f * m_x
        //f_y = f * m_y
        //solve for mx and my

        //return focal_mm / sensor_width_mm;
        return 1;
    }

    public void saveParams() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();

        editor.putInt("pxscale", pxToDistance());

        //TODO - figure out how to get threshold values to save to shared prefs
        /*
        String s;
        Scalar bound;
        bound = mDetectorL.getLowerBound();
        double output[] = {bound.get(0), bound.get(1), bound.get(2), bound.get(3)};
        s = Arrays.toString(output);
        s = s.substring(1, s.length() - 1);
        String[] s_array = s.split(", ");
        editor.putStringSet("left_lower_bound", s_array);

        editor.putString("right_lower_bound", json);

        editor.putString("left_upper_bound", json);

        editor.putString("right_upper_bound", json);
        */
        editor.commit();
    }

    public void getPixelColor(Mat mTouched, int xTouch, int yTouch) {
        //TODO - figure out why mTouched is 0
        int cols = mTouched.cols();
        int rows = mTouched.rows();

        //test lines - remove after
        int w = cameraView.getWidth();
        int h = cameraView.getHeight();
        int a = cameraView.getPreviewWidth();
        int b = cameraView.getPreviewHeight();

        int xOffset = (cameraView.getPreviewWidth() - cols) / 2;
        int yOffset = (cameraView.getPreviewHeight() - rows) / 2;

        //TODO fix incorrect calculation of x and y - need to scale xTouch and yTouch so x and y are within the matrix dimens
        int x = xTouch - xOffset;
        int y = yTouch - yOffset;

        Log.i(LOG_TAG, "Touched image coordinates: (" + x + ", " + y + ")");

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) {
            screenTouched = false;
            return;
        }

        Rect touchedRect = new Rect();

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

        mDetectorList.get(selector).setHsvColor(mBlobColorHsv);
        mIsColorSelected[selector] = true;
        selector ^= 1;

        touchedRegionRgba.release();
        touchedRegionHsv.release();

    }
}
