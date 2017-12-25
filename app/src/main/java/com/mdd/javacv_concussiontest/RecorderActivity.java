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
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.CvScalar;
import static org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvGet2D;
import static org.bytedeco.javacpp.opencv_core.sumElems;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class RecorderActivity extends Activity implements OnClickListener, View.OnTouchListener, CvCameraPreview.CvCameraViewListener {

    private final static String CLASS_LABEL = "RecorderActivity";
    private final static String LOG_TAG = CLASS_LABEL;
    private PowerManager.WakeLock wakeLock;
    private boolean recording;
    private CvCameraPreview cameraView;
    private Button btnRecorderControl;
    private File savePath = new File(Environment.getExternalStorageDirectory(), "bimanualtest.mp4");
    private FFmpegFrameRecorder recorder;
    private long startTime = 0;
    private OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
    private final Object semaphore = new Object();

    //Color Detection global variables
    private Scalar mBlobColorHsv = new Scalar(255);
    private CvScalar mBlobColorRgba = new CvScalar(255);
    private boolean[] mIsColorSelected = new boolean[2];
    private boolean screenTouched = false;
    private ColorBlobDetector mDetectorL = new ColorBlobDetector();
    private ColorBlobDetector mDetectorR = new ColorBlobDetector();
    private List<ColorBlobDetector> mDetectorList = new ArrayList<>();
    private int selector = 0;
    private double pxscale = 1;
    private int xTouch, yTouch;
    private String hexcolor;
    private Button leftButton;
    private Button rightButton;
    private final Handler mHandler = new Handler();

    final Runnable mUpdateLeftButton = new Runnable() {
        public void run() {
            updateLeftButton();
        }
    };
    final Runnable mUpdateRightButton = new Runnable() {
        public void run() {
            updateRightButton();
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_recorder);

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

        if (recorder != null) {
            try {
                recorder.release();
            } catch (FrameRecorder.Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {

        if (keyCode == KeyEvent.KEYCODE_BACK) {
            if (recording) {
                stopRecording();
            }

            finish();

            return true;
        }

        return super.onKeyDown(keyCode, event);
    }

    private void initLayout() {
        btnRecorderControl = (Button) findViewById(R.id.recorder_control);
        btnRecorderControl.setText("Start");
        btnRecorderControl.setOnClickListener(this);
        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        leftButton = (Button) findViewById(R.id.left_color);
        rightButton = (Button) findViewById(R.id.right_color);

        cameraView.setCvCameraViewListener(this);
        cameraView.setOnTouchListener(this);
    }

    private void initRecorder(int width, int height) {
        int degree = getRotationDegree();
        Camera.CameraInfo info = new Camera.CameraInfo();
        Camera.getCameraInfo(cameraView.getCameraId(), info);
        boolean isFrontFaceCamera = info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT;
        Log.i(LOG_TAG, "init recorder with width = " + width + " and height = " + height + " and degree = "
                + degree + " and isFrontFaceCamera = " + isFrontFaceCamera);
        int frameWidth, frameHeight;
        /*
         0 = 90CounterCLockwise and Vertical Flip (default)
         1 = 90Clockwise
         2 = 90CounterClockwise
         3 = 90Clockwise and Vertical Flip
         */
        switch (degree) {
            case 0:
                frameWidth = width;
                frameHeight = height;
                break;
            case 90:
                frameWidth = height;
                frameHeight = width;
                break;
            case 180:
                frameWidth = width;
                frameHeight = height;
                break;
            case 270:
                frameWidth = height;
                frameHeight = width;
                break;
            default:
                frameWidth = width;
                frameHeight = height;
        }

        Log.i(LOG_TAG, "saved file path: " + savePath.getAbsolutePath());
        recorder = new FFmpegFrameRecorder(savePath, frameWidth, frameHeight, 0);
        recorder.setFormat("mp4");
        recorder.setVideoCodec(AV_CODEC_ID_MPEG4);
        recorder.setVideoQuality(1);
        // Set in the surface changed method
        recorder.setFrameRate(30);

        Log.i(LOG_TAG, "recorder initialize success");
    }

    private int getRotationDegree() {
        int result;

        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        if (Build.VERSION.SDK_INT >= 9) {
            // on >= API 9 we can proceed with the CameraInfo method
            // and also we have to keep in mind that the camera could be the front one
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(cameraView.getCameraId(), info);

            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                result = (info.orientation + degrees) % 360;
                result = (360 - result) % 360;  // compensate the mirror
            } else {
                // back-facing
                result = (info.orientation - degrees + 360) % 360;
            }
        } else {
            // TODO: on the majority of API 8 devices, this trick works good
            // and doesn't produce an upside-down preview.
            // ... but there is a small amount of devices that don't like it!
            result = Math.abs(degrees - 90);
        }
        return result;
    }

    public void startRecording() {
        try {
            synchronized (semaphore) {
                recorder.start();
            }
            startTime = System.currentTimeMillis();
            recording = true;
        } catch (FFmpegFrameRecorder.Exception e) {
            e.printStackTrace();
        }
    }

    public void stopRecording() {
        if (recorder != null && recording) {
            recording = false;
            Log.v(LOG_TAG, "Finishing recording, calling stop and release on recorder");
            try {
                synchronized (semaphore) {
                    recorder.stop();
                    recorder.release();
                }
            } catch (FFmpegFrameRecorder.Exception e) {
                e.printStackTrace();
            }
            recorder = null;
        }
    }

    @Override
    public void onClick(View v) {
        if (mIsColorSelected[0] && mIsColorSelected[1]) {
            if (!recording) {
                saveThreshColors();
                startRecording();
                recording = true;
                Log.w(LOG_TAG, "Start Button Pushed");
                btnRecorderControl.setText("Stop");
                btnRecorderControl.setBackgroundResource(R.drawable.bg_red_circle_button);
            } else {
                // This will trigger the audio recording loop to stop and then set isRecorderStart = false;
                stopRecording();
                recording = false;
                Log.w(LOG_TAG, "Stop Button Pushed");
                btnRecorderControl.setVisibility(View.GONE);
                Toast.makeText(this, "Video file was saved to \"" + savePath + "\"", Toast.LENGTH_LONG).show();
                //saveParams();
            }
        } else {
            Toast.makeText(this, "Must select objects first", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mDetectorList.add(mDetectorL);
        mDetectorList.add(mDetectorR);
        initRecorder(width, height);
    }

    @Override
    public void onCameraViewStopped() {
        stopRecording();
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

        if (recording && mat != null) {
            //TODO - figure out if this synchronization could be messing with another one and causing deadlock
            //the main problem is when the record button is pressed it seems to get stuck waiting
            //maybe creating the color picking threads above messes up the timing/synchronization?
            synchronized (semaphore) {
                try {
                    Frame frame = converterToMat.convert(mat);
                    long t = 1000 * (System.currentTimeMillis() - startTime);
                    if (t > recorder.getTimestamp()) {
                        recorder.setTimestamp(t);
                    }
                    recorder.record(frame);
                } catch (FFmpegFrameRecorder.Exception e) {
                    Log.v(LOG_TAG, e.getMessage());
                    e.printStackTrace();
                }
            }
        }
        return mat;
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

        Log.i(LOG_TAG, "Touched image coordinates: (" + x + ", " + y + ")");

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

        //get Rgba values
        IplImage src = new IplImage(mTouched);
        mBlobColorRgba = cvGet2D(src, y, x);

        int red = (int)mBlobColorRgba.get(0);
        int green = (int)mBlobColorRgba.get(1);
        int blue = (int)mBlobColorRgba.get(2);
        int alpha = (int)mBlobColorRgba.get(3);
        hexcolor = String.format("#%02x%02x%02x", red, green, blue);
        Log.i(LOG_TAG, "Touched rgba color: (" + hexcolor + ")");
        Log.i(LOG_TAG, "red: (" + red + ")");
        Log.i(LOG_TAG, "green: (" + green + ")");
        Log.i(LOG_TAG, "blue: (" + blue + ")");
        Log.i(LOG_TAG, "alpha: (" + alpha + ")");

        //set the colour of one of the buttons to the rgba colour selected
        if (selector == 0) {
            mHandler.post(mUpdateLeftButton);
            //leftButton.setBackgroundColor(Color.parseColor(hexcolor));
        } else {
            mHandler.post(mUpdateRightButton);
            //rightButton.setBackgroundColor(Color.parseColor(hexcolor));
        }

        mDetectorList.get(selector).setHsvColor(mBlobColorHsv);
        mIsColorSelected[selector] = true;
        selector ^= 1;

        src.release();
        touchedRegionRgba.release();
        touchedRegionHsv.release();
    }

    public void saveThreshColors() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();

        Scalar leftLowerBound = mDetectorList.get(0).getLowerBound();
        Scalar rightLowerBound = mDetectorList.get(1).getLowerBound();
        editor.putInt(getString(R.string.leftHueLower), (int)leftLowerBound.get(0));
        editor.putInt(getString(R.string.leftSatLower), (int)leftLowerBound.get(1));
        editor.putInt(getString(R.string.leftBriLower), (int)leftLowerBound.get(2));
        editor.putInt(getString(R.string.rightHueLower), (int)rightLowerBound.get(0));
        editor.putInt(getString(R.string.rightSatLower), (int)rightLowerBound.get(1));
        editor.putInt(getString(R.string.rightBriLower), (int)rightLowerBound.get(2));

        Scalar leftUpperBound = mDetectorList.get(0).getUpperBound();
        Scalar rightUpperBound = mDetectorList.get(1).getUpperBound();
        editor.putInt(getString(R.string.leftHueUpper), (int)leftUpperBound.get(0));
        editor.putInt(getString(R.string.leftSatUpper), (int)leftUpperBound.get(1));
        editor.putInt(getString(R.string.leftBriUpper), (int)leftUpperBound.get(2));
        editor.putInt(getString(R.string.rightHueUpper), (int)rightUpperBound.get(0));
        editor.putInt(getString(R.string.rightSatUpper), (int)rightUpperBound.get(1));
        editor.putInt(getString(R.string.rightBriUpper), (int)rightUpperBound.get(2));

        editor.commit();
    }

    public void updateLeftButton() {
        leftButton.setBackgroundColor(Color.parseColor(hexcolor));
    }

    public void updateRightButton() {
        rightButton.setBackgroundColor(Color.parseColor(hexcolor));
    }
}

