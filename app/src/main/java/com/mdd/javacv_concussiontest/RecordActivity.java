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

/*
TODO FOR PROJECT
1) fix the fact that touch doesn't work for the whole screen (there is a mismatch between matrix size and screen size) - may need to live with this though
2) debug the actual processing
3) implement the pixel scaling using some type of calibration
4) store pixel scaling in shared preferences
5) simplify app by keeping only RecordActivity and the calibration activity
6) fix view.WindowLeaked error
 */

public class RecordActivity extends Activity implements OnClickListener, View.OnTouchListener, CvCameraPreview.CvCameraViewListener {

    private final static String CLASS_LABEL = "RecordActivity";
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
    private int xTouch, yTouch;
    private String hexcolor;
    private Button leftButton;
    private Button rightButton;
    private Button processButton;
    private final Handler mHandler = new Handler();
    private VideoProcessor vidProcessor;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_record);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        leftButton = (Button) findViewById(R.id.left_color);
        rightButton = (Button) findViewById(R.id.right_color);
        processButton = (Button) findViewById(R.id.process);
        processButton.setVisibility(View.GONE);

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

        vidProcessor.dismissDialog();

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

            //TODO - maybe remove this line so the app doesnt stop because of:
            //"Cancelling event due to no window focus: KeyEvent"
            //finish();

            return true;
        }

        return super.onKeyDown(keyCode, event);
    }

    private void initLayout() {
        btnRecorderControl = (Button) findViewById(R.id.recorder_control);
        btnRecorderControl.setText("Start");
        btnRecorderControl.setOnClickListener(this);

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
                processButton.setVisibility(View.VISIBLE);
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
            GetPixelColor color = new GetPixelColor(mRgba, xTouch, yTouch);
            mHandler.post(color);
            mRgba.release();
        }

        if (recording && mat != null) {
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

    private class GetPixelColor implements Runnable {
        private Mat mat;
        private int xPoint;
        private int yPoint;
        public GetPixelColor(Mat _mat, int _xTouch, int _yTouch) {
            this.mat = _mat.clone();
            this.xPoint = _xTouch;
            this.yPoint = _yTouch;
        }

        @Override
        public void run() {
            int cols = mat.cols();
            int rows = mat.rows();

            //test lines - remove after
            int w = cameraView.getWidth();
            int h = cameraView.getHeight();
            int a = cameraView.getPreviewWidth();
            int b = cameraView.getPreviewHeight();

            int xOffset = (cameraView.getPreviewWidth() - cols) / 2;
            int yOffset = (cameraView.getPreviewHeight() - rows) / 2;

            //TODO fix incorrect calculation of x and y - need to scale xTouch and yTouch so x and y are within the matrix dimens
            int x = xPoint - xOffset;
            int y = yPoint - yOffset;

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
            Mat touchedRegionRgba = new Mat(mat, touchedRect);
            Mat touchedRegionHsv = new Mat();
            cvtColor(touchedRegionRgba, touchedRegionHsv, COLOR_RGB2HSV_FULL);
            mBlobColorHsv = sumElems(touchedRegionHsv);
            int pointCount = touchedRect.width()*touchedRect.height();
            for (int i = 0; i < 3; i++) {
                mBlobColorHsv.put(i, mBlobColorHsv.get(i) / pointCount);
            }

            //get Rgba values
            IplImage src = new IplImage(mat);
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
            synchronized (semaphore) {
                if (selector == 0) {
                    //mHandler.post(mUpdateLeftButton);
                    leftButton.setBackgroundColor(Color.parseColor(hexcolor));
                } else {
                    //mHandler.post(mUpdateRightButton);
                    rightButton.setBackgroundColor(Color.parseColor(hexcolor));
                }
            }

            mDetectorList.get(selector).setHsvColor(mBlobColorHsv);
            mIsColorSelected[selector] = true;
            selector ^= 1;

            Log.i(LOG_TAG, "release temp mem");
            src.release();
            touchedRegionRgba.release();
            touchedRegionHsv.release();
            Log.i(LOG_TAG, "mem released");
        }
    }

    public void buttonProcess(View view) {
        Log.i(LOG_TAG, "Begin video processing");
        vidProcessor = new VideoProcessor(this, savePath.getAbsolutePath(), pxscale, mDetectorList);
        vidProcessor.execute();
    }
}
