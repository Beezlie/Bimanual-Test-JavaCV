package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.hardware.Camera;
import android.os.AsyncTask;
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

import com.google.gson.Gson;
import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.UByteArrayIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RotatedRect;
import org.bytedeco.javacpp.opencv_core.Moments;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.avcodec.AV_CODEC_ID_MPEG4;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_core.sumElems;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_HSV2RGB_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.moments;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

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
    private Mat mRgba;
    private Mat mSpectrum = new Mat();
    private opencv_core.Size SPECTRUM_SIZE = new opencv_core.Size(200, 64);
    private Scalar CONTOUR_COLOR_WHITE = new Scalar(255,255,255,255);
    private Scalar mBlobColorHsv = new Scalar(255);
    private Scalar mBlobColorRgba = new Scalar(255);
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
    final Handler mHandler = new Handler();

    final Runnable mUpdateLeftButton = new Runnable() {
        public void run() {
            updateLeftButtonColor();
        }
    };

    final Runnable mUpdateRightButton = new Runnable() {
        public void run() {
            updateRightButtonColor();
        }
    };

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
            Toast.makeText(this, "Video file was saved to \"" + savePath + "\"", Toast.LENGTH_LONG).show();
            //saveParams();

        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CV_8UC4);
        mDetectorList.add(mDetectorL);
        mDetectorList.add(mDetectorR);
        initRecorder(width, height);
    }

    @Override
    public void onCameraViewStopped() {
        stopRecording();
    }

    public boolean onTouch(View v, MotionEvent event) {
        xTouch = (int)event.getX();
        yTouch = (int)event.getY();
        screenTouched = true;

        return false; // don't need subsequent touch events
    }

    @Override
    public Mat onCameraFrame(Mat mat) {
        if (screenTouched) {
            mat.copyTo(mRgba);
            getPixelColor(xTouch, yTouch);
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

    private void getPixelColor(int xTouch, int yTouch) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        int w = cameraView.getWidth();
        int h = cameraView.getHeight();
        int a = cameraView.getPreviewWidth();
        int b = cameraView.getPreviewHeight();
        int xOffset = (cameraView.getPreviewWidth() - cols) / 2;
        int yOffset = (cameraView.getPreviewHeight() - rows) / 2;

        //TODO fix incorrect calculation of x and y - need to scale xTouch and yTouch so x and y are within the matrix dimens
        int x = xTouch- xOffset;
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

        Mat touchedRegionRgba = new Mat(mRgba, touchedRect);
        Mat touchedRegionHsv = new Mat();
        cvtColor(touchedRegionRgba, touchedRegionHsv, COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        mBlobColorHsv = sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width()*touchedRect.height();
        for (int i = 0; i < 3; i++) {
            mBlobColorHsv.put(i, mBlobColorHsv.get(i) / pointCount);
        }

        //convert the Hsv colour blob to rgba
        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);

        int red = (int)mBlobColorRgba.get(0);
        int green = (int)mBlobColorRgba.get(1);
        int blue = (int)mBlobColorRgba.get(2);
        int alpha = (int)mBlobColorRgba.get(3);
        //String hexcolor = String.format("#%02x%02x%02x%02x", alpha, red, green, blue);
        hexcolor = String.format("#%02x%02x%02x", red, green, blue);
        Log.i(LOG_TAG, "Touched rgba color: (" + hexcolor + ")");
        Log.i(LOG_TAG, "red: (" + red + ")");
        Log.i(LOG_TAG, "green: (" + green + ")");
        Log.i(LOG_TAG, "blue: (" + blue + ")");
        Log.i(LOG_TAG, "alpha: (" + alpha + ")");

        mDetectorList.get(selector).setHsvColor(mBlobColorHsv);

        resize(mDetectorList.get(selector).getSpectrum(), mSpectrum, SPECTRUM_SIZE);

        mIsColorSelected[selector] = true;

        //set the colour of one of the buttons to the rgba colour selected
        if (selector == 0) {
            mHandler.post(mUpdateLeftButton);
            //leftButton.setBackgroundColor(Color.parseColor(hexcolor));
        } else {
            mHandler.post(mUpdateRightButton);
        }

        selector ^= 1;
        screenTouched = false;

        touchedRegionRgba.release();
        touchedRegionHsv.release();
    }

    private Scalar convertScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat(1, 1, CV_8UC3);
        //TODO Fix the fact that color doesnt set properly
        DoublePointer dp = new DoublePointer(hsvColor.get(0), hsvColor.get(1), hsvColor.get(2), hsvColor.get(3));
        Mat pointMatHsv = new Mat(1, 1, CV_8UC3, dp);

        UByteRawIndexer idxRgba = pointMatRgba.createIndexer();

        cvtColor(pointMatHsv, pointMatRgba, COLOR_HSV2RGB_FULL, 4);

        double r = idxRgba.get(0,0);
        double g = idxRgba.get(0,1);
        double b = idxRgba.get(0,2);
        Scalar result = new Scalar(r, g, b, 0);

        return result;
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

    public void processVideo() throws FrameGrabber.Exception, InterruptedException {
        String LOG_TAG = "VideoProcessing";

        //elements required for image processing
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
        int[] centroidPoints = new int[2];
        String[] dirYprev = new String[2];

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(savePath.getAbsolutePath());
        OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

        // Open video file
        grabber.start();

        long delay = Math.round(1000d / grabber.getFrameRate());

        // Read frame by frame
        Frame frame;
        while ((frame = grabber.grab()) != null) {
            Mat mRgba = toMatConverter.convert(frame);
            MatVector contours[] = new MatVector[2];

            for (int k = 0; k < 2; k++) {
                contours[k] = mDetectorList.get(k).getContours();

                mDetectorList.get(k).process(mRgba, pxscale);

                Log.d(TAG, "Contours count: " + contours[k].size());

                //get bounding rectangle
                RotatedRect rect = minAreaRect(contours[k].get(0));

                double boundWidth = rect.size().width();
                double boundHeight = rect.size().height();
                int boundPos = 0;

                //update the width and height for the bounding rectangle based on the area of each rectangle calculated from the contour list
                for (int i = 1; i < contours[k].size(); i++) {
                    rect = minAreaRect(contours[k].get(i));
                    if (rect.size().width() * rect.size().height() > boundWidth * boundHeight) {
                        boundWidth = rect.size().width();
                        boundHeight = rect.size().height();
                        //store the location in the contour list of the maximum area bounding rectangle
                        boundPos = i;
                    }
                }

                //create a new bounding rectangle from the largest contour area
                Rect boundRect = boundingRect(contours[k].get(boundPos));
                rectangle(mRgba, boundRect.tl(), boundRect.br(), CONTOUR_COLOR_WHITE, 2, 8, 0);

                //get centroids of the bounding contour
                Moments mc = moments(contours[k].get(boundPos), false);
                int centroidx = (int) (mc.m10() / mc.m00());
                int centroidy = (int) (mc.m01() / mc.m00());
                centroidPoints[k] = centroidy;
            }

            if (contours[0].size() > 0 && contours[1].size() > 0) {
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
            }
            imshow("bimanual_test", mRgba);

            // Delay
            Thread.sleep(delay);
        }

        // Close the video file
        grabber.release();
    }

    public void buttonProcess(View view) {
        //TODO - check that saved file exists
        //code here

        try {
            processVideo();
        } catch (InterruptedException e) {
            //TODO - handle exception
        } catch (FrameGrabber.Exception e) {
            //TODO - handle exception
        }
    }

    public void updateLeftButtonColor() {
        leftButton.setBackgroundColor(Color.parseColor(hexcolor));
    }

    public void updateRightButtonColor() {
        //rightButton.setBackgroundColor(Color.argb(alpha, red, green, blue));
        rightButton.setBackgroundColor(Color.parseColor(hexcolor));
    }
}
