package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.app.ProgressDialog;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;

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

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.CvRect;
import static org.bytedeco.javacpp.opencv_core.CvSeq;
import static org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class VideoProcessor extends AsyncTask<Void, Void, Void> {
    private Activity activity;
    private ProgressDialog asyncDialog;
    private final String filename;
    private final double pxscale;
    private FFmpegFrameGrabber grabber;
    private Exception exception;
    private OpenCVFrameConverter.ToIplImage toIplConverter;
    private OpenCVFrameConverter.ToMat toMatConverter;
    private Frame frame;
    private int[] numCycles = new int[2];
    private int[] dirChange = new int[2];
    private int[] minPeak = {10000, 10000};
    private int[] maxPeak = {0, 0};
    private int[] centroidPoints = new int[2];
    private int[] boundRectTL = new int[2];
    private int[] boundRectBR = new int[2];
    private boolean finishedTest = false;
    private String[] dirY = new String[2];
    private String[] dirYprev = new String[2];
    private boolean[] movingY = new boolean[2];
    private ArrayDeque<int[]> movingWindow = new ArrayDeque<>();
    private List<Integer> amplitudesL = new ArrayList<>();
    private List<Integer> amplitudesR = new ArrayList<>();
    private List<ColorBlobDetector> mDetectorList = new ArrayList<>();
    private File root;
    private File amplitudeData;
    private File boundingData;
    private FileWriter amplitudeWriter;
    private FileWriter boundingWriter;

    public VideoProcessor(Activity activity, String filename, double pxscale, List<ColorBlobDetector> mDetectorList) {
        this.activity = activity;
        this.filename = filename;
        this.pxscale = pxscale;
        this.mDetectorList = mDetectorList;
    }

    @Override
    protected void onPreExecute() {
        asyncDialog = new ProgressDialog(activity);
        //set message of the dialog
        asyncDialog.setMessage("Loading");
        //show dialog
        asyncDialog.show();

        grabber = new FFmpegFrameGrabber(filename);
        toIplConverter = new OpenCVFrameConverter.ToIplImage();
        toMatConverter = new OpenCVFrameConverter.ToMat();

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

        super.onPreExecute();
    }

    @Override
    protected Void doInBackground(Void... params) {
        // Open video file
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }

        long delay = Math.round(1000d / grabber.getFrameRate());

        // Read frame by frame
        for (int f = 0; f < grabber.getLengthInFrames(); f++) {
            try {
                frame = grabber.grabFrame();
                if (frame.image == null) {
                    Log.d(TAG, "EMPTY IMAGE: " + f);
                    continue;
                }

                // process the frame
                //IplImage img = toIplConverter.convert(frame);
                //CvSeq contours[] = new CvSeq[2];
                Mat mat = toMatConverter.convertToMat(frame);
                MatVector mContours[] = new MatVector[2];

                for (int k = 0; k < 2; k++) {
                    mDetectorList.get(k).threshold(mat);
                    Mat threshed = mDetectorList.get(k).getThreshold();

                    MatVector contours = new MatVector();
                    findContours(threshed, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

                    if (contours.size() <= 0) {
                        break;
                    }
                    mContours[k] = contours;

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
                    Rect boundRect = boundingRect(contours.get(boundPos));
                    String box = "Bounding box " + k + ": width = " + boundRect.width() + ", height = " + boundRect.height() +
                            ", bottom right = " + boundRect.br() + ", top left = " + boundRect.tl();
                    try {
                        boundingWriter.append(box);
                    } catch (IOException e) {
                        exception = e;
                    }

                    CvMoments moments = new CvMoments();
                    cvMoments(new IplImage(contours.get(boundPos)), moments, 1);
                    double m00 = cvGetSpatialMoment(moments, 0, 0);
                    double m10 = cvGetSpatialMoment(moments, 1, 0);
                    double m01 = cvGetSpatialMoment(moments, 0, 1);
                    int centroidX = 0;
                    int centroidY = 0;
                    if (m00 != 0) {   // calculate center
                        centroidX = (int) Math.round(m10 / m00);
                        centroidY = (int) Math.round(m01 / m00);
                    }
                    centroidPoints[k] = centroidY;
                }

                if (mContours[0].size() > 0 && mContours[1].size() > 0) {
                    movingWindow.addFirst(centroidPoints);
                    if (movingWindow.size() > 5) {
                        movingWindow.removeLast();     //limit size of deque to 5
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
                }

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

        // Close the video file
        try {
            grabber.release();
        } catch (FrameGrabber.Exception e) {
            exception = e;
        }

        //display results once test completed
        for (int amp : amplitudesR) {
            String result = "Right hand amplitude: " + (amp / pxscale) + " mm";
            Log.i(TAG, result);
            try {
                amplitudeWriter.append(result);
            } catch (IOException e) {
                exception = e;
            }
        }
        for (int amp : amplitudesL) {
            String result = "Left hand amplitude: " + (amp / pxscale) + " mm";
            Log.i(TAG, result);
            try {
                amplitudeWriter.append(result);
            } catch (IOException e) {
                exception = e;
            }
        }

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

        return null;
    }

    @Override
    protected void onPostExecute(Void result) {
        //hide the dialog
        asyncDialog.dismiss();

        super.onPostExecute(result);
    }

    protected void dismissDialog() {
        asyncDialog.dismiss();
    }
}

