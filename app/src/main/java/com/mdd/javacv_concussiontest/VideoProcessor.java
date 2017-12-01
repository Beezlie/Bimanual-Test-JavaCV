package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.app.ProgressDialog;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;

import com.mdd.javacv_concussiontest.utils.ColorBlobDetector;

import static android.content.ContentValues.TAG;
import static java.lang.Math.abs;
import static org.bytedeco.javacpp.opencv_core.CV_8UC4;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.inRange;
import static org.bytedeco.javacpp.opencv_core.multiply;
import static org.bytedeco.javacpp.opencv_core.Moments;
import static org.bytedeco.javacpp.opencv_imgproc.moments;
import static org.bytedeco.javacpp.opencv_highgui.*;

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

public class VideoProcessor extends AsyncTask<Void, Void, Void> {
    private Activity activity;
    private ProgressDialog asyncDialog;
    private final String filename;
    private final double pxscale;
    private FFmpegFrameGrabber grabber;
    private Exception exception;
    private OpenCVFrameConverter.ToMat toMatConverter;
    private Frame frame;
    private Scalar CONTOUR_COLOR_WHITE = new Scalar(255,255,255,255);
    private int[] numCycles = new int[2];
    private int[] dirChange = new int[2];
    private int[] minPeak = {10000, 10000};
    private int[] maxPeak = {0, 0};
    private int[] centroidPoints = new int[2];
    private boolean finishedTest = false;
    private String[] dirY = new String[2];
    private String[] dirYprev = new String[2];
    private boolean[] movingY = new boolean[2];
    private ArrayDeque<int[]> movingWindow = new ArrayDeque<>();
    private List<Integer> amplitudesL = new ArrayList<>();
    private List<Integer> amplitudesR = new ArrayList<>();
    private List<ColorBlobDetector> mDetectorList = new ArrayList<>();
    private File root;
    private File datafile;
    private FileWriter writer;

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
        toMatConverter = new OpenCVFrameConverter.ToMat();

        root = new File(Environment.getExternalStorageDirectory().toString());
        datafile = new File(root, "data.txt");
        try {
            writer = new FileWriter(datafile);
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

                Mat mRgba = toMatConverter.convert(frame);
                MatVector contours[] = new MatVector[2];

                for (int k = 0; k < 2; k++) {
                    contours[k] = mDetectorList.get(k).getContours();

                    try {
                        mDetectorList.get(k).process(mRgba, pxscale);
                    } catch (InterruptedException e) {
                        exception = e;
                    }

                    Log.d(TAG, "Contours count: " + contours[k].size());
                    if (contours[k].size() <= 0) {
                        break;
                    }

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
                            String result = "Right hand amplitude: " + (amp / pxscale) + " mm" + " at cycle count " + numCycles[0];
                            Log.i(TAG, result);
                            try {
                                writer.append(result);
                            } catch (IOException e) {
                                exception = e;
                            }
                        }
                        for (int amp : amplitudesL) {
                            String result = "Left hand amplitude: " + (amp / pxscale) + " mm" + " at cycle count " + numCycles[0];
                            Log.i(TAG, result);
                            try {
                                writer.append(result);
                            } catch (IOException e) {
                                exception = e;
                            }
                        }
                        finishedTest = true;
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

        try {
            writer.append("Test Completed");
            writer.flush();
            writer.close();
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
}

