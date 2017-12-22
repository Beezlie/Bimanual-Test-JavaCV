package com.mdd.javacv_concussiontest.utils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core.CvBox2D;
import org.bytedeco.javacpp.opencv_core.CvContour;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.CvSize2D32f;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FrameGrabber;

import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvInRangeS;
import static org.bytedeco.javacpp.opencv_core.cvScalar;
import static org.bytedeco.javacpp.opencv_core.cvSize;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2HSV;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_MEDIAN;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_LIST;
import static org.bytedeco.javacpp.opencv_imgproc.CvMoments;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvDilate;
import static org.bytedeco.javacpp.opencv_imgproc.cvFindContours;
import static org.bytedeco.javacpp.opencv_imgproc.cvGetSpatialMoment;
import static org.bytedeco.javacpp.opencv_imgproc.cvMinAreaRect2;
import static org.bytedeco.javacpp.opencv_imgproc.cvMoments;
import static org.bytedeco.javacpp.opencv_imgproc.cvSmooth;

/**
 * Created by matt on 11/10/17.
 */

public class ColorBlobDetector {
    String LOG_TAG = "ColorBlobDetection";
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);
    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;
    private static final float SMALLEST_AREA = 600.0f;
    // Color radius for range checking in HSV color space
    private Scalar mColorRadius = new Scalar(25,50,50,0);
    //private MatVector mContours = new MatVector();
    private CvMemStorage storage = CvMemStorage.create();
    private static final double IMG_SCALE = 2;
    private int hueLower, hueUpper, satLower, satUpper, briLower, briUpper;
    private CvSeq largestContour = new CvSeq();
    private int centroidY;
    private int centroidX;

    //TODO try to use the hue/sat/bri instead of the scalars i was using before
    //Also try to use something other than MAt for spectrum hsv
    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.get(0) >= mColorRadius.get(0)) ? hsvColor.get(0)-mColorRadius.get(0) : 0;
        double maxH = (hsvColor.get(0)+mColorRadius.get(0) <= 255) ? hsvColor.get(0)+mColorRadius.get(0) : 255;

        mLowerBound.put(0, minH);
        mUpperBound.put(0, maxH);
        hueLower = (int)minH;
        hueUpper = (int)maxH;

        mLowerBound.put(1, hsvColor.get(1) - mColorRadius.get(1));
        mUpperBound.put(1, hsvColor.get(1) + mColorRadius.get(1));
        satLower = (int)(hsvColor.get(1) - mColorRadius.get(1));
        satUpper = (int)(hsvColor.get(1) + mColorRadius.get(1));

        mLowerBound.put(2, hsvColor.get(2) - mColorRadius.get(2));
        mUpperBound.put(2, hsvColor.get(2) + mColorRadius.get(2));
        briLower = (int)(hsvColor.get(2) - mColorRadius.get(2));
        briUpper = (int)(hsvColor.get(2) + mColorRadius.get(2));

        mLowerBound.put(3, 0);
        mUpperBound.put(3, 255);
    }

    public void process(IplImage img) throws FrameGrabber.Exception, InterruptedException {
        IplImage imgHsv = cvCreateImage(cvSize(img.width(), img.height()), img.depth(), img.nChannels());
        cvCvtColor(img, imgHsv, CV_BGR2HSV);

        IplImage imgThreshold = cvCreateImage(cvGetSize(imgHsv), 8, 1);
        CvScalar hsv_min = cvScalar(hueLower, satLower, briLower, 0);
        CvScalar hsv_max = cvScalar(hueUpper, satUpper, briUpper, 0);
        cvInRangeS(imgHsv, hsv_min, hsv_max, imgThreshold);

        cvSmooth(imgThreshold, imgThreshold, CV_MEDIAN, 15,0,0,0);
        IplImage imgDilated = cvCreateImage(cvGetSize(imgThreshold), 8, 1);
        cvDilate(imgThreshold, imgDilated);

        CvSeq bigContour = findBiggestContour(imgDilated);
        if (bigContour == null) {
            return;
        }

        extractCentroid(bigContour);
        largestContour = bigContour;
    }

    private CvSeq findBiggestContour(IplImage imgThreshed) {
        CvSeq bigContour = null;

        CvSeq contours = new CvSeq(null);
        cvFindContours(imgThreshed, storage, contours, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
        float maxArea = SMALLEST_AREA;
        while (contours != null && !contours.isNull()) {
            if (contours.elem_size() > 0) {
                CvBox2D box = cvMinAreaRect2(contours, storage);
                if (box != null) {
                    CvSize2D32f size = box.size();
                    float area = size.width() * size.height();
                    if (area > maxArea) {
                        maxArea = area;
                        bigContour = contours;
                    }
                }
            }
            contours = contours.h_next();
        }
        return bigContour;
    }

    private void extractCentroid(CvSeq bigContour) {
        CvMoments moments = new CvMoments();
        cvMoments(bigContour, moments, 1);

        // center of gravity
        double m00 = cvGetSpatialMoment(moments, 0, 0);
        double m10 = cvGetSpatialMoment(moments, 1, 0);
        double m01 = cvGetSpatialMoment(moments, 0, 1);

        if (m00 != 0) {   // calculate center
            centroidX = (int) Math.round(m10 / m00);
            centroidY = (int) Math.round(m01 / m00);
        }
    }

    public CvSeq getLargestContour() { return largestContour; }

    public int getYCentroid() { return centroidY; }

}

