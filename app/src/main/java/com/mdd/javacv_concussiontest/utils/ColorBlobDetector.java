package com.mdd.javacv_concussiontest.utils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core.CvBox2D;
import org.bytedeco.javacpp.opencv_core.CvContour;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.CvSize;
import org.bytedeco.javacpp.opencv_core.CvSize2D32f;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FrameGrabber;

import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_32F;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvInRangeS;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_core.cvScalar;
import static org.bytedeco.javacpp.opencv_core.cvScale;
import static org.bytedeco.javacpp.opencv_core.cvSeqPush;
import static org.bytedeco.javacpp.opencv_core.cvSize;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2HSV_FULL;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2HSV;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_GAUSSIAN;
import static org.bytedeco.javacpp.opencv_imgproc.CV_MOP_OPEN;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_LIST;
import static org.bytedeco.javacpp.opencv_imgproc.cvContourArea;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvDilate;
import static org.bytedeco.javacpp.opencv_imgproc.cvFindContours;
import static org.bytedeco.javacpp.opencv_imgproc.cvMinAreaRect2;
import static org.bytedeco.javacpp.opencv_imgproc.cvMorphologyEx;
import static org.bytedeco.javacpp.opencv_imgproc.cvPyrDown;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
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
    private CvSeq imgContours = new CvSeq();
    private CvMemStorage contourStorage;
    private CvMemStorage storage = null;
    private static final double IMG_SCALE = 2;
    private int hueLower, hueUpper, satLower, satUpper, briLower, briUpper;
    private IplImage scaleImg =  new IplImage();
    private IplImage hsvImg;
    private IplImage imgDilated;
    private IplImage imgThreshed;
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

    //TODO - why am I putting pxscale in here? it isnt used.  Should only be used in videoprocessor
    public void update(IplImage img) throws FrameGrabber.Exception, InterruptedException {
        String LOG_TAG = "VideoProcessing";
        double mMinContourArea = 0.1;
        IplImage hsvImg = new IplImage();
        IplImage maskImg = new IplImage();
        IplImage dilatedImg = new IplImage();
        IplImage heirarchyImg = new IplImage();

        for (int k = 0; k < 2; k++) {
            //Gaussian Blur
            //GaussianBlur(mRgba, mRgba, new Size(3, 3), 1);
            cvSmooth(img, img, CV_GAUSSIAN, 3, 3, 1, 1);

            //pyrdown x2
            //TODO - figure out why mPyrDownMat null after this
            CvSize pyr_sza = cvSize(img.width() / 2, img.height() / 2);
            IplImage pyrA = cvCreateImage(pyr_sza, IPL_DEPTH_32F, 1);
            CvSize pyr_szb = cvSize(pyrA.width() / 2, pyrA.height() / 2);
            IplImage pyrB = cvCreateImage(pyr_szb, IPL_DEPTH_32F, 1);
            cvPyrDown(img, pyrA);
            cvPyrDown(pyrA, pyrB);

            //convert RBGA to HSV colorspace
            //cvtColor(mPyrDownMat, mHsvMat, COLOR_RGB2HSV_FULL);
            cvCvtColor(pyrB, hsvImg, COLOR_RGB2HSV_FULL);

            //threshold hsv image for color range
            //inRange(intensity, new Mat(new Size(4, 1), CV_64FC1, zeroScalar), new Mat(new Size(4, 1), CV_64FC1, black_level), mask);
            //TODO - remove this (temporarily hard code thresholds)
            /*
            if (k == 0) {
                mLowerBound = new Scalar(135.26, 163.81, 49.42, 0);
                mUpperBound = new Scalar(185.26, 263.81, 149.42, 255);
            } else {
                mLowerBound = new Scalar(221.94, 177.92, 102.55, 0);
                mUpperBound = new Scalar(255, 277.92, 202.55, 255);
            }
            */
            //blue, green, red order
            CvScalar lowerBound;
            CvScalar upperBound;
            if (k == 0) {
                lowerBound = new CvScalar(135.26, 163.81, 49.42, 0);
                upperBound = new CvScalar(185.26, 263.81, 149.42, 255);
            } else {
                lowerBound = new CvScalar(221.94, 177.92, 102.55, 0);
                upperBound = new CvScalar(255, 277.92, 202.55, 255);
            }

            /*
            //TODO I'm setting these as CV_8UC4 but this may be incorrect. CONFIRM THIS
            DoublePointer dpl = new DoublePointer(mLowerBound.get(0), mLowerBound.get(1), mLowerBound.get(2), mLowerBound.get(3));
            Mat lowerBoundMat = new Mat(1, 1, CV_8UC4, dpl);
            DoublePointer dph = new DoublePointer(mLowerBound.get(0), mLowerBound.get(1), mLowerBound.get(2), mLowerBound.get(3));
            Mat upperBoundMat = new Mat(1, 1, CV_8UC4, dph);
            */

            //TODO - figure out why mHsvMay and mMask are both null after this
            //inRange(src, low, high, dst)
            //inRange(mHsvMat, lowerBoundMat, upperBoundMat, mMask);
            cvInRangeS(hsvImg, lowerBound, upperBound, maskImg);

            //dilate
            //dilate(mMask, mDilatedMask, new Mat());
            cvDilate(maskImg, dilatedImg);

            //initialize list of contours
            //MatVector contours = new MatVector();

            //find contours from the dilated matrix
            //findContours(mDilatedMask, contours, mHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            CvSeq contour = new CvSeq();
            cvFindContours(dilatedImg, storage, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

            //find max contour area
            double maxArea = 0;

            //iterate over list of contour vectors
            while (contour != null && !contour.isNull()) {
                //contour - Input vector of 2D points (contour vertices)
                double area = cvContourArea(contour);
                if (area > maxArea)
                    maxArea = area;

                // take the next contour
                contour = contour.h_next();
            }

            //filter contours by area and resize to fit original image size
            while (contour != null && !contour.isNull()) {
                if (cvContourArea(contour) > mMinContourArea*maxArea) {
                    //multiply - Calculates the per-element scaled product of two arrays.
                    //multiply(Mat src1, Scalar src2, Mat dst)
                    //src1 - First source array.
                    //src2 - Second source array of the same size and the same type as src1.
                    //dst - Destination array of the same size and type as src1.
                    //multiply(4, contour);
                    cvSeqPush(imgContours, contour);
                    //imgContours.put(contour);
                }

                // take the next contour
                contour = contour.h_next();
            }
        }
    }

    //TODO learn more about what these do - make this into the new process function
    //maybe I just keep biggest contour and feed that into the video processor
    //because I think the video processor just ends up getting the biggest contour anyway
    public void process(IplImage im) throws FrameGrabber.Exception, InterruptedException {
        cvSmooth(im, im, CV_GAUSSIAN, 3, 3, 1, 1);
        //TODO - fix error in line below
        //Assertion failed (src.type() == dst.type())
        //cvResize(im, scaleImg);
        //TODO fix below
        //Assertion failed (dst.data == dst0.data) in void cvCvtColor
        cvCvtColor(im, hsvImg, CV_BGR2HSV);
        cvInRangeS(
                hsvImg,
                cvScalar(hueLower, satLower, briLower, 0),
                cvScalar(hueUpper, satUpper, briUpper, 0), imgThreshed);
        cvDilate(imgThreshed, imgDilated);
        cvMorphologyEx(imgDilated, imgDilated, null, null, CV_MOP_OPEN, 1);
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
        cvFindContours(imgThreshed, contourStorage, contours, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
        float maxArea = SMALLEST_AREA;
        CvBox2D maxBox = null;
        while (contours != null && !contours.isNull()) {
            if (contours.elem_size() > 0) {
                CvBox2D box = cvMinAreaRect2(contours, contourStorage);
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

    public CvSeq getContours() {
        return imgContours;
    }

    public CvSeq getLargestContour() { return largestContour; }

    public int getYCentroid() { return centroidY; }

}

