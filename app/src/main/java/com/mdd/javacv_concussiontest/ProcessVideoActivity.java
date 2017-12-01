package com.mdd.javacv_concussiontest;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import com.google.gson.Gson;

import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.FrameGrabber;

import static android.content.ContentValues.TAG;

public class ProcessVideoActivity extends AppCompatActivity {
    String LOG_TAG = "ProcessVideoActivity";
    String filename = Environment.getExternalStorageDirectory() + "/bimanualtest.mp4";
    int pxscale;
    Scalar mUpperBound[];
    Scalar mLowerBound[];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_processvideo);

        //get pxscale and upper/lower thresholds
        getParams();
    }

    public void getParams() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();

        int pxscale = prefs.getInt("pxscale", 0);
        if (pxscale == 0) {
            Log.i(TAG, "Camera was not calibrated correctly");

            Context context = getApplicationContext();
            CharSequence text = "Camera was not calibrated correctly";
            int duration = Toast.LENGTH_SHORT;

            Toast toast = Toast.makeText(context, text, duration);
            toast.show();

            //return to main activity
            //startActivity(new Intent(ProcessVideoActivity.this, MainActivity.class));
        }

        //TODO - get shared prefs t work for these
        /*
        String json;
        Gson gson = new Gson();
        json = prefs.getString("left_lower_bound", "");
        Scalar leftLowerBound = gson.fromJson(json, Scalar.class);
        json = prefs.getString("right_lower_bound", "");
        Scalar rightLowerBound = gson.fromJson(json, Scalar.class);
        json = prefs.getString("left_upper_bound", "");
        Scalar leftUpperBound = gson.fromJson(json, Scalar.class);
        json = prefs.getString("right_upper_bound", "");
        Scalar rightUpperBound = gson.fromJson(json, Scalar.class);

        mLowerBound[0] = leftLowerBound;
        mLowerBound[1] = rightLowerBound;
        mUpperBound[0] = leftUpperBound;
        mUpperBound[1] = rightUpperBound;
        */
    }
}
