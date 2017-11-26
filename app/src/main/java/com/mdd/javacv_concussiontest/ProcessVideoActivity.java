package com.mdd.javacv_concussiontest;

import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import static android.content.ContentValues.TAG;

public class ProcessVideoActivity extends AppCompatActivity {
    String LOG_TAG = "ProcessVideoActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_processvideo);

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        String filename = prefs.getString("filename", null);
        if (filename == null) {
            Log.i(TAG, "Video file bimanualtest.mp4 does not exist");
        }
        int pxscale = prefs.getInt("pxscale", 0);
        if (pxscale == 0) {
            Log.i(TAG, "Camera was not calibrated correctly");
        }

        VideoProcessor vidProc = new VideoProcessor(filename, pxscale);
    }
}
