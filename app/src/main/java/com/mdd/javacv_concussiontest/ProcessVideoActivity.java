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

import org.bytedeco.javacv.FrameGrabber;

import static android.content.ContentValues.TAG;

public class ProcessVideoActivity extends AppCompatActivity {
    String LOG_TAG = "ProcessVideoActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_processvideo);

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        String filename = Environment.getExternalStorageDirectory() + "/bimanualtest.mp4";
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

        VideoProcessor vidProc = new VideoProcessor(filename, pxscale);
        try {
            vidProc.process(filename, pxscale);
        } catch (InterruptedException e) {

        } catch (FrameGrabber.Exception e) {

        }
    }
}
