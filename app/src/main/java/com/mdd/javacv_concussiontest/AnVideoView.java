package com.mdd.javacv_concussiontest;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.MediaController;
import android.widget.VideoView;

import java.io.File;

public class AnVideoView extends Activity {
    File savePath = new File(Environment.getExternalStorageDirectory(), "bimanualtest.mp4");
    String SrcPath = savePath.getAbsolutePath();

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_videoview);
        VideoView myVideoView = (VideoView)findViewById(R.id.myvideoview);
        myVideoView.setVideoPath(SrcPath);
        myVideoView.setMediaController(new MediaController(this));
        myVideoView.requestFocus();
        myVideoView.start();
    }
}

