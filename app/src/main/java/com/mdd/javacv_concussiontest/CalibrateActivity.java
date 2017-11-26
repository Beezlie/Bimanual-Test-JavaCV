package com.mdd.javacv_concussiontest;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

public class CalibrateActivity extends AppCompatActivity {
    private int focal_mm;
    private int sensor_width_mm;
    private EditText focalDistanceField;
    private EditText sensorWidthField;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calibrate);

        focalDistanceField = (EditText) findViewById(R.id.focal_mm);
        sensorWidthField = (EditText) findViewById(R.id.sensor_width_mm);
    }

    public void sendForm(View button) {
        // Do click handling here
        focal_mm = Integer.parseInt(focalDistanceField.getText().toString());
        sensor_width_mm = Integer.parseInt(sensorWidthField.getText().toString());

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putInt("focal_mm", focal_mm);
        editor.putInt("sensor_width_mm", sensor_width_mm);

        Context context = getApplicationContext();
        CharSequence text = "Parameters successfully saved";
        int duration = Toast.LENGTH_SHORT;

        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }
}
