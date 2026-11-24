package com.drivermonitoring.sensorlogger

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvSamplingRate: TextView
    private lateinit var tvDuration: TextView
    private lateinit var tvGpsStatus: TextView
    private lateinit var spinnerDriver: Spinner
    private lateinit var spinnerSegment: Spinner
    private lateinit var etBackendUrl: EditText
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var btnUpload: Button

    private var isRecording = false
    private var recordingStartTime = 0L
    private val handler = Handler(Looper.getMainLooper())
    private var currentSessionId = ""

    private val PERMISSIONS_REQUIRED = arrayOf(
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
        Manifest.permission.BODY_SENSORS
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvSamplingRate = findViewById(R.id.tvSamplingRate)
        tvDuration = findViewById(R.id.tvDuration)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        spinnerDriver = findViewById(R.id.spinnerDriver)
        spinnerSegment = findViewById(R.id.spinnerSegment)
        etBackendUrl = findViewById(R.id.etBackendUrl)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        btnUpload = findViewById(R.id.btnUpload)

        setupSpinners()
        checkAndRequestPermissions()

        btnStart.setOnClickListener { startRecording() }
        btnStop.setOnClickListener { stopRecording() }
        btnUpload.setOnClickListener { uploadLastRecording() }

        updateUi()
    }

    private fun setupSpinners() {
        val drivers = arrayOf("DRV_001", "DRV_002", "DRV_003", "DRV_004", "DRV_005", "DRV_006")
        spinnerDriver.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, drivers)

        val segments = arrayOf(
            "S0 - Pre-Drive 30s Static Cal",
            "S1 - Urban Calm Cruising",
            "S2 - Arterial Road & Turns",
            "S3 - Highway Cruising (Smooth)",
            "S4 - Rough Road / Bumps",
            "S5 - Closed-Course Aggressive Battery",
            "S6 - Post-Drive 30s Static Cal"
        )
        spinnerSegment.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, segments)

        spinnerSegment.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, pos: Int, id: Long) {
                if (isRecording) {
                    val segCode = segments[pos].substring(0, 2)
                    val intent = Intent(this@MainActivity, SensorLoggingService::class.java).apply {
                        action = SensorLoggingService.ACTION_SET_SEGMENT
                        putExtra(SensorLoggingService.EXTRA_SEGMENT_ID, segCode)
                    }
                    startService(intent)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun checkAndRequestPermissions() {
        val missing = PERMISSIONS_REQUIRED.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 1001)
        }
    }

    private fun startRecording() {
        val driverId = spinnerDriver.selectedItem.toString()
        val segCode = spinnerSegment.selectedItem.toString().substring(0, 2)
        currentSessionId = "sess_${System.currentTimeMillis()}"

        val intent = Intent(this, SensorLoggingService::class.java).apply {
            action = SensorLoggingService.ACTION_START
            putExtra(SensorLoggingService.EXTRA_DRIVER_ID, driverId)
            putExtra(SensorLoggingService.EXTRA_SESSION_ID, currentSessionId)
            putExtra(SensorLoggingService.EXTRA_SEGMENT_ID, segCode)
        }
        ContextCompat.startForegroundService(this, intent)

        isRecording = true
        recordingStartTime = System.currentTimeMillis()
        handler.post(durationUpdater)
        updateUi()
    }

    private fun stopRecording() {
        val intent = Intent(this, SensorLoggingService::class.java).apply {
            action = SensorLoggingService.ACTION_STOP
        }
        startService(intent)

        isRecording = false
        handler.removeCallbacks(durationUpdater)
        updateUi()
    }

    private val durationUpdater = object : Runnable {
        override fun run() {
            if (isRecording) {
                val elapsedSec = (System.currentTimeMillis() - recordingStartTime) / 1000
                val mins = elapsedSec / 60
                val secs = elapsedSec % 60
                tvDuration.text = String.format("Duration: %02d:%02d", mins, secs)
                tvSamplingRate.text = "Target: 20.0 Hz (Nominal 50ms)"
                handler.postDelayed(this, 1000)
            }
        }
    }

    private fun uploadLastRecording() {
        val driverId = spinnerDriver.selectedItem.toString()
        val dir = File(getExternalFilesDir(null), "recordings/$driverId")
        val file = File(dir, "raw_telemetry_${currentSessionId}.csv")
        if (!file.exists()) {
            Toast.makeText(this, "Recording file not found: ${file.name}", Toast.LENGTH_SHORT).show()
            return
        }

        val url = etBackendUrl.text.toString().trim()
        tvStatus.text = "Uploading ${file.name} to $url..."

        CoroutineScope(Dispatchers.IO).launch {
            val success = DataUploader.uploadRecording(url, file, driverId, currentSessionId)
            withContext(Dispatchers.Main) {
                if (success) {
                    tvStatus.text = "Upload complete! Verified SHA-256."
                    Toast.makeText(this@MainActivity, "Upload Successful!", Toast.LENGTH_LONG).show()
                } else {
                    tvStatus.text = "Upload failed. Check backend connection."
                    Toast.makeText(this@MainActivity, "Upload Failed!", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun updateUi() {
        if (isRecording) {
            tvStatus.text = "RECORDING ACTIVE"
            btnStart.isEnabled = false
            btnStop.isEnabled = true
            btnUpload.isEnabled = false
            spinnerDriver.isEnabled = false
        } else {
            tvStatus.text = "STANDBY"
            btnStart.isEnabled = true
            btnStop.isEnabled = false
            btnUpload.isEnabled = currentSessionId.isNotEmpty()
            spinnerDriver.isEnabled = true
        }
    }
}
