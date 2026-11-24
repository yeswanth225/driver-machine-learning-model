package com.drivermonitoring.sensorlogger

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import java.io.File
import java.io.FileWriter
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.*

class SensorLoggingService : Service(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var wakeLock: PowerManager.WakeLock? = null

    // Latest sensor cache with hardware event timestamps
    @Volatile private var latestAccX = 0.0f
    @Volatile private var latestAccY = 0.0f
    @Volatile private var latestAccZ = -9.81f
    @Volatile private var latestAccEventNanos = 0L

    @Volatile private var latestGyroX = 0.0f
    @Volatile private var latestGyroY = 0.0f
    @Volatile private var latestGyroZ = 0.0f
    @Volatile private var latestGyroEventNanos = 0L

    @Volatile private var latestLat = 0.0
    @Volatile private var latestLon = 0.0
    @Volatile private var latestSpeedMps = 0.0f
    @Volatile private var latestHeadingDeg = 0.0f
    @Volatile private var latestAltitude = 0.0
    @Volatile private var latestHdop = 1.0f

    @Volatile private var currentSegmentId = "S1"
    @Volatile private var isRecording = false

    private var driverId = "DRV_001"
    private var sessionId = ""
    private var sessionStartTimeNanos = 0L
    private var sessionStartEpochMs = 0L

    private var sampleWriter: FileWriter? = null
    private var outputFile: File? = null
    private var sampleCount = 0L

    // Sampling loop: precisely 20.0 Hz (50 ms interval)
    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var samplingJob: Job? = null

    companion object {
        const val ACTION_START = "ACTION_START"
        const val ACTION_STOP = "ACTION_STOP"
        const val ACTION_SET_SEGMENT = "ACTION_SET_SEGMENT"
        const val EXTRA_DRIVER_ID = "EXTRA_DRIVER_ID"
        const val EXTRA_SESSION_ID = "EXTRA_SESSION_ID"
        const val EXTRA_SEGMENT_ID = "EXTRA_SEGMENT_ID"
        const val CHANNEL_ID = "SensorLoggerChannel"
        const val NOTIFICATION_ID = 101
        const val TARGET_SAMPLE_INTERVAL_MS = 50L // 20.0 Hz
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                driverId = intent.getStringExtra(EXTRA_DRIVER_ID) ?: "DRV_001"
                sessionId = intent.getStringExtra(EXTRA_SESSION_ID) ?: "sess_${System.currentTimeMillis()}"
                startLogging()
            }
            ACTION_STOP -> {
                stopLogging()
                stopSelf()
            }
            ACTION_SET_SEGMENT -> {
                val seg = intent.getStringExtra(EXTRA_SEGMENT_ID)
                if (seg != null) currentSegmentId = seg
            }
        }
        return START_STICKY
    }

    private fun startLogging() {
        if (isRecording) return
        isRecording = true

        startForeground(NOTIFICATION_ID, buildNotification("Recording 20 Hz Telemetry for $driverId..."))

        // Safely acquire partial wake lock to prevent CPU sleep during multi-minute drives.
        // NOTE: Wake lock prevents CPU standby but does NOT guarantee against vendor-specific battery killers.
        try {
            val powerManager = getSystemService(Context.POWER_SERVICE) as? PowerManager
            wakeLock = powerManager?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "IDAMS::SensorLoggerWakeLock")?.apply {
                acquire(12 * 60 * 60 * 1000L) // 12 hours timeout safety
            }
        } catch (_: Exception) {}

        // Register hardware sensors at SENSOR_DELAY_GAME (~50Hz max rate, sampled cleanly at 20Hz)
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 500L, 0.0f, this)
        } catch (_: SecurityException) {
            // Location permission handled by MainActivity
        }

        sessionStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        sessionStartEpochMs = System.currentTimeMillis()
        sampleCount = 0L

        // Initialize output file
        val dir = File(getExternalFilesDir(null), "recordings/$driverId")
        if (!dir.exists()) dir.mkdirs()
        outputFile = File(dir, "raw_telemetry_${sessionId}.csv")
        sampleWriter = FileWriter(outputFile!!)

        // Write canonical header matching acceptance criteria
        sampleWriter!!.write("timestamp,timestamp_epoch_ms,Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z,speed_mps,heading_deg,latitude,longitude,altitude,hdop,segment_id\n")

        // Start deterministic 20 Hz loop (50 ms periods)
        samplingJob = serviceScope.launch {
            var nextTick = System.currentTimeMillis()
            while (isActive && isRecording) {
                writeSample()
                nextTick += TARGET_SAMPLE_INTERVAL_MS
                val sleepTime = nextTick - System.currentTimeMillis()
                if (sleepTime > 0) {
                    delay(sleepTime)
                } else {
                    // Slight catch-up
                    nextTick = System.currentTimeMillis()
                }
            }
        }
    }

    private fun writeSample() {
        val nowNanos = SystemClock.elapsedRealtimeNanos()
        val elapsedSec = (nowNanos - sessionStartTimeNanos) / 1_000_000_000.0
        val nowEpochMs = System.currentTimeMillis()

        // Convert m/s^2 to g (1 g = 9.80665 m/s^2) for compatibility with Dataset C format
        val accX_g = latestAccX / 9.80665
        val accY_g = latestAccY / 9.80665
        val accZ_g = latestAccZ / 9.80665

        val line = String.format(
            Locale.US,
            "%.4f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.1f,%.6f,%.6f,%.1f,%.2f,%s\n",
            elapsedSec, nowEpochMs,
            accX_g, accY_g, accZ_g,
            latestGyroX, latestGyroY, latestGyroZ,
            latestSpeedMps, latestHeadingDeg,
            latestLat, latestLon, latestAltitude, latestHdop,
            currentSegmentId
        )

        try {
            sampleWriter?.write(line)
            sampleCount++
        } catch (_: Exception) {}
    }

    private fun stopLogging() {
        isRecording = false
        samplingJob?.cancel()

        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)

        // Safely release wake lock
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (_: Exception) {}
        wakeLock = null

        try {
            sampleWriter?.flush()
            sampleWriter?.close()
        } catch (_: Exception) {}

        // Compute SHA-256 checksum
        outputFile?.let { file ->
            if (file.exists()) {
                val sha256 = computeSha256(file)
                val checksumFile = File(file.parentFile, "sha256_${sessionId}.txt")
                checksumFile.writeText(sha256)
            }
        }
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.forEachBlock { buffer, bytesRead ->
            digest.update(buffer, 0, bytesRead)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccX = event.values[0]
                latestAccY = event.values[1]
                latestAccZ = event.values[2]
                latestAccEventNanos = event.timestamp
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyroX = event.values[0]
                latestGyroY = event.values[1]
                latestGyroZ = event.values[2]
                latestGyroEventNanos = event.timestamp
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        latestLat = location.latitude
        latestLon = location.longitude
        latestSpeedMps = location.speed
        latestHeadingDeg = location.bearing
        latestAltitude = location.altitude
        latestHdop = if (location.hasAccuracy()) location.accuracy / 5.0f else 1.0f
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Driver Telemetry Logger Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Driver Sensor Logger")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopLogging()
        serviceScope.cancel()
        super.onDestroy()
    }
}
