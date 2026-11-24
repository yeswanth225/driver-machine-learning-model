package com.drivermonitoring.sensorlogger

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

object DataUploader {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    fun uploadRecording(baseUrl: String, csvFile: File, driverId: String, sessionId: String): Boolean {
        return try {
            val endpoint = if (baseUrl.endsWith("/")) "${baseUrl}api/sensor/upload" else "${baseUrl}/api/sensor/upload"

            val sha256 = computeSha256(csvFile)
            val fileBody = csvFile.asRequestBody("text/csv".toMediaTypeOrNull())

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("driver_id", driverId)
                .addFormDataPart("session_id", sessionId)
                .addFormDataPart("client_sha256", sha256)
                .addFormDataPart("file", csvFile.name, fileBody)
                .build()

            val request = Request.Builder()
                .url(endpoint)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            response.isSuccessful
        } catch (_: Exception) {
            false
        }
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.forEachBlock { buffer, bytesRead ->
            digest.update(buffer, 0, bytesRead)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
