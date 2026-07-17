package com.example.ocr_test

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException

/**
 * Instrumented test for Google ML Kit OCR (bundled model).
 *
 * Runs on a real Android device. Test images are in androidTest/assets/.
 *
 * ML Kit dependencies:
 *   - com.google.mlkit:text-recognition:16.0.1   (base Latin bundled)
 *   - com.google.mlkit:text-recognition-chinese:16.0.1  (Chinese bundled)
 *
 * Both models are bundled in the APK (offline, no Play Services required).
 */
@RunWith(AndroidJUnit4::class)
class GoogleMlKitOcrTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
    }

    @Test
    fun test_full_fields_ocr() {
        val imageName = "full_fields.jpg"
        val bitmap = loadBitmap(imageName)

        // Use Chinese-specific bundled recognizer for better CJK accuracy
        val recognizer = TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())

        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val result = com.google.android.gms.tasks.Tasks.await(recognizer.process(image))

            val actualText = result.text
            println("========== Google ML Kit OCR Result: $imageName ==========")
            println(actualText)
            println("=============================================================")
            saveResult(imageName, actualText)

            // ── Verify key expected fields are present ──
            val expectedLines = listOf(
                "奥美拉唑肠溶胶囊",
                "国药准字 H20046430",
                "药品上市许可持有人",
                "石药集团欧意药业有限公司",
                "生产企业"
            )

            val missing = expectedLines.filter { it !in actualText }
            assertTrue(
                "Missing expected text lines in OCR result for $imageName:\n" +
                        missing.joinToString("\n") + "\n\nFull OCR result:\n$actualText",
                missing.isEmpty()
            )
        } finally {
            recognizer.close()
        }
    }

    @Test
    fun test_date_field_ocr() {
        val imageName = "date_field.jpg"
        val bitmap = loadBitmap(imageName)

        val recognizer = TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())

        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val result = com.google.android.gms.tasks.Tasks.await(recognizer.process(image))

            val actualText = result.text
            println("========== Google ML Kit OCR Result: $imageName ==========")
            println(actualText)
            println("=============================================================")
            saveResult(imageName, actualText)

            // ── Verify key expected fields ──
            val expectedFields = listOf(
                "产品批号",
                "生产日期",
                "有效期至"
            )
            val missingFields = expectedFields.filter { it !in actualText }

            assertTrue(
                "Missing expected date-field labels in OCR result for $imageName:\n" +
                        missingFields.joinToString("\n") + "\n\nFull OCR result:\n$actualText",
                missingFields.isEmpty()
            )
        } finally {
            recognizer.close()
        }
    }

    // -------------------------------------------------------------------
    //  Helpers
    // -------------------------------------------------------------------

    private fun saveResult(imageName: String, text: String) {
        val file = java.io.File(context.filesDir, "ocr_result_$imageName.txt")
        file.writeText(text)
        println("[SAVED] OCR result → ${file.absolutePath}")
    }

    private fun loadBitmap(name: String): android.graphics.Bitmap {
        val stream = context.assets.open(name)
        val bitmap = BitmapFactory.decodeStream(stream)
            ?: throw IOException("Failed to decode bitmap from assets: $name")
        stream.close()
        return bitmap
    }
}
