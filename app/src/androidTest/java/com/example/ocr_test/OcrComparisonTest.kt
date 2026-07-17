package com.example.ocr_test

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.ocr_test.ocr.MnnOcrEngine
import com.example.ocr_test.ocr.OnnxOcrEngine
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.IOException

/**
 * 综合 OCR 对比测试 — 运行 5 种引擎对比识别效果.
 *
 * 引擎列表:
 *   1. Google ML Kit（中文 bundled）
 *   2. PP-OCRv6 ONNX small（精度优先）
 *   3. PP-OCRv6 ONNX tiny（速度优先）
 *   4. PP-OCRv6 MNN small（精度优先）
 *   5. PP-OCRv6 MNN tiny（速度优先）
 *
 * 每张测试图输出到一个 .txt 文件，命名格式: {engine}_result_{image}.txt
 * 运行后通过 adb pull 拉取:
 *   adb exec-out run-as com.example.ocr_test cat /data/data/com.example.ocr_test/files/<filename> > results/<filename>
 */
@RunWith(AndroidJUnit4::class)
class OcrComparisonTest {

    private val context: Context = ApplicationProvider.getApplicationContext()
    private val images = listOf("full_fields.jpg", "date_field.jpg")

    // ─── 1. Google ML Kit ─────────────────────────────────────

    @Test
    fun google_full_fields() = runGoogle("full_fields.jpg")
    @Test
    fun google_date_field() = runGoogle("date_field.jpg")

    private fun runGoogle(imageName: String) {
        val bitmap = loadBitmap(imageName)
        val recognizer = TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val result = com.google.android.gms.tasks.Tasks.await(recognizer.process(image))
            saveResult("google", imageName, result.text)
        } finally {
            recognizer.close()
        }
    }

    // ─── 2. ONNX small ────────────────────────────────────────

    @Test
    fun onnx_small_full_fields() = runOnnx("full_fields.jpg", OnnxOcrEngine.ModelSize.SMALL)
    @Test
    fun onnx_small_date_field() = runOnnx("date_field.jpg", OnnxOcrEngine.ModelSize.SMALL)

    // ─── 3. ONNX tiny ─────────────────────────────────────────

    @Test
    fun onnx_tiny_full_fields() = runOnnx("full_fields.jpg", OnnxOcrEngine.ModelSize.TINY)
    @Test
    fun onnx_tiny_date_field() = runOnnx("date_field.jpg", OnnxOcrEngine.ModelSize.TINY)

    private fun runOnnx(imageName: String, size: OnnxOcrEngine.ModelSize) {
        val bitmap = loadBitmap(imageName)
        val tag = "onnx_${size.name.lowercase()}"
        OnnxOcrEngine(context, size).use { engine ->
            val lines = engine.ocr(bitmap)
            val allText = lines.joinToString("\n") { it.text }
            println("========== ONNX ${size.name} Result: $imageName ==========")
            println("共检测到 ${lines.size} 行文字:")
            lines.forEach { line ->
                val b = line.bbox
                println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
            }
            println("========================================================")
            println("\n合并文本:\n$allText")
            println("========================================================")
            saveResult(tag, imageName, allText)
        }
    }

    // ─── 4. MNN small ─────────────────────────────────────────

    @Test
    fun mnn_small_full_fields() = runMnn("full_fields.jpg", MnnOcrEngine.ModelSize.SMALL)
    @Test
    fun mnn_small_date_field() = runMnn("date_field.jpg", MnnOcrEngine.ModelSize.SMALL)

    // ─── 5. MNN tiny ──────────────────────────────────────────

    @Test
    fun mnn_tiny_full_fields() = runMnn("full_fields.jpg", MnnOcrEngine.ModelSize.TINY)
    @Test
    fun mnn_tiny_date_field() = runMnn("date_field.jpg", MnnOcrEngine.ModelSize.TINY)

    private fun runMnn(imageName: String, size: MnnOcrEngine.ModelSize) {
        val bitmap = loadBitmap(imageName)
        val tag = "mnn_${size.name.lowercase()}"
        MnnOcrEngine(context, size).use { engine ->
            val lines = engine.ocr(bitmap)
            val allText = lines.joinToString("\n") { it.text }
            println("========== MNN ${size.name} Result: $imageName ==========")
            println("共检测到 ${lines.size} 行文字:")
            lines.forEach { line ->
                val b = line.bbox
                println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
            }
            println("========================================================")
            println("\n合并文本:\n$allText")
            println("========================================================")
            saveResult(tag, imageName, allText)
        }
    }

    // ─── 辅助方法 ─────────────────────────────────────────────

    private fun saveResult(engine: String, imageName: String, text: String) {
        val fileName = "${engine}_result_$imageName.txt"
        val file = File(context.filesDir, fileName)
        file.writeText(text)
        println("[SAVED] $engine result → ${file.absolutePath}")
    }

    private fun loadBitmap(name: String): android.graphics.Bitmap {
        val stream = context.assets.open(name)
        val bitmap = BitmapFactory.decodeStream(stream)
            ?: throw IOException("Failed to decode bitmap from assets: $name")
        stream.close()
        return bitmap
    }
}
