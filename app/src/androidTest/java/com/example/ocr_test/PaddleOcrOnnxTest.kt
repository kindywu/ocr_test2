package com.example.ocr_test

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.ocr_test.ocr.OnnxOcrEngine
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException

/**
 * Instrumented test for PP-OCRv6 ONNX OCR Engine (纯 onnxruntime 实现).
 *
 * 运行在真机或模拟器上, 需要 arm64-v8a 架构.
 * 测试图片位于 androidTest/assets/.
 */
@RunWith(AndroidJUnit4::class)
class PaddleOcrOnnxTest {

    private lateinit var context: Context
    private lateinit var engine: OnnxOcrEngine

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        engine = OnnxOcrEngine(context)
    }

    @After
    fun tearDown() {
        engine.close()
    }

    @Test
    fun test_full_fields_ocr() {
        val imageName = "full_fields.jpg"
        val bitmap = loadBitmap(imageName)

        val lines = engine.ocr(bitmap)
        val allText = lines.joinToString("\n") { it.text }

        println("========== PP-OCRv6 ONNX Result: $imageName ==========")
        println("共检测到 ${lines.size} 行文字:")
        lines.forEach { line ->
            val b = line.bbox
            println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
        }
        println("======================================================")
        println("\n合并文本:\n$allText")
        println("======================================================")

        saveResult(imageName, allText)

        // ── 验证关键字段 ──
        val expectedLines = listOf(
            "奥美拉唑肠溶胶囊",
            "国药准字 H20046430",
            "药品上市许可持有人",
            "石药集团欧意药业有限公司",
            "生产企业",
        )

        val missing = expectedLines.filter { it !in allText }
        assertTrue(
            "PP-OCRv6 检测结果中缺少以下预期字段:\n" +
                    missing.joinToString("\n") +
                    "\n\n完整 OCR 结果:\n$allText",
            missing.isEmpty()
        )
    }

    @Test
    fun test_date_field_ocr() {
        val imageName = "date_field.jpg"
        val bitmap = loadBitmap(imageName)

        val lines = engine.ocr(bitmap)
        val allText = lines.joinToString("\n") { it.text }

        println("========== PP-OCRv6 ONNX Result: $imageName ==========")
        println("共检测到 ${lines.size} 行文字:")
        lines.forEach { line ->
            val b = line.bbox
            println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
        }
        println("======================================================")

        saveResult(imageName, allText)

        // ── 验证关键字段 ──
        val expectedFields = listOf(
            "产品批号",
            "生产日期",
            "有效期至",
        )
        val missing = expectedFields.filter { it !in allText }
        assertTrue(
            "PP-OCRv6 检测结果中缺少以下日期字段:\n" +
                    missing.joinToString("\n") +
                    "\n\n完整 OCR 结果:\n$allText",
            missing.isEmpty()
        )
    }

    // ── 辅助方法 ────────────────────────────────────────

    private fun saveResult(imageName: String, text: String) {
        val file = java.io.File(context.filesDir, "ppocr_onnx_result_$imageName.txt")
        file.writeText(text)
        println("[SAVED] PP-OCRv6 result → ${file.absolutePath}")
    }

    private fun loadBitmap(name: String): android.graphics.Bitmap {
        val stream = context.assets.open(name)
        val bitmap = BitmapFactory.decodeStream(stream)
            ?: throw IOException("Failed to decode bitmap from assets: $name")
        stream.close()
        return bitmap
    }
}
