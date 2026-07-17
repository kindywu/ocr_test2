package com.example.ocr_test

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.ocr_test.ocr.MnnOcrEngine
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException

/**
 * Instrumented test for PP-OCRv6 MNN OCR Engine (MNN SMALL 模型, 精度优先).
 *
 * 运行在真机或模拟器上, 需要 arm64-v8a 架构 + libMNN.so / libmnn_ocr.so.
 * 测试图片位于 androidTest/assets/.
 * SMALL 模型文件位于 assets/model/mnn/PP-OCRv6_small_{det,rec}.mnn
 * 字符字典来自 inference.yml（与 UNNX 引擎一致）。
 *
 * 运行:
 *   ./gradlew :app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.example.ocr_test.MnnOcrEngineTest
 *
 * 查看结果:
 *   adb logcat -s MnnOcrTest
 *   adb pull /data/data/com.example.ocr_test/files/
 */
@RunWith(AndroidJUnit4::class)
class MnnOcrEngineTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
    }

    @After
    fun tearDown() {
        // 引擎在每次测试中通过 use {} 自动 close
    }

    @Test
    fun test_full_fields_ocr() {
        val imageName = "full_fields.jpg"
        val bitmap = loadBitmap(imageName)

        MnnOcrEngine(context, MnnOcrEngine.ModelSize.SMALL).use { engine ->
            val lines = engine.ocr(bitmap)
            val allText = lines.joinToString("\n") { it.text }

            println("========== MNN OCR (SMALL) Result: $imageName ==========")
            println("共检测到 ${lines.size} 行文字:")
            lines.forEach { line ->
                val b = line.bbox
                println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
            }
            println("========================================================")
            println("\n合并文本:\n$allText")
            println("========================================================")

            saveResult(imageName, allText)

            // ── 验证关键字段（SMALL 精度高）──
            val expectedLines = listOf(
                "奥美拉唑肠溶胶囊",
                "国药准字H2004",
                "有限公司",
            )
            val missing = expectedLines.filter { it !in allText }
            assertTrue(
                "MNN OCR (SMALL) 检测结果缺少以下预期字段:\n" +
                        missing.joinToString("\n") +
                        "\n\n完整 OCR 结果:\n$allText",
                missing.isEmpty()
            )
        }
    }

    @Test
    fun test_date_field_ocr() {
        val imageName = "date_field.jpg"
        val bitmap = loadBitmap(imageName)

        MnnOcrEngine(context, MnnOcrEngine.ModelSize.SMALL).use { engine ->
            val lines = engine.ocr(bitmap)
            val allText = lines.joinToString("\n") { it.text }

            println("========== MNN OCR (SMALL) Result: $imageName ==========")
            println("共检测到 ${lines.size} 行文字:")
            lines.forEach { line ->
                val b = line.bbox
                println("  [(${b[0]},${b[1]})-(${b[2]},${b[3]})] ${line.score} ${line.text}")
            }
            println("========================================================")

            saveResult(imageName, allText)

            // ── 验证关键字段 ──
            val expectedFields = listOf(
                "产品批号",
                "生产日期",
                "有效期至",
            )
            val missing = expectedFields.filter { it !in allText }
            assertTrue(
                "MNN OCR (SMALL) 检测结果中缺少以下日期字段:\n" +
                        missing.joinToString("\n") +
                        "\n\n完整 OCR 结果:\n$allText",
                missing.isEmpty()
            )
        }
    }

    // ── 辅助方法 ────────────────────────────────────────

    private fun saveResult(imageName: String, text: String) {
        val file = java.io.File(context.filesDir, "mnn_ocr_small_result_$imageName.txt")
        file.writeText(text)
        println("[SAVED] MNN OCR (SMALL) result → ${file.absolutePath}")
    }

    private fun loadBitmap(name: String): android.graphics.Bitmap {
        val stream = context.assets.open(name)
        val bitmap = BitmapFactory.decodeStream(stream)
            ?: throw IOException("Failed to decode bitmap from assets: $name")
        stream.close()
        return bitmap
    }
}
