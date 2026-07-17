package com.example.ocr_test.ocr

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream

/**
 * MNN PP-OCRv6 OCR 引擎 — JNI 实现
 *
 * 使用阿里 MNN 框架（C++ API via JNI）运行 PP-OCRv6 检测 + 识别模型。
 * 复用 ImageProcessors、DBPostProcessor、CTCDecoder 组件。
 *
 * 模型文件（assets/model/mnn/）:
 *   PP-OCRv6_small_det.mnn  — 检测（精度优先）
 *   PP-OCRv6_small_rec.mnn  — 识别（精度优先）
 *   PP-OCRv6_tiny_det.mnn   — 检测（速度优先）
 *   PP-OCRv6_tiny_rec.mnn   — 识别（速度优先）
 *
 * 字符字典:
 *   SMALL 从 model/mnn/inference_small.yml 加载（全量 18709 字典）
 *   TINY 从 model/rec/inference.yml 加载（TINY 专用 ~6906 字典）
 *
 * 依赖的 native 库:
 *   libMNN.so       — MNN 推理引擎（从 GitHub Releases 下载）
 *   libmnn_ocr.so   — 本文件的 JNI 桥接（由 CMake 构建）
 *   见 app/download_mnn.sh 和 app/src/main/cpp/CMakeLists.txt
 */
class MnnOcrEngine(context: Context, private val modelSize: ModelSize = ModelSize.SMALL) : AutoCloseable {

    data class OcrLine(
        val text: String,
        val score: Float,
        val bbox: IntArray, // [x1, y1, x2, y2]
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is OcrLine) return false
            return text == other.text && score == other.score && bbox.contentEquals(other.bbox)
        }
        override fun hashCode(): Int = text.hashCode() + score.hashCode() + bbox.contentHashCode()
    }

    enum class ModelSize { SMALL, TINY }

    private var nativeHandle: Long = 0
    private val charDict: List<String> = loadCharDict(context, modelSize)

    companion object {
        init {
            System.loadLibrary("mnn_ocr")
        }

        private fun loadCharDict(context: Context, modelSize: ModelSize): List<String> {
            val yamlPath = when (modelSize) {
                ModelSize.SMALL -> "model/ppocrv6_mnn/inference_small.yml"
                ModelSize.TINY -> "model/ppocrv6_mnn/inference_tiny.yml"
            }
            val dict = CharDictLoader.load(context, yamlPath)
            android.util.Log.d("MnnOcrEngine", "${modelSize.name} charDict size: ${dict.size} ($yamlPath)")
            return dict
        }
    }

    // ── JNI native methods ───────────────────────────────

    /** 初始化 MNN 模型，返回 native handle */
    private external fun nativeInit(detModelPath: String, recModelPath: String): Long

    /**
     * 运行检测模型
     * @return floatArray = [oW, oH, probMap...] 或 null
     */
    private external fun nativeDetect(handle: Long, input: FloatArray, h: Int, w: Int): FloatArray?

    /**
     * 运行识别模型
     * @return floatArray = [timeSteps, vocabSize, data...] 或 null
     *         data 的形状为 [batch=1, timeSteps, vocabSize]
     */
    private external fun nativeRecognize(handle: Long, input: FloatArray): FloatArray?

    /** 释放 native 资源 */
    private external fun nativeRelease(handle: Long)

    init {
        val detAsset = "model/ppocrv6_mnn/PP-OCRv6_${modelSize.name.lowercase()}_det.mnn"
        val recAsset = "model/ppocrv6_mnn/PP-OCRv6_${modelSize.name.lowercase()}_rec.mnn"
        val detCache = copyAssetToCache(context, detAsset)
        val recCache = copyAssetToCache(context, recAsset)

        nativeHandle = nativeInit(detCache, recCache)
        if (nativeHandle == 0L) {
            throw RuntimeException("MNN native init failed for $detAsset / $recAsset")
        }
    }

    // ── 文件管理 ────────────────────────────────────────

    private fun copyAssetToCache(context: Context, assetPath: String): String {
        val file = File(context.cacheDir, assetPath)
        if (!file.exists()) {
            file.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                FileOutputStream(file).use { output -> input.copyTo(output) }
            }
        }
        return file.absolutePath
    }

    // ── 公开方法 ────────────────────────────────────────

    fun ocr(bitmap: Bitmap, minScore: Float = 0.3f): List<OcrLine> {
        val origW = bitmap.width
        val origH = bitmap.height
        val boxes = detect(bitmap, origW, origH)
        return recognize(bitmap, origW, origH, boxes, minScore)
    }

    fun detect(bitmap: Bitmap, origW: Int, origH: Int): List<IntArray> {
        val detResult = DetPreprocessor.process(bitmap)
        val h = detResult.height
        val w = detResult.width

        val result = nativeDetect(nativeHandle, detResult.inputTensor, h, w)
            ?: return emptyList()

        val oW = result[0].toInt()
        val oH = result[1].toInt()
        val probMap = result.copyOfRange(2, result.size)

        return DBPostProcessor.process(
            probMap = probMap,
            probW = oW,
            probH = oH,
            origW = origW,
            origH = origH,
            scaleX = detResult.scaleX,
            scaleY = detResult.scaleY,
        )
    }

    fun recognize(
        bitmap: Bitmap,
        origW: Int,
        origH: Int,
        boxes: List<IntArray>,
        minScore: Float = 0.3f,
    ): List<OcrLine> {
        if (boxes.isEmpty()) return emptyList()

        val lines = mutableListOf<OcrLine>()

        for (box in boxes) {
            val x1 = box[0]; val y1 = box[1]; val x2 = box[2]; val y2 = box[3]
            val bw = x2 - x1 + 1
            val bh = y2 - y1 + 1
            if (bw < 4 || bh < 4) continue

            val cx = x1.coerceIn(0, origW - 1)
            val cy = y1.coerceIn(0, origH - 1)
            val cw = bw.coerceAtMost(origW - cx)
            val ch = bh.coerceAtMost(origH - cy)
            if (cw < 4 || ch < 4) continue

            val crop = Bitmap.createBitmap(bitmap, cx, cy, cw, ch)
            val recInput = RecPreprocessor.process(crop)

            val outData = nativeRecognize(nativeHandle, recInput.inputTensor)
                ?: continue

            // ── 从输出数据提取实际 timeSteps 和 vocabSize ──────
            // JNI 返回 [timeSteps, vocabSize, data...]
            // 这样就不会被 charDict.size 误导（TINY 模型 vocab=6906 ≠ 18709）
            val actualTimeSteps = outData[0].toInt()
            val actualVocabSize = outData[1].toInt()
            if (actualTimeSteps <= 0 || actualVocabSize <= 0) continue

            // 从 data 部分截取 [batch=1, timeSteps, vocabSize] 的 flat 数据
            val headerLen = 2
            val expectedLen = actualTimeSteps * actualVocabSize
            if (outData.size < headerLen + expectedLen) continue

            val logits = outData.copyOfRange(headerLen, headerLen + expectedLen)

            // ── 关键修正： ──────────────────────────────────
            // 模型输出 flat 数据布局为 [batch=1, timeSteps, actualVocabSize]，
            // 每个时间步占 actualVocabSize 个 float，时间步之间的 stride = actualVocabSize。
            // charDict 可能小于 actualVocabSize（TINY: dict=6905 vs model=6906），
            // 但 stride 必须用 actualVocabSize，否则后续时间步全错位。
            // CTCDecoder 内部会处理 maxIdx >= charDict.size 的越界。
            val effectiveDict = charDict.subList(0, minOf(actualVocabSize, charDict.size))

            val decoded = CTCDecoder.decode(logits, actualTimeSteps, actualVocabSize, effectiveDict)
            if (decoded.isNotEmpty()) {
                val r = decoded[0]
                if (r.score >= minScore && r.text.isNotBlank()) {
                    lines.add(OcrLine(r.text, r.score, intArrayOf(x1, y1, x2, y2)))
                }
            }
        }

        lines.sortWith(compareBy({ it.bbox[1] }, { it.bbox[0] }))
        return lines
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle)
            nativeHandle = 0L
        }
    }
}
