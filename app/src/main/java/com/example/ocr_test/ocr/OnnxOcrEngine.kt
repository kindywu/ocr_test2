package com.example.ocr_test.ocr

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * PP-OCRv6 ONNX OCR 引擎 — 对应 main2.py 的 ocr_image()
 *
 * 编排检测 → 识别流水线。
 * 所有 ONNX 模型运行在 CPU（ORT 默认），
 * 如需 GPU 加速可在 SessionOptions 中配置。
 *
 * 支持 SMALL（精度优先）和 TINY（速度优先）两种模型尺寸。
 * 模型文件位于 assets/model/ppocrv6_onnx/{small,tiny}_{det,rec}/inference.onnx
 */
class OnnxOcrEngine(context: Context, private val modelSize: ModelSize = ModelSize.SMALL) : AutoCloseable {

    enum class ModelSize { SMALL, TINY }

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

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val detSession: OrtSession
    private val recSession: OrtSession
    private val charDict: List<String>

    init {
        val sizeDir = modelSize.name.lowercase() // "small" or "tiny"
        detSession = env.createSession(
            loadModelFile(context, "model/ppocrv6_onnx/${sizeDir}_det/inference.onnx"),
            OrtSession.SessionOptions(),
        )
        recSession = env.createSession(
            loadModelFile(context, "model/ppocrv6_onnx/${sizeDir}_rec/inference.onnx"),
            OrtSession.SessionOptions(),
        )
        val ymlPath = when (modelSize) {
            ModelSize.SMALL -> "model/ppocrv6_onnx/small_rec/inference.yml"
            ModelSize.TINY -> "model/ppocrv6_onnx/tiny_rec/inference.yml"
        }
        charDict = CharDictLoader.load(context, ymlPath)
    }

    /** 从 assets 拷贝 ONNX 到缓存目录，ORT 需要文件路径 */
    private fun loadModelFile(context: Context, assetPath: String): String {
        val cacheFile = File(context.cacheDir, assetPath)
        if (!cacheFile.exists()) {
            cacheFile.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                cacheFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return cacheFile.absolutePath
    }

    /** float[] → FloatBuffer（供 OnnxTensor.createTensor 使用） */
    private fun floatArrayToBuffer(data: FloatArray): FloatBuffer = FloatBuffer.wrap(data)

    // ── 公开方法 ────────────────────────────────────────

    /**
     * 对单张图片做 OCR
     *
     * @param bitmap   输入图片（ARGB_8888）
     * @param minScore 最低置信度过滤
     * @return 按从上到下、从左到右排序的文字行
     */
    fun ocr(bitmap: Bitmap, minScore: Float = 0.3f): List<OcrLine> {
        val origW = bitmap.width
        val origH = bitmap.height

        val boxes = detect(bitmap, origW, origH)
        return recognize(bitmap, origW, origH, boxes, minScore)
    }

    /** 仅做文字检测，返回轴对齐矩形 [x1,y1,x2,y2] */
    fun detect(bitmap: Bitmap, origW: Int, origH: Int): List<IntArray> {
        val detResult = DetPreprocessor.process(bitmap)
        val detShape = longArrayOf(1, 3, detResult.height.toLong(), detResult.width.toLong())
        val inputTensor = OnnxTensor.createTensor(env, floatArrayToBuffer(detResult.inputTensor), detShape)

        val detOut = detSession.run(mapOf(detSession.inputNames.iterator().next() to inputTensor))
        val probTensor = detOut.get(0) as OnnxTensor
        val probShape = probTensor.info.shape
        val oH = probShape[2].toInt()
        val oW = probShape[3].toInt()

        val probArray = FloatArray(oH * oW)
        probTensor.floatBuffer.rewind()
        probTensor.floatBuffer.get(probArray)

        return DBPostProcessor.process(
            probMap = probArray,
            probW = oW,
            probH = oH,
            origW = origW,
            origH = origH,
            scaleX = detResult.scaleX,
            scaleY = detResult.scaleY,
        )
    }

    /** 对检测框逐一做文字识别 */
    fun recognize(
        bitmap: Bitmap,
        origW: Int,
        origH: Int,
        boxes: List<IntArray>,
        minScore: Float = 0.3f,
    ): List<OcrLine> {
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
            val recShape = longArrayOf(1, 3, 48, 320)
            val recTensor = OnnxTensor.createTensor(env, floatArrayToBuffer(recInput.inputTensor), recShape)

            val recOut = recSession.run(mapOf(recSession.inputNames.iterator().next() to recTensor))
            val recTensorOut = recOut.get(0) as OnnxTensor
            val shape = recTensorOut.info.shape
            val tSteps = shape[1].toInt()
            val vSize = shape[2].toInt()

            val outData = FloatArray(tSteps * vSize)
            recTensorOut.floatBuffer.rewind()
            recTensorOut.floatBuffer.get(outData)

            val decoded = CTCDecoder.decode(outData, tSteps, vSize, charDict)
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
        detSession.close()
        recSession.close()
    }
}
