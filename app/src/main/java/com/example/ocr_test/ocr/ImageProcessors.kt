package com.example.ocr_test.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 检测模型预处理 — 对应 main2.py 的 det_resize() + det_preprocess()
 */
object DetPreprocessor {

    private const val LIMIT_SIDE = 960
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class DetResult(
        val inputTensor: FloatArray,  // shape [1, 3, H, W]
        val width: Int,               // 缩放后的宽（32 对齐）
        val height: Int,              // 缩放后的高（32 对齐）
        val scaleX: Float,            // 原图宽 → 缩放图宽 的比例
        val scaleY: Float,            // 原图高 → 缩放图高 的比例
    )

    /**
     * DetResizeForTest — limit_type='max', 保证 32 对齐
     */
    private fun resize(bitmap: Bitmap): Pair<Bitmap, Pair<Float, Float>> {
        val (h, w) = bitmap.height to bitmap.width
        var ratio = 1.0f

        if (max(h, w) > LIMIT_SIDE) {
            ratio = if (h > w) LIMIT_SIDE.toFloat() / h else LIMIT_SIDE.toFloat() / w
        }

        val newH = max((h * ratio / 32f).roundToInt() * 32, 32)
        val newW = max((w * ratio / 32f).roundToInt() * 32, 32)

        val scaleH = newH.toFloat() / h
        val scaleW = newW.toFloat() / w

        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        return resized to (scaleW to scaleH)
    }

    /**
     * 对检测图做完整预处理
     * 输出: NCHW float 数组，归一化到 [0,1] 后再减均值除方差
     */
    fun process(bitmap: Bitmap): DetResult {
        val pair = resize(bitmap)
        val resized = pair.first
        val scaleW = pair.second.first
        val scaleH = pair.second.second
        val w = resized.width
        val h = resized.height

        val pixels = IntArray(w * h)
        resized.getPixels(pixels, 0, w, 0, 0, w, h)

        // NCHW layout: [batch=1, channel=3, H, W]
        val tensor = FloatArray(1 * 3 * h * w)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val pixel = pixels[y * w + x]
                // Android Bitmap 是 ARGB, 需要转 BGR
                val b = (pixel and 0xff).toFloat() / 255f
                val g = ((pixel shr 8) and 0xff).toFloat() / 255f
                val r = ((pixel shr 16) and 0xff).toFloat() / 255f

                val idx = y * w + x
                // channel 0 (B)
                tensor[0 * h * w + idx] = (b - MEAN[0]) / STD[0]
                // channel 1 (G)
                tensor[1 * h * w + idx] = (g - MEAN[1]) / STD[1]
                // channel 2 (R)
                tensor[2 * h * w + idx] = (r - MEAN[2]) / STD[2]
            }
        }

        return DetResult(tensor, w, h, scaleW, scaleH)
    }
}

/**
 * 识别模型预处理 — 对应 main2.py 的 rec_preprocess()
 * 缩放到高 48，宽等比缩放 + 补齐到 320
 */
object RecPreprocessor {

    private const val TARGET_H = 48
    private const val TARGET_W = 320

    data class RecResult(
        val inputTensor: FloatArray,  // shape [1, 3, 48, 320]
    )

    fun process(bitmap: Bitmap): RecResult {
        val srcH = bitmap.height
        val srcW = bitmap.width

        val ratio = TARGET_H.toFloat() / srcH
        val resizeW = (srcW * ratio).toInt()    // truncate — 匹配 Python int()
        val padded: Bitmap

        if (resizeW > TARGET_W) {
            // 文字太宽 → 按宽缩放，补高度
            val r2 = TARGET_W.toFloat() / srcW
            val resizeH = max((srcH * r2).toInt(), 4)  // truncate — 匹配 Python int()
            val scaled = Bitmap.createScaledBitmap(bitmap, TARGET_W, resizeH, true)
            padded = Bitmap.createBitmap(TARGET_W, TARGET_H, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(padded)
            canvas.drawBitmap(scaled, 0f, 0f, null)
        } else {
            // 按高缩放，补宽度
            val rw = max(resizeW, 4)
            val scaled = Bitmap.createScaledBitmap(bitmap, rw, TARGET_H, true)
            padded = Bitmap.createBitmap(TARGET_W, TARGET_H, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(padded)
            canvas.drawBitmap(scaled, 0f, 0f, null)
        }

        val w = padded.width
        val h = padded.height
        val pixels = IntArray(w * h)
        padded.getPixels(pixels, 0, w, 0, 0, w, h)

        // 识别模型: 仅缩放到 [0,1]，不需要 mean/std 归一化
        // (inference.yml 无 NormalizeImage)
        val tensor = FloatArray(1 * 3 * h * w)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val pixel = pixels[y * w + x]
                // BGR order
                val b = (pixel and 0xff).toFloat() / 255f
                val g = ((pixel shr 8) and 0xff).toFloat() / 255f
                val r = ((pixel shr 16) and 0xff).toFloat() / 255f

                val idx = y * w + x
                tensor[0 * h * w + idx] = b
                tensor[1 * h * w + idx] = g
                tensor[2 * h * w + idx] = r
            }
        }

        return RecResult(tensor)
    }
}
