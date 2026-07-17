package com.example.ocr_test.ocr

/**
 * CTC Greedy 解码 — 对应 main2.py 的 ctc_decode()
 *
 * 流程: argmax → 去重 → 去 blank → 字符映射
 */
object CTCDecoder {

    data class DecodeResult(
        val text: String,
        val score: Float,
    )

    /**
     * @param output     模型输出 (N, T, vocabSize) — N=1 通常是
     * @param timeSteps  T: 时间步数
     * @param vocabSize  vocab 大小
     * @param charDict   字符表 (index 0 = blank)
     * @return 解码结果列表 (N 条)
     */
    fun decode(
        output: FloatArray,
        timeSteps: Int,
        vocabSize: Int,
        charDict: List<String>,
    ): List<DecodeResult> {
        val batchSize = output.size / (timeSteps * vocabSize)
        val results = mutableListOf<DecodeResult>()

        for (b in 0 until batchSize) {
            val chars = mutableListOf<String>()
            val probs = mutableListOf<Float>()
            var prevIdx = -1

            for (t in 0 until timeSteps) {
                val offset = b * (timeSteps * vocabSize) + t * vocabSize

                // argmax
                var maxIdx = 0
                var maxProb = output[offset]
                for (c in 1 until vocabSize) {
                    val p = output[offset + c]
                    if (p > maxProb) {
                        maxProb = p
                        maxIdx = c
                    }
                }

                if (maxIdx == 0) {
                    // blank, 跳过
                    prevIdx = -1
                    continue
                }

                if (maxIdx != prevIdx) {
                    // 去重
                    if (maxIdx < charDict.size) {
                        chars.add(charDict[maxIdx])
                        probs.add(maxProb)
                    }
                    prevIdx = maxIdx
                }
            }

            val text = chars.joinToString("")
            val avgProb = if (probs.isNotEmpty()) {
                probs.sum() / probs.size
            } else {
                0f
            }

            results.add(DecodeResult(text, avgProb))
        }

        return results
    }
}
