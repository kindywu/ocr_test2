package com.example.ocr_test.ocr

import android.content.Context

/**
 * 从 rec/inference.yml 加载 character_dict（模型实际使用的字符集）
 *
 * 与 main2.py 的 load_char_dict() 对应：
 * - 读取 YAML 中 PostProcess.character_dict 列表
 * - index 0 保留给 CTC blank
 */
object CharDictLoader {

    /**
     * @param assetPath YAML 文件在 assets 中的路径，默认 model/rec/inference.yml
     */
    fun load(context: Context, assetPath: String = "model/rec/inference.yml"): List<String> {
        val yaml = context.assets.open(assetPath)
            .bufferedReader()
            .readText()
        return parseCharacterDict(yaml)
    }

    private fun parseCharacterDict(yaml: String): List<String> {
        val chars = mutableListOf("") // index 0 = CTC blank
        var inDict = false

        for (line in yaml.lines()) {
            val trimmed = line.trimStart()

            if (trimmed == "character_dict:") {
                inDict = true
                continue
            }

            if (!inDict) continue

            // 一旦遇到非 "- " 开头的行，说明 character_dict 结束了
            if (!trimmed.startsWith("- ")) break

            val raw = trimmed.removePrefix("- ").trim()

            // 去掉单引号或双引号
            val ch = when {
                raw.startsWith("'") && raw.endsWith("'") && raw.length >= 2 ->
                    raw.substring(1, raw.length - 1)
                raw.startsWith("\"") && raw.endsWith("\"") && raw.length >= 2 ->
                    raw.substring(1, raw.length - 1)
                else -> raw
            }

            chars.add(ch)
        }

        return chars
    }
}
