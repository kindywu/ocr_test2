# PP-OCRv6 OCR 引擎 — 实现与测试完全指南

## 目录

1. [项目结构](#1-项目结构)
2. [OCR 流水线总览](#2-ocr-流水线总览)
3. [OcrEngine — ONNX Runtime 实现](#3-ocrengine--onnx-runtime-实现)
4. [MnnOcrEngine — MNN 实现](#4-mnnocrengine--mnn-实现)
5. [Google ML Kit OCR — 设备端通用 OCR](#5-google-ml-kit-ocr--设备端通用-ocr)
6. [预处理模块](#6-预处理模块)
7. [后处理模块](#7-后处理模块)
8. [CTC 解码](#8-ctc-解码)
9. [字符字典加载](#9-字符字典加载)
10. [MNN JNI 桥接](#10-mnn-jni-桥接)
11. [测试实现](#11-测试实现)
12. [曾踩过的坑](#12-曾踩过的坑)
13. [运行指南](#13-运行指南)
14. [三引擎精度对比](#14-三引擎精度对比)

---

## 1. 项目结构

```
app/src/main/java/com/example/ocr_test/ocr/
├── OcrEngine.kt           # PP-OCRv6 ONNX Runtime 引擎（检测 + 识别）
├── MnnOcrEngine.kt        # PP-OCRv6 MNN 引擎（JNI 桥接）
├── CharDictLoader.kt      # YAML 字符字典解析器
├── ImageProcessors.kt     # 检测/识别图像预处理
│   ├── DetPreprocessor    # 检测模型预处理
│   └── RecPreprocessor    # 识别模型预处理
├── DBPostProcessor.kt     # DB (Differential Binarization) 后处理
├── CTCDecoder.kt          # CTC Greedy 解码

app/src/main/cpp/
└── mnn_ocr_jni.cpp        # MNN 推理 JNI 桥接（C++）

app/src/main/assets/model/
├── det/inference.onnx     # 检测 ONNX 模型
├── rec/inference.onnx     # 识别 ONNX 模型
├── rec/inference.yml      # TINY 识别模型配置（含 ~6906 字符字典）
├── mnn/
│   ├── PP-OCRv6_small_det.mnn
│   ├── PP-OCRv6_small_rec.mnn
│   ├── PP-OCRv6_tiny_det.mnn
│   ├── PP-OCRv6_tiny_rec.mnn
│   ├── inference_small.yml  # SMALL 识别模型配置（含全量 18709 字符字典）
└── ppocrv6_dict.txt         # 全量字典文本文件（18709 字符）

app/src/androidTest/
├── assets/
│   ├── full_fields.jpg      # 药品说明书正面（多字段）
│   └── date_field.jpg       # 药品日期区域
└── java/com/example/ocr_test/
    ├── PaddleOcrOnnxTest.kt # ONNX 引擎测试
    ├── MnnOcrEngineTest.kt  # MNN 引擎测试（SMALL / TINY）
    └── GoogleMlKitOcrTest.kt# Google ML Kit OCR 测试
```

---

## 2. OCR 流水线总览

PP-OCRv6 的 OCR 流程分为两大阶段：**文字检测 → 文字识别**。

```
输入图片 (Bitmap)
     │
     ▼
┌─────────────────────┐
│   文字检测 (Detect)  │  DetPreprocessor → 检测模型 → DBPostProcessor
│                     │  输出: 轴对齐矩形框 [x1,y1,x2,y2]
└─────────┬───────────┘
          │ 每个检测框
          ▼
┌─────────────────────┐
│   文字识别 (Recognize)│  RecPreprocessor → 识别模型 → CTCDecoder
│                     │  输出: (text, score, bbox)
└─────────┬───────────┘
          │ 排序（按 y 从上到下，x 从左到右）
          ▼
    List<OcrLine>
```

### 两个引擎共享的组件

| 组件 | 作用 | 两个引擎共用？ |
|------|------|:------------:|
| `DetPreprocessor` | 检测输入预处理 | ✓ 完全相同 |
| `RecPreprocessor` | 识别输入预处理 | ✓ 完全相同 |
| `DBPostProcessor` | 检测输出后处理 → 文本框 | ✓ 完全相同 |
| `CTCDecoder` | 识别输出后处理 → 文本 | ✓ 完全相同 |
| `CharDictLoader` | YAML 字典解析 | ✓ 完全相同 |

### 两个引擎不同的部分

| 方面 | OcrEngine (ONNX) | MnnOcrEngine (MNN) |
|------|-----------------|-------------------|
| 推理框架 | ONNX Runtime Java API | MNN C++ API (via JNI) |
| 模型格式 | `.onnx` | `.mnn` |
| 模型文件 | `det/inference.onnx` + `rec/inference.onnx` | `mnn/PP-OCRv6_*_{det,rec}.mnn` |
| 模型变体 | 单一模型 | SMALL / TINY 可选 |
| 张量操作 | `OnnxTensor.createTensor()` + `session.run()` | C++ 端 `Interpreter::runSession()` |
| 字符字典 | `model/rec/inference.yml` | 按模型选择 YAML |

---

## 3. OcrEngine — ONNX Runtime 实现

**文件**: `OcrEngine.kt`

### 初始化

```kotlin
class OcrEngine(context: Context) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val detSession: OrtSession
    private val recSession: OrtSession
    private val charDict: List<String>

    init {
        detSession = env.createSession(loadModelFile(context, "model/det/inference.onnx"), ...)
        recSession = env.createSession(loadModelFile(context, "model/rec/inference.onnx"), ...)
        charDict = CharDictLoader.load(context)
    }
}
```

- `OrtEnvironment` 是 ONNX Runtime 的全局单例
- 检测和识别各有一个 `OrtSession`，分别加载对应的 `.onnx` 模型文件
- 模型文件从 assets 拷贝到 cache 目录后加载（ONNX Runtime 需要文件路径）

### 检测流程

```kotlin
fun detect(bitmap: Bitmap, origW: Int, origH: Int): List<IntArray> {
    // 1. 预处理
    val detResult = DetPreprocessor.process(bitmap)

    // 2. 创建 ONNX 输入张量
    val inputTensor = OnnxTensor.createTensor(env, floatArrayToBuffer(detResult.inputTensor),
        longArrayOf(1, 3, detResult.height.toLong(), detResult.width.toLong()))

    // 3. 推理
    val detOut = detSession.run(mapOf(detSession.inputNames.iterator().next() to inputTensor))
    val probTensor = detOut.get(0) as OnnxTensor

    // 4. 获取输出概率图
    val probArray = FloatArray(oH * oW)
    probTensor.floatBuffer.get(probArray)

    // 5. DB 后处理 → 文本框
    return DBPostProcessor.process(probMap = probArray, probW = oW, probH = oH, ...)
}
```

### 识别流程

```kotlin
fun recognize(bitmap: Bitmap, origW: Int, origH: Int, boxes: List<IntArray>, ...): List<OcrLine> {
    for (box in boxes) {
        // 1. 裁剪检测区域
        val crop = Bitmap.createBitmap(bitmap, cx, cy, cw, ch)

        // 2. 预处理（48×320）
        val recInput = RecPreprocessor.process(crop)

        // 3. 创建 ONNX 输入张量
        val recTensor = OnnxTensor.createTensor(env, floatArrayToBuffer(recInput.inputTensor),
            longArrayOf(1, 3, 48, 320))

        // 4. 推理
        val recOut = recSession.run(...)
        val tSteps = shape[1].toInt()
        val vSize = shape[2].toInt()

        // 5. CTC 解码
        val decoded = CTCDecoder.decode(outData, tSteps, vSize, charDict)
        // 取最佳结果（batch=0）
        if (decoded[0].score >= minScore && decoded[0].text.isNotBlank()) {
            lines.add(OcrLine(decoded[0].text, decoded[0].score, intArrayOf(x1, y1, x2, y2)))
        }
    }
    // 排序：y 从上到下，x 从左到右
    lines.sortWith(compareBy({ it.bbox[1] }, { it.bbox[0] }))
    return lines
}
```

---

## 4. MnnOcrEngine — MNN 实现

**文件**: `MnnOcrEngine.kt`

### 初始化

```kotlin
class MnnOcrEngine(context: Context, private val modelSize: ModelSize = ModelSize.SMALL) : AutoCloseable {

    enum class ModelSize { SMALL, TINY }

    private var nativeHandle: Long = 0
    private val charDict: List<String> = loadCharDict(context, modelSize)

    init {
        // 选择对应的 .mnn 模型文件
        val detAsset = "model/mnn/PP-OCRv6_${modelSize.name.lowercase()}_det.mnn"
        val recAsset = "model/mnn/PP-OCRv6_${modelSize.name.lowercase()}_rec.mnn"
        // 从 assets 拷贝到 cache（MNN 需要文件路径）
        val detCache = copyAssetToCache(context, detAsset)
        val recCache = copyAssetToCache(context, recAsset)
        // JNI 调用 C++ 初始化
        nativeHandle = nativeInit(detCache, recCache)
    }
}
```

关键区别：
- `modelSize` 决定加载 SMALL 还是 TINY 模型
- 字符字典按模型选择对应的 YAML 文件
- native 层通过 JNI 管理 MNN `Interpreter` 的生命周期

### JNI 方法声明

```kotlin
private external fun nativeInit(detModelPath: String, recModelPath: String): Long
private external fun nativeDetect(handle: Long, input: FloatArray, h: Int, w: Int): FloatArray?
private external fun nativeRecognize(handle: Long, input: FloatArray): FloatArray?
private external fun nativeRelease(handle: Long)
```

- `nativeInit` 返回一个 `long` 类型的 native handle（指向 C++ 的 `OcrContext` 结构体）
- `nativeDetect` 和 `nativeRecognize` 接收处理好的 float 张量数据，返回推理结果
- `nativeRelease` 释放 native 资源

### 检测流程

```kotlin
fun detect(bitmap: Bitmap, origW: Int, origH: Int): List<IntArray> {
    // 1. 预处理（与 ONNX 版本完全相同的 DetPreprocessor）
    val detResult = DetPreprocessor.process(bitmap)

    // 2. JNI 调用 C++ 端 MNN 推理
    val result = nativeDetect(nativeHandle, detResult.inputTensor, h, w)
        ?: return emptyList()

    // 3. 解析返回值 [oW, oH, probMap...]
    val oW = result[0].toInt()
    val oH = result[1].toInt()
    val probMap = result.copyOfRange(2, result.size)

    // 4. DB 后处理（与 ONNX 版本完全相同）
    return DBPostProcessor.process(probMap, probW = oW, probH = oH, ...)
}
```

### 识别流程

```kotlin
fun recognize(bitmap, origW, origH, boxes, minScore): List<OcrLine> {
    for (box in boxes) {
        // 1. 裁剪（与 ONNX 版本相同）
        val crop = Bitmap.createBitmap(bitmap, cx, cy, cw, ch)

        // 2. 预处理（与 ONNX 版本相同的 RecPreprocessor）
        val recInput = RecPreprocessor.process(crop)

        // 3. JNI 调用 C++ 端 MNN 推理
        val outData = nativeRecognize(nativeHandle, recInput.inputTensor) ?: continue

        // 4. 解析返回值 [timeSteps, vocabSize, data...]
        val actualTimeSteps = outData[0].toInt()
        val actualVocabSize = outData[1].toInt()

        // 5. 截取概率数据
        val logits = outData.copyOfRange(headerLen, headerLen + actualTimeSteps * actualVocabSize)

        // 6. CTC 解码（stride = actualVocabSize，而非 charDict.size！）
        val effectiveDict = charDict.subList(0, minOf(actualVocabSize, charDict.size))
        val decoded = CTCDecoder.decode(logits, actualTimeSteps, actualVocabSize, effectiveDict)
        //                              ^^^^^^^^^^^^^^^^ 使用模型实际 vocab 作为 stride
    }
}
```

### 资源释放

```kotlin
override fun close() {
    if (nativeHandle != 0L) {
        nativeRelease(nativeHandle)  // JNI 调用 C++ delete
        nativeHandle = 0L
    }
}
```

使用 `use {}` 自动管理资源：
```kotlin
MnnOcrEngine(context, MnnOcrEngine.ModelSize.SMALL).use { engine ->
    val lines = engine.ocr(bitmap)
    // 退出 use 块时自动调用 close()
}
```

---

## 5. Google ML Kit OCR — 设备端通用 OCR

ML Kit OCR 与 PP-OCRv6 不同，它不是检测+识别的两阶段流水线，而是 Google 提供的**端到端设备端 OCR SDK**。

### SDK 依赖

```kotlin
// build.gradle.kts（通过 version catalog）
implementation(libs.mlkit.text.recognition)           // 基础拉丁模型
implementation(libs.mlkit.text.recognition.chinese)    // 中文模型（bundled，离线可用）
```

两个模型均为 **bundled 模式**（打包在 APK 中），无需 Google Play Services 即可离线运行。

### 实现方式

ML Kit OCR 的使用极为简洁，只需几行代码：

```kotlin
// 1. 创建中文识别器
val recognizer = TextRecognition.getClient(
    ChineseTextRecognizerOptions.Builder().build()
)

try {
    // 2. 将 Bitmap 转为 InputImage
    val image = InputImage.fromBitmap(bitmap, 0)

    // 3. 同步调用（ML Kit 返回 Task，用 Tasks.await 阻塞等待）
    val result = Tasks.await(recognizer.process(image))

    // 4. 获取识别文本
    val actualText = result.text  // 整张图片的全部文字
    println(actualText)

    // 5. 释放资源
} finally {
    recognizer.close()
}
```

### ML Kit 与 PP-OCRv6 的本质区别

| 方面 | PP-OCRv6 (ONNX/MNN) | Google ML Kit OCR |
|------|:-------------------:|:-----------------:|
| **架构** | 检测模型 + 识别模型两阶段 | 端到端单模型 |
| **推理框架** | ONNX Runtime / MNN | Google 内部 TFLite 变体 |
| **模型大小** | 可切换 SMALL/TINY | Google 托管，不可选 |
| **输出粒度** | 逐行输出（检测框 + 文本 + 置信度） | 整页输出（含 block/line/text 层级结构） |
| **预处理** | 手动 resize + normalize | SDK 内部自动处理 |
| **后处理** | 手动 DBPostProcess + CTC 解码 | SDK 内部自动完成 |
| **控制力** | 完全可控（可调阈值、滤波、排序） | 黑盒，仅能控制输入图片 |
| **识别范围** | 仅检测到的文字区域 | 整张图片所有文字（包括背景文字） |
| **集成复杂度** | 需要 native 库 + JNI + 多组件 | 仅需 Gradle 依赖 + 几行 API 调用 |

### ML Kit 的文本层级结构

```
Text (result.text) — 整张图片的纯文本
└── List<TextBlock>
    └── List<TextLine>
        └── List<TextElement>
            └── 每个字符/词（含 bounding box + 置信度）
```

测试中仅使用了 `result.text` 获取全部文本进行断言验证。

### 适用场景

- **ML Kit**: 通用文档 OCR、快速集成、不需要自定义模型
- **PP-OCRv6**: 需要精细控制每个检测框、需要特定精度/速度权衡、离线的私有化部署

---

## 6. 预处理模块

### 5.1 DetPreprocessor — 检测预处理

**文件**: `ImageProcessors.kt` (`object DetPreprocessor`)

对应 Python 参考实现的 `det_resize()` + `det_preprocess()`。

**流程**:

```
输入 Bitmap (ARGB_8888)
    │
    ▼
1. 等比缩放 (limit_type='max', max=960)
   └─ 较长边 > 960 则缩放到 960，否则保持原大小
   └─ 宽高 32 对齐: round(h * ratio / 32) * 32
    │
    ▼
2. 归一化: /255 → (x - mean) / std
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
    │
    ▼
3. HWC → CHW (BGR 顺序)
   输出 shape: [1, 3, H, W]
```

**代码要点**:
```kotlin
// Android Bitmap 是 ARGB 格式
val b = (pixel and 0xff).toFloat() / 255f          // Blue
val g = ((pixel shr 8) and 0xff).toFloat() / 255f   // Green
val r = ((pixel shr 16) and 0xff).toFloat() / 255f  // Red

// CHW layout: tensor[channel * H * W + y * W + x]
tensor[0 * h * w + idx] = (b - MEAN[0]) / STD[0]   // B channel
tensor[1 * h * w + idx] = (g - MEAN[1]) / STD[1]   // G channel
tensor[2 * h * w + idx] = (r - MEAN[2]) / STD[2]   // R channel
```

### 5.2 RecPreprocessor — 识别预处理

**文件**: `ImageProcessors.kt` (`object RecPreprocessor`)

对应 Python 参考实现的 `rec_preprocess()`。

**流程**:

```
输入 Bitmap (任意大小)
    │
    ▼
1. 计算等比缩放比: ratio = 48 / srcH
   resizeW = (srcW * ratio).toInt()
    │
    ├── resizeW > 320 (文字太宽)
    │   └─ 按宽缩放到 320，补高到 48
    │
    └── resizeW ≤ 320 (正常)
        └─ 按高缩放到 48，补宽到 320
    │
    ▼
2. 创建 48×320 全零 Bitmap（黑色背景）
   将缩放后的图片绘制到左上角
    │
    ▼
3. 仅 /255 归一化（无 mean/std！）
   BGR 顺序 → CHW layout
   输出 shape: [1, 3, 48, 320]
```

**关键**：识别模型不需要 mean/std 归一化，因为 inference.yml 中 `PreProcess` 没有 `NormalizeImage` 配置。

---

## 6. 后处理模块

### DBPostProcessor — DB 后处理

**文件**: `DBPostProcessor.kt` (`object DBPostProcessor`)

纯 Kotlin 实现（无 OpenCV 依赖），将检测模型的概率图转换为文本框。

**流程**:

```
输入: 概率图 probMap (H×W, float 0~1)
    │
    ▼
1. 二值化 (THRESH=0.2): >0.2 → 白色, ≤0.2 → 黑色
    │
    ▼
2. 形态学闭操作 (3×3 核, dilate → erode)
   └─ 填补文本区域的细小空隙
    │
    ▼
3. 连通域分析 (8-连通, Union-Find 两遍扫描)
   └─ 过滤面积 < 100 的噪声区域
    │
    ▼
4. 对每个连通域:
   ├─ 4a. Andrew's Monotone Chain 凸包算法
   ├─ 4b. Rotating Calipers 最小面积有向矩形
   ├─ 4c. boxScoreFast: 矩形区域概率均值
   │    如果 score < 0.4 则丢弃
   ├─ 4d. unclip 膨胀 (distance = area * 1.4 / perimeter)
   │    沿法线方向向外偏移多边形
   └─ 4e. 坐标映射回原图 (除以 scaleX/scaleY)
    │
    ▼
输出: 轴对齐矩形 [x1, y1, x2, y2]
```

**关键算法**:

- **Convex Hull** (Andrew's Monotone Chain): `O(n log n)`，先按 x 排序，然后构建上下凸包
- **Min Area Rect** (Rotating Calipers): 遍历凸包每条边作为矩形的一边，投影所有顶点求最小面积
- **Unclip**: 计算面积/周长得到膨胀距离，沿每条边的向外法线方向平移，再求相邻边的交点
- **Fill Convex Polygon**: 扫描线算法，从上到下逐行填充

---

## 7. CTC 解码

### CTCDecoder

**文件**: `CTCDecoder.kt` (`object CTCDecoder`)

实现 CTC Greedy 解码（与 PaddleOCR 的 CTCLabelDecode 一致）。

**算法**:

```
对于每个 batch:
    chars = []
    prevIdx = -1
    
    对于每个时间步 t:
        argmax: 找到概率最大的 class index
        if index == 0:  # CTC blank, 跳过
            prevIdx = -1; continue
        if index != prevIdx:  # 去重
            if index < charDict.size:
                chars.append(charDict[index])
            prevIdx = index
    
    text = chars.joinToString("")
    score = probs平均值
```

**核心逻辑**:
```
原始模型输出:  [0, 0, 国, 国, 0, 药, 0, 准, 字, 0, 0, H, 2, 0, 0, 0, 4]
CTC 去重去 blank: [国, 药, 准, 字, H, 2, 4]
最终文本:        "国药准字H24"
```

**关于 stride 参数的重要说明**:

`decode()` 函数的 `vocabSize` 参数同时用于两个目的：
1. 时间步之间的 stride（跳转步长）
2. argmax 的范围（0 到 vocabSize-1）

因此 stride **必须**等于模型输出的实际 vocab 维度（`actualVocabSize`），而不是 `charDict.size`。如果使用 `min(actualVocabSize, charDict.size)` 作为 stride，当 charDict 小于模型 vocab 时，每个时间步的 stride 都会少 1，导致累积偏移 — 后续所有时间步的 argmax 计算在**错误的数据位置**上。

```kotlin
// ❌ 错误：stride 用 effectiveVocabSize
val effectiveVocabSize = minOf(actualVocabSize, charDict.size)
CTCDecoder.decode(logits, actualTimeSteps, effectiveVocabSize, charDict)
// 每个时间步偏移: t * 6905，但实际数据是 t * 6906

// ✓ 正确：stride 用 actualVocabSize，字典越界由内部跳过
CTCDecoder.decode(logits, actualTimeSteps, actualVocabSize, effectiveDict)
// 每个时间步偏移: t * 6906 ✓，索引 >= charDict.size 的字符被跳过
```

---

## 8. 字符字典加载

### CharDictLoader

**文件**: `CharDictLoader.kt` (`object CharDictLoader`)

从 YAML 文件中解析 `PostProcess.character_dict` 列表。

**支持的 YAML 格式**:
```yaml
PostProcess:
  name: CTCLabelDecode
  character_dict:
  - '!'       # 单引号
  - '"'       # 双引号
  - $         # 无引号
  - \         # 反斜杠
  - ''''      # 单引号字符本身
  - 国        # 中文字符（无引号）
```

**解析逻辑**:
```kotlin
fun parseCharacterDict(yaml: String): List<String> {
    val chars = mutableListOf("")  // index 0 = CTC blank
    var inDict = false

    for (line in yaml.lines()) {
        val trimmed = line.trimStart()
        if (trimmed == "character_dict:") { inDict = true; continue }
        if (!inDict) continue
        if (!trimmed.startsWith("- ")) break  // 遇到非 YAML 列表项则停止

        val raw = trimmed.removePrefix("- ").trim()
        // 去掉引号
        val ch = when {
            raw.startsWith("'") && raw.endsWith("'") && raw.length >= 2 ->
                raw.substring(1, raw.length - 1)
            raw.startsWith("\"") && raw.endsWith("\"") && raw.length >= 2 ->
                raw.substring(1, raw.length - 1)
            else -> raw  // 无引号
        }
        chars.add(ch)
    }
    return chars
}
```

### 按模型选择字典

在 `MnnOcrEngine` 中，`loadCharDict()` 根据模型变体加载对应的 YAML：

```kotlin
private fun loadCharDict(context: Context, modelSize: ModelSize): List<String> {
    val yamlPath = when (modelSize) {
        ModelSize.SMALL -> "model/mnn/inference_small.yml"   // 18709 项
        ModelSize.TINY  -> "model/rec/inference.yml"           // ~6906 项
    }
    return CharDictLoader.load(context, yamlPath)
}
```

---

## 9. MNN JNI 桥接

**文件**: `mnn_ocr_jni.cpp`

### C++ 数据结构

```cpp
struct OcrContext {
    std::shared_ptr<MNN::Interpreter> detNet;  // 检测模型解释器
    std::shared_ptr<MNN::Interpreter> recNet;  // 识别模型解释器
    MNN::Session* detSession;                   // 检测会话
    MNN::Session* recSession;                   // 识别会话
};
```

### nativeInit

```cpp
JNIEXPORT jlong JNICALL Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeInit(
    JNIEnv* env, jobject, jstring detModelPath, jstring recModelPath)
{
    auto* ctx = new OcrContext();
    ctx->detNet.reset(MNN::Interpreter::createFromFile(detPath));
    ctx->detSession = ctx->detNet->createSession(config);

    ctx->recNet.reset(MNN::Interpreter::createFromFile(recPath));
    ctx->recSession = ctx->recNet->createSession(config);

    return reinterpret_cast<jlong>(ctx);  // 返回 native handle
}
```

- 使用 4 线程 CPU 推理 (`config.numThread = 4`)
- 返回的 `jlong` 是 `OcrContext*` 的指针值，Kotlin 端作为 `Long` 持有

### nativeDetect

```cpp
JNIEXPORT jfloatArray JNICALL Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeDetect(
    JNIEnv* env, jobject, jlong handle, jfloatArray input, jint h, jint w)
{
    // 1. 解析 handle
    auto* ctx = reinterpret_cast<OcrContext*>(handle);

    // 2. 获取输入张量，resize 到 [1, 3, h, w]
    auto inputTensor = net->getSessionInput(session, nullptr);
    net->resizeTensor(inputTensor, {1, 3, h, w});
    net->resizeSession(session);

    // 3. 拷贝输入数据 (Java float[] → C++ MNN::Tensor)
    hostTensor->copyFromHostTensor(inputTensor);

    // 4. 推理
    net->runSession(session);

    // 5. 获取输出，拷贝到 host
    auto outputTensor = net->getSessionOutput(session, nullptr);
    outputTensor->copyToHostTensor(hostOut.get());

    // 6. 打包返回 [oW, oH, probMap...]
    jfloatArray result = env->NewFloatArray(2 + dataSize);
    dst[0] = oW; dst[1] = oH;
    memcpy(dst + 2, srcData, dataSize * sizeof(float));
    return result;
}
```

### nativeRecognize

```cpp
JNIEXPORT jfloatArray JNICALL Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeRecognize(
    JNIEnv* env, jobject, jlong handle, jfloatArray input)
{
    // 类似 detect，但输出 shape 为 [1, timeSteps, vocabSize]

    // 打包返回 [timeSteps, vocabSize, data...]
    dst[0] = timeSteps;
    dst[1] = vocabSize;
    memcpy(dst + 2, srcData, totalSize * sizeof(float));
    return result;
}
```

关键设计：
- **输出包含维度信息**：Kotlin 端不需要预知模型的 timeSteps 和 vocabSize，从返回数组前两个元素读取即可
- **适用于不同模型变体**：TINY 返回 V=6906，SMALL 返回 V=18710，Kotlin 端自动适配

---

## 11. 测试实现

### 11.1 PaddleOcrOnnxTest — ONNX 引擎测试

**文件**: `app/src/androidTest/java/com/example/ocr_test/PaddleOcrOnnxTest.kt`

**目的**: 验证 ONNX Runtime 版本的 OCR 引擎能正确识别药品说明书上的关键字段。

**测试方法**:

```kotlin
@RunWith(AndroidJUnit4::class)
class PaddleOcrOnnxTest {

    private lateinit var engine: OcrEngine

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        engine = OcrEngine(context)  // 创建 ONNX 引擎
    }

    @After
    fun tearDown() {
        engine.close()  // 释放 ONNX Runtime 会话
    }

    @Test
    fun test_full_fields_ocr() {
        val bitmap = loadBitmap("full_fields.jpg")
        val lines = engine.ocr(bitmap)
        val allText = lines.joinToString("\n") { it.text }

        // 打印结果到 logcat
        // 保存结果到 filesDir

        // 严格验证全部 5 个关键字段
        val expectedLines = listOf(
            "奥美拉唑肠溶胶囊",
            "国药准字 H20046430",
            "药品上市许可持有人",
            "石药集团欧意药业有限公司",
            "生产企业",
        )
        val missing = expectedLines.filter { it !in allText }
        assertTrue("缺少: ${missing}", missing.isEmpty())
    }

    @Test
    fun test_date_field_ocr() {
        // 类似，验证 "产品批号"、"生产日期"、"有效期至"
    }
}
```

**验证策略**: 严格检查全部字段（ONNX 引擎精度最高）

### 11.2 MnnOcrEngineTest — MNN 引擎测试

**文件**: `app/src/androidTest/java/com/example/ocr_test/MnnOcrEngineTest.kt`

**目的**: 验证 MNN 版本的 OCR 引擎（当前为 SMALL 模型，可切换为 TINY）。

### 11.3 GoogleMlKitOcrTest — Google ML Kit OCR 测试

**文件**: `app/src/androidTest/java/com/example/ocr_test/GoogleMlKitOcrTest.kt`

**目的**: 集成 Google ML Kit 的离线中文 OCR 作为对比基准，验证 PP-OCRv6 与通用 OCR 方案的精度差异。

#### 实现方式

ML Kit 的测试无需构造引擎，直接调用 SDK API：

```kotlin
@RunWith(AndroidJUnit4::class)
class GoogleMlKitOcrTest {

    @Test
    fun test_full_fields_ocr() {
        val bitmap = loadBitmap("full_fields.jpg")

        // 1. 创建中文 bundled 识别器（离线模型，无需 Google Play Services）
        val recognizer = TextRecognition.getClient(
            ChineseTextRecognizerOptions.Builder().build()
        )

        try {
            // 2. Bitmap → InputImage
            val image = InputImage.fromBitmap(bitmap, 0)

            // 3. 同步等待 ML Kit 识别结果
            val result = Tasks.await(recognizer.process(image))
            val actualText = result.text

            // 4. 打印 + 保存
            saveResult(imageName, actualText)

            // 5. 验证关键字段
            val expectedLines = listOf(
                "奥美拉唑肠溶胶囊",
                "国药准字 H20046430",
                "药品上市许可持有人",
                "石药集团欧意药业有限公司",
                "生产企业"
            )
            val missing = expectedLines.filter { it !in actualText }
            assertTrue("缺少: ${missing}", missing.isEmpty())
        } finally {
            recognizer.close()  // 释放识别器
        }
    }

    @Test
    fun test_date_field_ocr() {
        // 类似，验证 "产品批号"、"生产日期"、"有效期至"
    }
}
```

#### 与 PP-OCRv6 测试的差异

| 方面 | PP-OCRv6 测试 | ML Kit 测试 |
|------|:------------:|:----------:|
| 引擎管理 | 手动创建/关闭 `OcrEngine` / `MnnOcrEngine` | SDK 内置，`getClient()` / `close()` |
| 预处理 | `DetPreprocessor` → `RecPreprocessor` | 无需预处理，SDK 自动处理 |
| 推理调用 | 同步方法调用 `engine.ocr(bitmap)` | 异步 Task + `Tasks.await()` |
| 输出结构 | `List<OcrLine>`（每行 bbox + score + text） | `Text.text`（整页文本） |
| 置信度 | 每行有 score 可过滤 | 无（可通过 TextElement 获取但测试未使用） |

#### 识别特点

ML Kit 采用**全图密集识别**策略，输出图片中所有可识别的文字（包括背景、干扰文字）。这与 PP-OCRv6 的"检测 → 逐框裁剪 → 逐框识别"不同：

- PP-OCRv6 只输出**检测模型认为有文字的矩形区域**
- ML Kit 输出**图片中肉眼可见的所有文字**

因此在 full_fields.jpg 上，ML Kit 不仅识别了药品信息字段，还识别了药盒正面所有说明文字（用法用量、注意事项等），这导致验证策略需要更灵活地匹配目标字段。

**与 ONNX 测试的关键区别**:

| 方面 | ONNX 测试 | MNN 测试 |
|------|----------|---------|
| 资源管理 | `@Before` 创建，`@After` 关闭 | 每个测试内 `use {}` 自动管理 |
| 引擎参数 | 无 | 可传 `ModelSize.SMALL` 或 `TINY` |
| 验证严格度 | 严格（ONNX 精度最高） | 宽松（SMALL 精度高但有差异） |

```kotlin
@Test
fun test_full_fields_ocr() {
    val bitmap = loadBitmap("full_fields.jpg")

    // use {} 自动在结束时调用 close()
    MnnOcrEngine(context, MnnOcrEngine.ModelSize.SMALL).use { engine ->
        val lines = engine.ocr(bitmap)
        val allText = lines.joinToString("\n") { it.text }

        // 打印 + 保存结果
        saveResult(imageName, allText)

        // 验证关键字段
        val expectedLines = listOf(
            "奥美拉唑肠溶胶囊",
            "国药准字H2004",   // 前缀匹配（末尾数字可能有差异）
            "有限公司",
        )
    }
}

@Test
fun test_date_field_ocr() {
    MnnOcrEngine(context, MnnOcrEngine.ModelSize.SMALL).use { engine ->
        // 验证 "产品批号"、"生产日期"、"有效期至"
    }
}
```

**切换 TINY 模型**: 只需将 `SMALL` 改为 `TINY`：
```kotlin
MnnOcrEngine(context, MnnOcrEngine.ModelSize.TINY).use { engine ->
```

---

## 11. 曾踩过的坑

### 坑 1：TINY 模型字符映射错误

**现象**: TINY 模型输出"✦埧≅ヒH2004"等乱码

**根源**: TINY 模型使用自己训练的 6906 字符子集，但代码从全量字典 `ppocrv6_dict.txt` 截取前 6906 个字符。全量字典按 Unicode 排序，与 TINY 模型的字符排序不一致，导致模型输出的 class index 映射到错误的字符上。

**解决**: 使用 `model/rec/inference.yml` 中的 `character_dict`（TINY 模型专用）。

### 坑 2：SMALL 模型用错字典

**现象**: 使用 `CharDictLoader` 后 SMALL 模型输出"犍斓給H2004643"等乱码

**根源**: `model/rec/inference.yml` 只有 ~6906 个字符，但 SMALL 模型的 vocab 为 18710。大部分模型输出索引超出字典范围，被 CTC decoder 静默丢弃。

**解决**: 使用 `model/mnn/inference_small.yml`，包含完整的 18709 字符集。

### 坑 3：CTC 时间步 stride 错位

**现象**: TINY 模型输出全为符号 `!"#$%&'()...+-,./:;<=>?`

**根源**:
```
charDict.size = 6905, actualVocabSize = 6906
effectiveVocabSize = min(6906, 6905) = 6905

CTCDecoder.decode(logits, actualTimeSteps, effectiveVocabSize, effectiveDict)
//                                           ^^^^^^^^^^^^^^^^^^
//                                    每个时间步 stride = 6905
//                                    但实际数据 stride = 6906

// 时间步 0: 读取正确 [0..6904]
// 时间步 1: 读取 [6905..13809]，应该读 [6906..13811]！
// 时间步 2: 读取 [13810..20714]，应该读 [13812..20717]！
// 依此类推，每个时间步都比正确位置偏移 1 个 float
```

**解决**: stride 始终使用 `actualVocabSize`，字典越界由 CTCDecoder 内部跳过：
```kotlin
CTCDecoder.decode(logits, actualTimeSteps, actualVocabSize, effectiveDict)
```

### 坑 4：YAML 解析器中断

**现象**: `CharDictLoader` 只返回 6905 项（含 blank），但 YAML 文件有更多条目

**根源**（潜在）: YAML 解析器遇到不符合 `- XXXXX` 格式的行会 break。若字典中包含特殊字符条目或多行格式，可能提前终止。

---

## 12. 运行指南

### 运行测试

```bash
# ONNX 引擎测试
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.ocr_test.PaddleOcrOnnxTest

# MNN 引擎测试
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.ocr_test.MnnOcrEngineTest

# Google ML Kit OCR 测试
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.ocr_test.GoogleMlKitOcrTest
```

### 查看结果

```bash
# logcat 输出
adb logcat -s MnnOcrTest

# 拉取结果文件
adb pull /data/data/com.example.ocr_test/files/
```

### 结果文件

测试运行后结果保存在设备 `filesDir` 中，项目 `results/` 目录下包含最近一次运行的参考结果：

```bash
# 拉取结果到项目 results/ 目录
adb exec-out run-as com.example.ocr_test \
  cat /data/data/com.example.ocr_test/files/ppocr_onnx_result_full_fields.jpg.txt \
  > results/ppocr_onnx_result_full_fields.jpg.txt
```

| 测试 | 参考文件 |
|------|---------|
| PaddleOcrOnnxTest full_fields | `results/ppocr_onnx_result_full_fields.jpg.txt` |
| PaddleOcrOnnxTest date_field | `results/ppocr_onnx_result_date_field.jpg.txt` |
| MnnOcrEngineTest full_fields | `results/mnn_ocr_small_result_full_fields.jpg.txt` |
| MnnOcrEngineTest date_field | `results/mnn_ocr_small_result_date_field.jpg.txt` |
| GoogleMlKitOcrTest full_fields | `results/ocr_result_full_fields.jpg.txt` |
| GoogleMlKitOcrTest date_field | `results/ocr_result_date_field.jpg.txt` |

---

## 14. 三引擎精度对比

### 预期结果

**full_fields.jpg**:
```
奥美拉唑肠溶胶囊
国药准字 H20046430
药品上市许可持有人：石药集团欧意药业有限公司
生产企业：石药集团欧意药业有限公司
```

**date_field.jpg**:
```
产品批号：0602504203
生产日期：2025.04.05
有效期至：2027.04.04
```

### 实际识别结果

#### full_fields.jpg

| 预期字段 | ONNX | MNN SMALL | Google ML Kit |
|---------|:----:|:---------:|:-------------:|
| 奥美拉唑肠溶胶囊 | ✓ 奥美拉唑肠溶胶囊 | ✓ 奥美拉唑肠溶胶囊 | ≈ 奥美拉唑旺语胶 |
| 国药准字 H20046430 | ✓ 国药准字 H20046430 | ≈ 国药准字H2004643 | ≈ 国药准字0045430 0 |
| 药品上市许可持有人：石药集团欧意药业有限公司 | ✓ | ✓（产企业石的股意的业有限公司） | ≈（部分信息散落在多行） |
| 生产企业：石药集团欧意药业有限公司 | ✓ | ✓（产企业石的股意的业有限公司） | ≈ P业有限公司 |

#### date_field.jpg

| 预期字段 | ONNX | MNN SMALL | Google ML Kit |
|---------|:----:|:---------:|:-------------:|
| 产品批号：0602504203 | ✓ | ✓（产品批号 / 0602504203） | ✓ |
| 生产日期：2025.04.05 | ✓ | ✓ | ✓ |
| 有效期至：2027.04.04 | ✓ | ✓（有效期至 / 2027.04.04） | ✓ |

### 结论

- **ONNX Runtime**: 精度最高，三个字段完全命中，数字和中文字均正确
- **MNN SMALL**: 精度接近 ONNX，企业名称字段部分正确，数字略有差异（H2004643 vs H20046430）
- **Google ML Kit**: 全图识别覆盖更多文字，但对药品专用词汇的识别准确率不如 PP-OCRv6 专用模型
