//
// ncnn_ocr_jni.cpp — JNI bridge for ncnn PP-OCRv6 inference
//
// Uses ncnn C++ API to load and run PP-OCRv6 models.
// Models must be pre-converted from ONNX to ncnn .param/.bin format
// using the onnx2ncnn tool.
//
// cv2 → ncnn mapping (for reference):
//   resize         → ncnn::resize_bilinear     (simpleocv.h)
//   cvtColor       → ncnn::convert_pixel       (simpleocv.h)
//   morphologyEx   → ncnn::morphology          (simpleocv.h)
//   findContours   → ncnn::find_contours       (simpleocv.h)
//   minAreaRect    → ncnn::min_area_rect       (simpleocv.h)
//   fillPoly       → ncnn::fill_poly           (simpleocv.h)
//

#include <jni.h>
#include <android/log.h>
#include <cstring>

#include "ncnn/net.h"

#define LOG_TAG "ncnn_ocr"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Context holding both model nets ───────────────────

struct OcrContext {
    ncnn::Net* detNet;
    ncnn::Net* recNet;

    OcrContext() : detNet(nullptr), recNet(nullptr) {}
    ~OcrContext() {
        delete detNet;
        delete recNet;
    }
};

// ── Init: load .param + .bin for both models ──────────

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ocr_1test_ocr_NcnnOcrEngine_nativeInit(
    JNIEnv* env, jobject /*thiz*/,
    jstring detParamPath, jstring detBinPath,
    jstring recParamPath, jstring recBinPath)
{
    const char* detParam = env->GetStringUTFChars(detParamPath, nullptr);
    const char* detBin   = env->GetStringUTFChars(detBinPath, nullptr);
    const char* recParam = env->GetStringUTFChars(recParamPath, nullptr);
    const char* recBin   = env->GetStringUTFChars(recBinPath, nullptr);

    auto* ctx = new OcrContext();

    // ── Detection model ──
    ctx->detNet = new ncnn::Net();
    ctx->detNet->opt.num_threads = 4;
    ctx->detNet->opt.use_packing_layout = false;
    ctx->detNet->opt.use_bf16_storage = false;
    ctx->detNet->opt.use_fp16_packed = false;
    ctx->detNet->opt.use_fp16_storage = false;
    ctx->detNet->opt.use_fp16_arithmetic = false;

    if (ctx->detNet->load_param(detParam) != 0) {
        LOGE("Failed to load det param: %s", detParam);
        goto fail;
    }
    if (ctx->detNet->load_model(detBin) != 0) {
        LOGE("Failed to load det bin: %s", detBin);
        goto fail;
    }
    LOGI("Detection model loaded: %s / %s", detParam, detBin);

    // ── Recognition model ──
    ctx->recNet = new ncnn::Net();
    ctx->recNet->opt.num_threads = 4;
    ctx->recNet->opt.use_packing_layout = false;
    ctx->recNet->opt.use_bf16_storage = false;
    ctx->recNet->opt.use_fp16_packed = false;
    ctx->recNet->opt.use_fp16_storage = false;
    ctx->recNet->opt.use_fp16_arithmetic = false;

    if (ctx->recNet->load_param(recParam) != 0) {
        LOGE("Failed to load rec param: %s", recParam);
        goto fail;
    }
    if (ctx->recNet->load_model(recBin) != 0) {
        LOGE("Failed to load rec bin: %s", recBin);
        goto fail;
    }
    LOGI("Recognition model loaded: %s / %s", recParam, recBin);

    env->ReleaseStringUTFChars(detParamPath, detParam);
    env->ReleaseStringUTFChars(detBinPath, detBin);
    env->ReleaseStringUTFChars(recParamPath, recParam);
    env->ReleaseStringUTFChars(recBinPath, recBin);

    return reinterpret_cast<jlong>(ctx);

fail:
    delete ctx;
    env->ReleaseStringUTFChars(detParamPath, detParam);
    env->ReleaseStringUTFChars(detBinPath, detBin);
    env->ReleaseStringUTFChars(recParamPath, recParam);
    env->ReleaseStringUTFChars(recBinPath, recBin);
    return 0;
}

// ── Detection inference ───────────────────────────────

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_ocr_1test_ocr_NcnnOcrEngine_nativeDetect(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jfloatArray input, jint w, jint h)
{
    auto* ctx = reinterpret_cast<OcrContext*>(handle);
    if (!ctx || !ctx->detNet) return nullptr;

    // Float array data must stay valid until extract() completes
    jfloat* inputData = env->GetFloatArrayElements(input, nullptr);
    ncnn::Mat inMat(w, h, 3, inputData);

    ncnn::Extractor ex = ctx->detNet->create_extractor();
    ex.input("in0", inMat);

    ncnn::Mat outMat;
    int ret = ex.extract("out0", outMat);

    env->ReleaseFloatArrayElements(input, inputData, JNI_ABORT);

    if (ret != 0) {
        LOGE("nativeDetect extract failed (ret=%d)", ret);
        return nullptr;
    }

    // Detection output
    int oW = outMat.w;
    int oH = outMat.h;
    int dataSize = oW * oH;
    LOGI("nativeDetect output: w=%d h=%d c=%d", oW, oH, outMat.c);

    // Pack: [oW, oH, probMap...]
    jfloatArray result = env->NewFloatArray(2 + dataSize);
    jfloat* dst = env->GetFloatArrayElements(result, nullptr);
    dst[0] = static_cast<jfloat>(oW);
    dst[1] = static_cast<jfloat>(oH);
    memcpy(dst + 2, outMat.channel(0), dataSize * sizeof(float));
    env->ReleaseFloatArrayElements(result, dst, 0);

    return result;
}

// ── Recognition inference ─────────────────────────────

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_ocr_1test_ocr_NcnnOcrEngine_nativeRecognize(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jfloatArray input)
{
    auto* ctx = reinterpret_cast<OcrContext*>(handle);
    if (!ctx || !ctx->recNet) return nullptr;

    jfloat* inputData = env->GetFloatArrayElements(input, nullptr);
    ncnn::Mat inMat(320, 48, 3, inputData);

    ncnn::Extractor ex = ctx->recNet->create_extractor();
    ex.input("in0", inMat);

    ncnn::Mat outMat;
    int ret = ex.extract("out0", outMat);

    env->ReleaseFloatArrayElements(input, inputData, JNI_ABORT);

    if (ret != 0) {
        LOGE("nativeRecognize extract failed (ret=%d)", ret);
        return nullptr;
    }

    // Output dimensions
    int vocabSize = outMat.w;
    int timeSteps = outMat.h;
    int channels = outMat.c;
    int total = vocabSize * timeSteps * channels;

    LOGI("nativeRecognize output: w=%d h=%d c=%d total=%d",
         vocabSize, timeSteps, channels, total);

    // Log first 8 values for debugging
    if (total > 0) {
        const float* raw = outMat.channel(0);
        LOGI("  out[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
    }

    jfloatArray result = env->NewFloatArray(total);
    jfloat* dst = env->GetFloatArrayElements(result, nullptr);
    memcpy(dst, outMat.channel(0), total * sizeof(float));
    env->ReleaseFloatArrayElements(result, dst, 0);

    return result;
}

// ── Release ───────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_example_ocr_1test_ocr_NcnnOcrEngine_nativeRelease(
    JNIEnv* /*env*/, jobject /*thiz*/,
    jlong handle)
{
    delete reinterpret_cast<OcrContext*>(handle);
    LOGI("ncnn resources released");
}
