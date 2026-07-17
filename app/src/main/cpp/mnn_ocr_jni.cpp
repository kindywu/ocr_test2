//
// mnn_ocr_jni.cpp — JNI bridge for MNN PP-OCRv6 inference
//
// Uses MNN C++ API to load and run PP-OCRv6 .mnn models.
// Follows the same pattern as ncnn_ocr_jni.cpp.
//
// MNN C++ API reference:
//   https://github.com/alibaba/MNN/tree/master/include/MNN
//

#include <jni.h>
#include <android/log.h>
#include <cstring>
#include <string>
#include <memory>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#define LOG_TAG "mnn_ocr"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Context holding both model interpreters ──────────────

struct OcrContext {
    std::shared_ptr<MNN::Interpreter> detNet;
    std::shared_ptr<MNN::Interpreter> recNet;
    MNN::Session* detSession;
    MNN::Session* recSession;

    OcrContext() : detSession(nullptr), recSession(nullptr) {}
    ~OcrContext() {
        // MNN::Interpreter destructor handles session cleanup
    }
};

// ── Helper: create MNN session ───────────────────────────

static MNN::Session* createSession(MNN::Interpreter* net) {
    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type = MNN_FORWARD_CPU;
    return net->createSession(config);
}

// ── Init: load .mnn files for both models ────────────────

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeInit(
    JNIEnv* env, jobject /*thiz*/,
    jstring detModelPath, jstring recModelPath)
{
    const char* detPath = env->GetStringUTFChars(detModelPath, nullptr);
    const char* recPath = env->GetStringUTFChars(recModelPath, nullptr);

    auto* ctx = new OcrContext();

    // Detection model
    ctx->detNet.reset(MNN::Interpreter::createFromFile(detPath));
    if (!ctx->detNet) {
        LOGE("Failed to load det model: %s", detPath);
        goto fail;
    }
    ctx->detSession = createSession(ctx->detNet.get());
    if (!ctx->detSession) {
        LOGE("Failed to create det session");
        goto fail;
    }
    LOGI("Detection model loaded: %s", detPath);

    // Recognition model
    ctx->recNet.reset(MNN::Interpreter::createFromFile(recPath));
    if (!ctx->recNet) {
        LOGE("Failed to load rec model: %s", recPath);
        goto fail;
    }
    ctx->recSession = createSession(ctx->recNet.get());
    if (!ctx->recSession) {
        LOGE("Failed to create rec session");
        goto fail;
    }
    LOGI("Recognition model loaded: %s", recPath);

    env->ReleaseStringUTFChars(detModelPath, detPath);
    env->ReleaseStringUTFChars(recModelPath, recPath);

    return reinterpret_cast<jlong>(ctx);

fail:
    delete ctx;
    env->ReleaseStringUTFChars(detModelPath, detPath);
    env->ReleaseStringUTFChars(recModelPath, recPath);
    return 0;
}

// ── Detection inference ──────────────────────────────────
//
// Input:  float array in NCHW layout [1, 3, h, w]
// Output: packed float array [oW, oH, probMap...]

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeDetect(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jfloatArray input, jint h, jint w)
{
    auto* ctx = reinterpret_cast<OcrContext*>(handle);
    if (!ctx || !ctx->detNet) return nullptr;

    MNN::Interpreter* net = ctx->detNet.get();
    MNN::Session* session = ctx->detSession;

    // Get input tensor
    auto inputTensor = net->getSessionInput(session, nullptr);
    if (!inputTensor) {
        LOGE("nativeDetect: no input tensor");
        return nullptr;
    }

    // Resize input to [1, 3, h, w]
    net->resizeTensor(inputTensor, {1, 3, h, w});
    net->resizeSession(session);

    // Copy input data from Java
    jfloat* inputData = env->GetFloatArrayElements(input, nullptr);

    // Create host tensor and copy data
    std::unique_ptr<MNN::Tensor> hostTensor(
        new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE)); // CAFFE = NCHW
    ::memcpy(hostTensor->host<float>(), inputData,
             static_cast<size_t>(1 * 3 * h * w) * sizeof(float));
    inputTensor->copyFromHostTensor(hostTensor.get());

    env->ReleaseFloatArrayElements(input, inputData, JNI_ABORT);

    // Run inference
    net->runSession(session);

    // Get output tensor
    auto outputTensor = net->getSessionOutput(session, nullptr);
    if (!outputTensor) {
        LOGE("nativeDetect: no output tensor");
        return nullptr;
    }

    // Copy output to host
    std::unique_ptr<MNN::Tensor> hostOut(
        new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE));
    outputTensor->copyToHostTensor(hostOut.get());

    auto dims = hostOut->shape();
    int oH = (dims.size() >= 4) ? static_cast<int>(dims[2]) : h;
    int oW = (dims.size() >= 4) ? static_cast<int>(dims[3]) : w;
    int dataSize = oH * oW;

    LOGI("nativeDetect output shape: [%s] → %dx%d",
         [&dims]() -> std::string {
             std::string s;
             for (auto d : dims) s += std::to_string(d) + " ";
             return s;
         }().c_str(),
         oH, oW);

    // Pack: [oW, oH, probMap...]
    jfloatArray result = env->NewFloatArray(2 + dataSize);
    jfloat* dst = env->GetFloatArrayElements(result, nullptr);
    dst[0] = static_cast<jfloat>(oW);
    dst[1] = static_cast<jfloat>(oH);

    const float* srcData = hostOut->host<float>();
    if (dims.size() >= 4) {
        // Output is [1, 1, oH, oW] - skip batch/channel dims
        int channelSize = oH * oW;
        ::memcpy(dst + 2, srcData + channelSize * 0,  // channel 0
                 static_cast<size_t>(dataSize) * sizeof(float));
    } else {
        ::memcpy(dst + 2, srcData,
                 static_cast<size_t>(dataSize) * sizeof(float));
    }

    env->ReleaseFloatArrayElements(result, dst, 0);
    return result;
}

// ── Recognition inference ────────────────────────────────
//
// Input:  float array in NCHW layout [1, 3, 48, 320]
// Output: packed float array [timeSteps, vocabSize, data...]
//
// We prepend the actual output dimensions so the Kotlin side
// doesn't have to guess — the model's vocabSize may differ from
// the dictionary file size (e.g. PP-OCRv6_tiny uses 6906 vs 18710).
// This mirrors what the Python reference does: read outputShape
// from the MNN tensor and use it directly in CTC decode.

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeRecognize(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jfloatArray input)
{
    auto* ctx = reinterpret_cast<OcrContext*>(handle);
    if (!ctx || !ctx->recNet) return nullptr;

    MNN::Interpreter* net = ctx->recNet.get();
    MNN::Session* session = ctx->recSession;

    // Get input tensor
    auto inputTensor = net->getSessionInput(session, nullptr);
    if (!inputTensor) {
        LOGE("nativeRecognize: no input tensor");
        return nullptr;
    }

    // Resize input to [1, 3, 48, 320]
    net->resizeTensor(inputTensor, {1, 3, 48, 320});
    net->resizeSession(session);

    // Copy input data
    jfloat* inputData = env->GetFloatArrayElements(input, nullptr);

    std::unique_ptr<MNN::Tensor> hostTensor(
        new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE));
    ::memcpy(hostTensor->host<float>(), inputData,
             static_cast<size_t>(1 * 3 * 48 * 320) * sizeof(float));
    inputTensor->copyFromHostTensor(hostTensor.get());

    env->ReleaseFloatArrayElements(input, inputData, JNI_ABORT);

    // Run inference
    net->runSession(session);

    // Get output
    auto outputTensor = net->getSessionOutput(session, nullptr);
    if (!outputTensor) {
        LOGE("nativeRecognize: no output tensor");
        return nullptr;
    }

    // Copy to host
    std::unique_ptr<MNN::Tensor> hostOut(
        new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE));
    outputTensor->copyToHostTensor(hostOut.get());

    auto dims = hostOut->shape();
    size_t totalSize = hostOut->elementSize();

    // dims = [1, timeSteps, vocabSize] for rec models
    int timeSteps = (dims.size() >= 3) ? static_cast<int>(dims[1]) : 0;
    int vocabSize = (dims.size() >= 3) ? static_cast<int>(dims[2]) : 0;
    // Fallback for 2D output: treat as [timeSteps, vocabSize]
    if (dims.size() == 2) {
        timeSteps = static_cast<int>(dims[0]);
        vocabSize = static_cast<int>(dims[1]);
    }

    LOGI("nativeRecognize output shape: [%s] → T=%d V=%d total=%zu",
         [&dims]() -> std::string {
             std::string s;
             for (auto d : dims) s += std::to_string(d) + " ";
             return s;
         }().c_str(),
         timeSteps, vocabSize, totalSize);

    // Pack: [timeSteps, vocabSize, data...]
    const float* srcData = hostOut->host<float>();
    int headerLen = 2;
    jfloatArray result = env->NewFloatArray(headerLen + static_cast<jsize>(totalSize));
    jfloat* dst = env->GetFloatArrayElements(result, nullptr);
    dst[0] = static_cast<jfloat>(timeSteps);
    dst[1] = static_cast<jfloat>(vocabSize);
    ::memcpy(dst + headerLen, srcData, totalSize * sizeof(float));
    env->ReleaseFloatArrayElements(result, dst, 0);

    return result;
}

// ── Release ──────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_example_ocr_1test_ocr_MnnOcrEngine_nativeRelease(
    JNIEnv* /*env*/, jobject /*thiz*/,
    jlong handle)
{
    delete reinterpret_cast<OcrContext*>(handle);
    LOGI("MNN resources released");
}
