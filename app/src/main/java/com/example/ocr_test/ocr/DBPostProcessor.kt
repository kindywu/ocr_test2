package com.example.ocr_test.ocr

import kotlin.math.*
import kotlin.math.roundToInt

/**
 * DB (Differential Binarization) 后处理 — 对应 main2.py 的 db_postprocess() + boxes_from_polys()
 *
 * 纯 Kotlin 实现，无 OpenCV 依赖。
 *
 * 流程:
 *   1. 二值化 (THRESH=0.2)
 *   2. 形态学闭操作 (3×3 核)
 *   3. 连通域分析 → 候选区域
 *   4. 对每个区域: 凸包 → minAreaRect → 置信度 → unclip → 映射回原图
 *   5. 输出轴对齐矩形
 */
object DBPostProcessor {

    private const val DB_THRESH = 0.2f
    private const val DB_BOX_THRESH = 0.4f
    private const val DB_UNCLIP_RATIO = 1.4f
    private const val MIN_AREA = 100

    data class Box4(
        val x: FloatArray,  // 4 个点的 x 坐标
        val y: FloatArray,  // 4 个点的 y 坐标
    )

    /**
     * @param probMap   检测模型输出的概率图 (H×W, float 0~1)
     * @param probW     概率图宽度
     * @param probH     概率图高度
     * @param origW     原图宽度
     * @param origH     原图高度
     * @param scaleX    缩放比 (原图宽 / 缩放后宽)
     * @param scaleY    缩放比 (原图高 / 缩放后高)
     * @return 轴对齐矩形 [x1, y1, x2, y2]
     */
    fun process(
        probMap: FloatArray,
        probW: Int,
        probH: Int,
        origW: Int,
        origH: Int,
        scaleX: Float,
        scaleY: Float,
    ): List<IntArray> {
        // 1. 二值化
        val binary = ByteArray(probW * probH) { i ->
            if (probMap[i] > DB_THRESH) 1 else 0
        }

        // 2. 形态学闭操作 (dilate → erode, 3×3)
        val closed = morphologyClose(binary, probW, probH)

        // 3. 连通域分析
        val components = connectedComponents(closed, probW, probH)

        val boxes = mutableListOf<IntArray>()

        for (points in components) {
            if (points.size < 6) continue

            // 4a. 凸包
            val hull = convexHull(points)
            if (hull.size < 4) continue

            // 4b. 最小面积有向矩形
            val rect = minAreaRect(hull) ?: continue
            if (rect.x.size != 4) continue

            // 4c. box_score_fast — 矩形区域的概率均值
            val score = boxScoreFast(probMap, probW, probH, rect)
            if (score < DB_BOX_THRESH) continue

            // 4d. unclip 膨胀
            val expanded = unclip(rect, DB_UNCLIP_RATIO) ?: continue

            // 4e. 映射回原图坐标
            for (i in 0 until 4) {
                expanded.x[i] /= scaleX
                expanded.y[i] /= scaleY
                expanded.x[i] = expanded.x[i].coerceIn(0f, origW.toFloat())
                expanded.y[i] = expanded.y[i].coerceIn(0f, origH.toFloat())
            }

            // 转轴对齐矩形 [x1, y1, x2, y2]
            val x1 = expanded.x.min().roundToInt()
            val y1 = expanded.y.min().roundToInt()
            val x2 = expanded.x.max().roundToInt()
            val y2 = expanded.y.max().roundToInt()
            boxes.add(intArrayOf(x1, y1, x2, y2))
        }

        return boxes
    }

    // ── 形态学操作 ──────────────────────────────────────

    private fun morphologyClose(src: ByteArray, w: Int, h: Int): ByteArray {
        val dilated = dilate(src, w, h)
        return erode(dilated, w, h)
    }

    private fun dilate(src: ByteArray, w: Int, h: Int): ByteArray {
        val dst = ByteArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var maxV = 0
                for (ky in -1..1) {
                    for (kx in -1..1) {
                        val nx = x + kx
                        val ny = y + ky
                        if (nx in 0 until w && ny in 0 until h) {
                            if (src[ny * w + nx].toInt() > maxV) maxV = 1
                        }
                    }
                }
                dst[y * w + x] = maxV.toByte()
            }
        }
        return dst
    }

    private fun erode(src: ByteArray, w: Int, h: Int): ByteArray {
        val dst = ByteArray(w * h) { 1 }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var minV = 1
                for (ky in -1..1) {
                    for (kx in -1..1) {
                        val nx = x + kx
                        val ny = y + ky
                        if (nx in 0 until w && ny in 0 until h) {
                            if (src[ny * w + nx].toInt() < minV) minV = 0
                        }
                    }
                }
                dst[y * w + x] = minV.toByte()
            }
        }
        return dst
    }

    // ── 连通域分析 (两遍扫描 + Union-Find) ──────────────

    private data class Point(val x: Int, val y: Int)

    private fun connectedComponents(binary: ByteArray, w: Int, h: Int): List<List<Point>> {
        val labels = IntArray(w * h) { 0 }
        val parent = IntArray(w * h) { it } // Union-Find
        var nextLabel = 1

        fun find(x: Int): Int {
            var p = x
            while (parent[p] != p) { parent[p] = parent[parent[p]]; p = parent[p] }
            return p
        }

        fun union(a: Int, b: Int) {
            val ra = find(a); val rb = find(b)
            if (ra != rb) parent[rb] = ra
        }

        // 第一遍: 分配临时标签 (8-connectivity — 匹配 OpenCV findContours)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                if (binary[idx].toInt() == 0) continue

                val left     = if (x > 0)             labels[y * w + (x - 1)] else 0
                val topLeft  = if (x > 0 && y > 0)   labels[(y - 1) * w + (x - 1)] else 0
                val top      = if (y > 0)             labels[(y - 1) * w + x] else 0
                val topRight = if (x < w - 1 && y > 0) labels[(y - 1) * w + (x + 1)] else 0

                val neighbors = listOf(left, topLeft, top, topRight).filter { it > 0 }
                if (neighbors.isEmpty()) {
                    labels[idx] = nextLabel++
                } else {
                    val minLabel = neighbors.min()
                    labels[idx] = minLabel
                    for (nl in neighbors) {
                        if (nl != minLabel) union(minLabel, nl)
                    }
                }
            }
        }

        // 第二遍: 通过 Union-Find 合并标签
        val labelMap = IntArray(nextLabel) { 0 }
        var current = 0
        for (i in 1 until nextLabel) {
            val root = find(i)
            if (labelMap[root] == 0) labelMap[root] = ++current
            labelMap[i] = labelMap[root]
        }

        // 收集每个连通域的像素
        val components = Array(current + 1) { mutableListOf<Point>() }
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                if (binary[idx].toInt() == 0) continue
                val label = labelMap[labels[idx]]
                if (label > 0) components[label].add(Point(x, y))
            }
        }

        // 过滤面积过小的
        return components.filter { it.size >= MIN_AREA }
    }

    // ── 凸包 (Andrew's Monotone Chain) ─────────────────

    private fun convexHull(points: List<Point>): List<Point> {
        if (points.size < 3) return points

        val sorted = points.sortedWith(compareBy({ it.x }, { it.y }))
        val n = sorted.size
        val hull = mutableListOf<Point>()

        // 下凸包
        for (p in sorted) {
            while (hull.size >= 2 && cross(hull[hull.size - 2], hull.last(), p) <= 0) {
                hull.removeAt(hull.lastIndex)
            }
            hull.add(p)
        }

        // 上凸包
        val lowerSize = hull.size
        for (i in n - 2 downTo 0) {
            val p = sorted[i]
            while (hull.size > lowerSize && cross(hull[hull.size - 2], hull.last(), p) <= 0) {
                hull.removeAt(hull.lastIndex)
            }
            hull.add(p)
        }

        // 移除最后一个重复的点（回到起点）
        if (hull.size > 1) hull.removeAt(hull.lastIndex)
        return hull
    }

    private fun cross(o: Point, a: Point, b: Point): Long {
        return (a.x - o.x).toLong() * (b.y - o.y) - (a.y - o.y).toLong() * (b.x - o.x)
    }

    // ── 最小面积有向矩形 (Rotating Calipers) ──────────

    private fun minAreaRect(hull: List<Point>): Box4? {
        val n = hull.size
        if (n < 3) return null

        // 转换为 double 方便计算
        val hx = hull.map { it.x.toDouble() }
        val hy = hull.map { it.y.toDouble() }

        var minArea = Double.MAX_VALUE
        var bestBox: Box4? = null

        for (i in 0 until n) {
            val j = (i + 1) % n
            val dx = hx[j] - hx[i]
            val dy = hy[j] - hy[i]
            val edgeLen = sqrt(dx * dx + dy * dy)
            if (edgeLen < 1e-10) continue

            // 单位方向向量和法向量
            val ux = dx / edgeLen
            val uy = dy / edgeLen
            val vx = -uy
            val vy = ux

            // 投影所有顶点到这两个轴上
            var minU = Double.MAX_VALUE
            var maxU = -Double.MAX_VALUE
            var minV = Double.MAX_VALUE
            var maxV = -Double.MAX_VALUE

            for (k in 0 until n) {
                val ru = (hx[k] - hx[i]) * ux + (hy[k] - hy[i]) * uy
                val rv = (hx[k] - hx[i]) * vx + (hy[k] - hy[i]) * vy
                if (ru < minU) minU = ru
                if (ru > maxU) maxU = ru
                if (rv < minV) minV = rv
                if (rv > maxV) maxV = rv
            }

            val area = (maxU - minU) * (maxV - minV)
            if (area < minArea) {
                minArea = area

                // 四个角点
                val c1x = hx[i] + minU * ux + minV * vx
                val c1y = hy[i] + minU * uy + minV * vy
                val c2x = hx[i] + maxU * ux + minV * vx
                val c2y = hy[i] + maxU * uy + minV * vy
                val c3x = hx[i] + maxU * ux + maxV * vx
                val c3y = hy[i] + maxU * uy + maxV * vy
                val c4x = hx[i] + minU * ux + maxV * vx
                val c4y = hy[i] + minU * uy + maxV * vy

                bestBox = Box4(
                    x = floatArrayOf(c1x.toFloat(), c2x.toFloat(), c3x.toFloat(), c4x.toFloat()),
                    y = floatArrayOf(c1y.toFloat(), c2y.toFloat(), c3y.toFloat(), c4y.toFloat()),
                )
            }
        }

        return bestBox
    }

    // ── box_score_fast ──────────────────────────────────

    private fun boxScoreFast(
        probMap: FloatArray,
        probW: Int,
        probH: Int,
        box: Box4,
    ): Float {
        // 获取矩形包围盒 (floor min / ceil max — 匹配 Python np.floor / np.ceil)
        val xMin = floor(box.x.min().toDouble()).toInt().coerceIn(0, probW - 1)
        val xMax = ceil(box.x.max().toDouble()).toInt().coerceIn(0, probW - 1)
        val yMin = floor(box.y.min().toDouble()).toInt().coerceIn(0, probH - 1)
        val yMax = ceil(box.y.max().toDouble()).toInt().coerceIn(0, probH - 1)

        val rw = xMax - xMin + 1
        val rh = yMax - yMin + 1

        // 在局部区域创建 mask
        val mask = ByteArray(rw * rh)
        val localX = FloatArray(4) { box.x[it] - xMin }
        val localY = FloatArray(4) { box.y[it] - yMin }
        fillConvexPolygon(mask, rw, rh, localX, localY)

        // 计算 mask 内概率均值
        var sum = 0f
        var count = 0
        for (y in 0 until rh) {
            for (x in 0 until rw) {
                if (mask[y * rw + x].toInt() == 1) {
                    sum += probMap[(yMin + y) * probW + (xMin + x)]
                    count++
                }
            }
        }

        return if (count > 0) sum / count else 0f
    }

    /**
     * 填充凸四边形 (扫描线算法)
     */
    private fun fillConvexPolygon(
        mask: ByteArray, w: Int, h: Int,
        xs: FloatArray, ys: FloatArray,
    ) {
        // 按 y 排序顶点
        val order = (0 until 4).sortedBy { ys[it] }
        val yTop = ys[order[0]].roundToInt().coerceIn(0, h - 1)
        val yBot = ys[order[3]].roundToInt().coerceIn(0, h - 1)

        for (y in yTop..yBot) {
            val fy = y.toFloat()
            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE

            // 找当前扫描线与四边形四条边的交点
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                val y1 = ys[i]; val y2 = ys[j]
                if ((fy >= y1 && fy < y2) || (fy >= y2 && fy < y1)) {
                    val t = (fy - y1) / (y2 - y1)
                    val ix = xs[i] + t * (xs[j] - xs[i])
                    if (ix < minX) minX = ix
                    if (ix > maxX) maxX = ix
                }
            }

            val sx = ceil(minX).roundToInt().coerceIn(0, w - 1)
            val ex = floor(maxX).roundToInt().coerceIn(0, w - 1)
            for (x in sx..ex) {
                mask[y * w + x] = 1
            }
        }
    }

    // ── unclip (多边形膨胀) ──────────────────────────────

    private fun unclip(box: Box4, ratio: Float): Box4? {
        val area = polygonArea(box)
        val perimeter = polygonPerimeter(box)
        if (perimeter < 1e-6f) return null

        val distance = area * ratio / perimeter

        // 对四条边分别向外偏移
        val result = offsetConvexPolygon(box, distance)
        return result
    }

    private fun polygonArea(box: Box4): Float {
        var area = 0f
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            area += box.x[i] * box.y[j] - box.x[j] * box.y[i]
        }
        return abs(area) / 2f
    }

    private fun polygonPerimeter(box: Box4): Float {
        var perim = 0f
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            val dx = box.x[j] - box.x[i]
            val dy = box.y[j] - box.y[i]
            perim += sqrt(dx * dx + dy * dy)
        }
        return perim
    }

    /**
     * 凸四边形向外膨胀
     * 原理: 对每条边计算向外法线方向，沿法线平移 distance
     * 然后求四对相邻边的交点
     */
    private fun offsetConvexPolygon(box: Box4, distance: Float): Box4? {
        // 计算每条边的单位法向量 (指向外侧)
        val nx = FloatArray(4)
        val ny = FloatArray(4)

        for (i in 0 until 4) {
            val j = (i + 1) % 4
            val edx = box.x[j] - box.x[i]
            val edy = box.y[j] - box.y[i]
            val len = sqrt(edx * edx + edy * edy)
            if (len < 1e-6f) continue

            // 向外法线: (-edy/len, edx/len) — 因为顶点是逆时针
            nx[i] = -edy / len
            ny[i] = edx / len
        }

        // 对每个顶点，偏移相邻两边的交点
        val ox = FloatArray(4)
        val oy = FloatArray(4)

        for (i in 0 until 4) {
            val prev = (i + 3) % 4
            val a1 = nx[prev]; val b1 = ny[prev]
            val a2 = nx[i]; val b2 = ny[i]
            val det = a1 * b2 - a2 * b1
            if (abs(det) < 1e-6f) continue

            // 原顶点沿 prev 边法线方向偏移
            val p1x = box.x[i] + a1 * distance
            val p1y = box.y[i] + b1 * distance
            val p2x = box.x[i] + a2 * distance
            val p2y = box.y[i] + b2 * distance

            // 求两条直线的交点
            // Line 1: (p1x, p1y) + t * a1, b1
            // Line 2: (p2x, p2y) + s * a2, b2
            // 解: (p1x - p2x, p1y - p2y) = s*(a2,b2) - t*(a1,b1)
            // 用 Cramer's rule
            val dx = p2x - p1x
            val dy = p2y - p1y
            // a1 * t - a2 * s = dx
            // b1 * t - b2 * s = dy
            val t = (-dx * b2 + dy * a2) / det
            ox[i] = p1x + a1 * t
            oy[i] = p1y + b1 * t
        }

        return Box4(ox, oy)
    }
}
