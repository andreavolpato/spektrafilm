# 135（35mm）胶片品牌边框模板套装

本包有 6 款模板，每款均有横向和竖向两套：透明 PNG（可直接叠图）与可编辑 SVG（可修改文字、帧号、颜色）。另外附带每款效果预览，以及六款总览。

## 六款模板
1. Kodak Portra 400 — 边缘印字 `KODAK PORTRA 400`
2. Kodak Gold 200 — 选用有实拍资料支持的代际边缘印字 `GB 200-7  KODAK`，**并不是**随意写 `KODAK GOLD 200`。
3. Kodak Ektar 100 — `KODAK EKTAR 100`
4. Kodak ColorPlus 200 — 选用实拍/讨论资料支持的旧版 `200-8  KODAK`，不同批次可能不同。
5. Fujifilm Superia X-TRA 400 — 使用可核实的 `S-400`、`FUJI` 和红绿边缘条纹；不擅自套用未经核实的批次号。
6. 标准无品牌 135 — 没有任何杜撰的型号、数字或编码。

## 模板定位与真实性边界（重要）
- 所有模板均为**依据真实 135 胶片规格创作的可编辑设计模板**，并不是某一具体年份、生产批次、胶卷的逐像素扫描复刻。
- `12` 和 `12A` 是**画幅编号示例**，不是此照片的真实编号；可以在SVG中修改，或由软件明确提供用户输入并可隐藏。
- 没有加入无法验证的批次号、生产序列、DX条码或随机纹样，不要自动生成假的机器可读码。
- 若要达到**严格考据级**的某一具体胶卷版本，必须另有对应年代和批次的底片扫描，并逐个确认文字、尺寸、位置、编码。保留可覆盖这些模板的“真实扫描版”方案。
- 电影胶卷的 KEYKODE 不能直接套进 135 普通静态摄影胶片里。

## 精确尺寸、像素坐标
- 原始胶片横向画布：3800 × 3500 px；**中心透明照片开窗**：`x=100, y=550, width=3600, height=2400`。
- 竖向画布：3500 × 3800 px；**中心透明照片开窗**：`x=550, y=100, width=2400, height=3600`。
- 这些是35mm×38mm的135胶片片段，以及24mm×36mm照片成像区域按 100px/mm 缩放所得。
- 每侧8个齿孔，沿胶片长度每4.75mm排列。照片窗口、齿孔均透明。
- **横竖版是整张胶片旋转，边缘文字也跟着旋转**，不是只旋转照片。

## 集成原则
1. 加载图像，完成EXIF方向矫正，判断横竖比例。
2. 横向选择横向模板，竖向选择竖向模板；方形/其他比例允许用户选择“等比裁切填满”或“完整适配留白”。
3. 背景先画暗/亮色底（用于透出齿孔），将照片等比适配到相应开窗坐标，照片不能延伸到齿孔下方。
4. 将 PNG/SVG 边框整体作为图层覆盖在照片之上，**不重新画胶片齿孔和文字**。
5. 用户选择胶卷名称 = 加载对应文件。照片只变更中间窗口的内容；边框不因照片而重新生成。
6. 编号作为可选项：用户输入后，仅替换模板内预留编号，不允许随机写。日期默认为关闭；需要时作为独立照片日期戳层，不算胶卷印字。
7. 预览和导出复用同一套图层坐标和绘制逻辑；整个画布同比例缩放，不能破坏内部布局。
8. 不要把模板像素直接拉伸到非匹配宽高比的画布，不可对照片做非等比拉伸。

## 来源（核对边缘文字时参照）
- Kodak Portra 400 官方页面：https://www.kodakprofessional.com/photographers/film/color/kodak-professional-portra-400-film/516
- Kodak Gold 200 实拍扫描：https://www.pentaxforums.com/forums/107-film-processing-scanning-darkroom/466730-some-weird-light-leaks-pentax-z-20-kodak-gold-200-a.html
- Kodak ColorPlus 200 历史边码交流：https://www.photrio.com/forum/threads/kodak-color-plus-200-edge-code-what-is-this-film.129541/
- Fuji Superia X-TRA 400 官方规格：https://asset.fujifilm.com/www/hu/files/2019-09/af234c58ab3d3b67b41cde22b1155309/films_superia-xtra400_datasheet_01.pdf
- Ektar 100 底片边缘实拍：https://www.photrio.com/forum/threads/bellini-c-41-chemistry-expensive-and-unusual-what-do-we-know-about-it.202038/
- 135 规格图示：https://standards.iteh.ai/catalog/standards/iso/8208bdce-8324-419b-9542-ac3bc6b2e742/iso-1007-1979

## 文件说明
`*_横向.png` `*_竖向.png` 是透明图层；同名 `*.svg` 为可编辑源文件；`*_预览.jpg` 用用户提供的屏幕截图中的摄影画面做示例。`00_六款模板总览.jpg`方便给开发者先看；`template_manifest.json`供开发者读取精准坐标。
