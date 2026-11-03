"""Chinese (zh-CN) translations for the spektrafilm GUI.

Strings are translated at the presentation boundary only. Identifiers
(enum values, film/print profile ids, napari layer names) stay in
English so persistence and the runtime are untouched.

The language is selected with the ``SPEKTRAFILM_LANG`` environment
variable:

* ``auto`` (default) - follow the system locale (Chinese on zh locales,
                       English elsewhere)
* ``zh``             - Chinese UI
* ``en``             - English UI

Keys are the exact strings used at the call sites. Strings that flow
through :func:`normalize_ui_text` arrive lowercased, so their keys are
lowercase; strings translated directly (tab names, action buttons,
status templates) keep their original case.
"""

from __future__ import annotations

import os
import sys

TRANSLATIONS: dict[str, str] = {
    # --- section titles -----------------------------------------------------
    'couplers': '成色剂',
    'halation': '光晕',
    'grain': '颗粒',
    'glare': '眩光',
    'chemistry': '显影化学',
    'preflash': '预闪曝',
    'diffusion': '柔光镜',
    'camera': '相机',
    'scanner': '扫描仪',
    'input gamut compress': '输入色域压缩',
    'output gamut compress': '输出色域压缩',
    'input': '输入',
    'import raw': '导入 RAW',
    'import rgb': '导入 RGB',
    'crop and upscale': '裁剪与放大',
    'spectral upsampling': '光谱上采样',
    'tune': '微调',
    'experimental': '实验性功能',
    'gui parameters': '界面参数',
    'display': '显示',
    'profiles': '档案',
    'output': '输出',
    'exposure control': '曝光控制',
    'enlarger': '放大机',
    'napari layers': 'napari 图层',

    # --- tabs and action buttons (preserve original case) -------------------
    'MAIN': '主控',
    'FILM': '胶片',
    'PRINT': '相纸',
    'ADVANCED': '高级',
    'CONFIG': '设置',
    'PREVIEW': '预览',
    'SCAN': '扫描',
    'SAVE': '保存',
    'update': '更新',

    # --- buttons / placeholders / dialog titles ------------------------------
    'select file': '选择文件',
    'reprocess raw': '重新处理 RAW',
    'save current as default': '保存为默认',
    'save current to file': '保存到文件',
    'load from file': '从文件载入',
    'restore factory default': '恢复出厂默认',
    'no raw selected': '未选择 RAW 文件',
    'no image selected': '未选择图像',
    'select input raw': '选择输入 RAW 文件',
    'select input image': '选择输入图像',
    'ccw rotate': '逆时针旋转',
    'cw rotate': '顺时针旋转',
    'reset view': '重置视图',

    # --- DIR couplers ---------------------------------------------------
    'active': '启用',
    'amount': '强度',
    'inhibition samelayer': '同层抑制',
    'inhibition interlayer': '层间抑制',
    'gamma samelayer rgb': '同层伽马 (R,G,B)',
    'gamma interlayer r to gb': 'R→G/B 层间伽马',
    'gamma interlayer g to rb': 'G→R/B 层间伽马',
    'gamma interlayer b to rg': 'B→R/G 层间伽马',
    'diffusion size um': '扩散尺寸 (µm)',
    'Enable DIR coupler inhibition.': '启用 DIR 成色剂抑制作用。',
    'Global multiplier on the DIR coupler inhibition matrix. 1.0 leaves the per-channel gammas as-is.':
        'DIR 成色剂抑制矩阵的全局系数。1.0 表示保持各通道伽马不变。',
    'Multiplier on the same-layer (diagonal) inhibition. Controls overall contrast / gamma reduction within each RGB layer.':
        '同层(对角)抑制的系数。控制每个 RGB 层内的整体反差/伽马降低。',
    'Multiplier on the cross-layer (off-diagonal) inhibition. Controls saturation enhancement from interlayer DIR effects.':
        '跨层(非对角)抑制的系数。控制层间 DIR 效应带来的饱和度增强。',
    'Per-channel same-layer DIR gamma (R, G, B). Effective gamma reduction of each layer\'s density curve.':
        '各通道的同层 DIR 伽马 (R, G, B)。即每层密度曲线的有效伽马降低量。',
    'DIR inhibition from the R layer onto the G and B layers respectively (g_R->G, g_R->B).':
        'R 层分别对 G、B 层的 DIR 抑制 (g_R→G、g_R→B)。',
    'DIR inhibition from the G layer onto the R and B layers respectively (g_G->R, g_G->B).':
        'G 层分别对 R、B 层的 DIR 抑制 (g_G→R、g_G→B)。',
    'DIR inhibition from the B layer onto the R and G layers respectively (g_B->R, g_B->G).':
        'B 层分别对 R、G 层的 DIR 抑制 (g_B→R、g_B→G)。',
    'Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.':
        '成色剂扩散的 σ,单位微米,典型范围 5–20 µm,控制锐度并影响饱和度。',

    # --- halation ---------------------------------------------------
    'Enable halation and in-emulsion scatter.': '启用光晕和乳剂内散射。',
    'scatter amount': '散射强度',
    'High-level scatter strength. 1.0 = full physical scatter, 0.0 = no scatter. Scales the fraction of light that undergoes in-emulsion scattering.':
        '总体散射强度。1.0 = 完全物理散射,0.0 = 无散射。缩放发生乳剂内散射的光的比例。',
    'scatter spatial scale': '散射空间尺度',
    'High-level scatter size multiplier (1.0 = physical defaults). Scales both core and tail sigmas.':
        '总体散射尺寸倍率(1.0 = 物理默认值)。同时缩放核心与尾部的 σ。',
    'halation amount': '光晕强度',
    'High-level halation strength multiplier (1.0 = physical defaults). Scales the per-channel halation amplitudes.':
        '总体光晕强度倍率(1.0 = 物理默认值)。缩放各通道的光晕幅度。',
    'halation spatial scale': '光晕空间尺度',
    'High-level halation size multiplier (1.0 = physical defaults). Scales the first-bounce sigma.':
        '总体光晕尺寸倍率(1.0 = 物理默认值)。缩放首次反弹的 σ。',
    'boost ev': '高光增强 (EV)',
    'Maximum highlight boost in stops.': '高光增强的上限,以档(EV)为单位。',
    'protect ev': '高光保护 (EV)',
    'Protected range above midgray for the boost onset in stops.':
        '中灰之上受保护的范围,增强从该范围之上才开始,以档(EV)为单位。',
    'boost range': '增强过渡速度',
    'Controls how quickly the highlight boost ramps in, from 0 to 1.':
        '控制高光增强的过渡快慢,范围 0–1。',
    'scatter core um': '散射核心 σ (µm)',
    'Sigma of the scatter core Gaussian per channel [R,G,B], in micrometers. Controls fine-scale sharpness loss in the emulsion.':
        '各通道 [R,G,B] 散射核心高斯的 σ,单位微米。控制乳剂内的细尺度锐度损失。',
    'scatter tail um': '散射尾部衰减 (µm)',
    'Decay constant of the scatter exponential tail per channel [R,G,B], in micrometers (internally approximated by a sum of Gaussians). Controls extended low-level spread within the emulsion.':
        '各通道 [R,G,B] 散射指数尾部的衰减常数,单位微米(内部以高斯和近似)。控制乳剂内更大范围的低强度扩散。',
    'scatter tail weight': '散射尾部权重',
    'Weight of the scatter tail Gaussian per channel [R,G,B] (0-1). Tail weight + core weight = 1. Higher values put more scattered light into the long tail.':
        '各通道 [R,G,B] 散射尾部高斯的权重 (0–1)。尾部权重 + 核心权重 = 1。数值越大,进入长尾的散射光越多。',
    'halation strength': '光晕幅度',
    'Total back-reflection halation amplitude per channel [R,G,B] (0-1). Typical red channel: weak AH 0.02-0.08, no AH 0.08-0.25. The blue channel is usually near zero.':
        '各通道 [R,G,B] 的背反射光晕总幅度 (0–1)。红通道典型值:弱防光晕层 0.02–0.08,无防光晕层 0.08–0.25。蓝通道通常接近零。',
    'halation first sigma um': '首次反弹 σ (µm)',
    'Sigma of the first halation bounce per channel [R,G,B], in micrometers. Set by the base thickness (40-80 um for typical cine/still bases).':
        '各通道 [R,G,B] 首次光晕反弹的 σ,单位微米。由片基厚度决定(典型电影/静态胶片片基为 40–80 µm)。',
    'halation n bounces': '反弹次数',
    'Number of multi-bounce Gaussians summed in the halation pass. Subsequent bounces use sqrt(k)-spaced widths. Typical: 2-3.':
        '光晕计算中累加的多反弹高斯数量。后续反弹宽度按 √k 间距排列。典型值:2–3。',
    'halation bounce decay': '反弹衰减系数',
    'Per-bounce amplitude decay ratio (rho). Physical range 0.3-0.7. Controls how fast the halation energy falls off between bounces.':
        '每次反弹的幅度衰减比 (ρ)。物理范围 0.3–0.7。控制光晕能量在反弹间的衰减速度。',
    'halation renormalize': '光晕归一化',
    'If enabled, divide by (1 + sum of bounce amplitudes) so mid-grey is preserved. If disabled, halation is purely additive and subtly lifts shadows as well as highlights.':
        '启用时除以(1 + 反弹幅度之和),以保持中灰不变。禁用时,光晕为纯叠加,会同时轻微提亮阴影和高光。',

    # --- grain ---------------------------------------------------
    'Add grain to the negative': '向底片添加颗粒',
    'sublayers active': '启用子层',
    'Enable per-sublayer grain structure.': '启用逐子层的颗粒结构。',
    'particle area um2': '颗粒面积 (µm²)',
    'Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.':
        '颗粒面积(µm²),与 ISO 相关。约为 ISO 100 对应 0.1、ISO 200 对应 0.1、ISO 400 对应 0.4,依此类推。',
    'particle scale': 'RGB 层颗粒尺度',
    'Scale of particle area for the RGB layers, multiplies particle_area_um2':
        'RGB 各层颗粒面积的缩放系数,与 particle_area_um2 相乘。',
    'particle scale layers': '子层颗粒尺度',
    'Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2':
        '每个颜色层内各子层颗粒面积的缩放系数,与 particle_area_um2 相乘。',
    'density min': '最小密度',
    'Minimum density of the grain, typical values (0.03-0.06)':
        '颗粒的最小密度,典型值 (0.03–0.06)。',
    'uniformity': '均匀度',
    'Uniformity of the grain, typical values (0.94-0.98)':
        '颗粒的均匀度,典型值 (0.94–0.98)。',
    'blur': '模糊',
    'Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).':
        '颗粒的高斯模糊 σ(像素)。高倍放大时应增大;高分辨率下应为 0.8–0.9,较低分辨率可降至 0.6。',
    'blur dye clouds um': '染料云模糊 (µm)',
    'Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)':
        '染料云高斯模糊 σ 的缩放(微米),用于高倍放大(默认 1)。',
    'micro structure': '微观结构',
    'Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.':
        '分子级团簇导致的微观结构参数,[微观结构模糊 σ / 极限光学分辨率(默认 0.10 µm),分子团簇尺寸(纳米,默认 30 nm)]。仅用于极端放大倍率。',

    # --- glare ---------------------------------------------------
    'Add glare to the print': '向放印照片添加眩光',
    'percent': '比例',
    'Percentage of the glare light (typically 0.1-0.25)': '眩光光量的比例(通常 0.1–0.25)。',
    'roughness': '粗糙度',
    'Roughness of the glare light (0-1)': '眩光的粗糙度 (0–1)。',
    'Sigma of gaussian blur in pixels for the glare': '眩光的高斯模糊 σ(像素)。',

    # --- chemistry ---------------------------------------------------
    'Enable print density-curve morphing.': '启用相纸密度曲线变形。',
    'gamma factor': '伽马系数',
    'Global coupled gamma multiplier for the print density curves.':
        '相纸密度曲线的全局耦合伽马系数。',
    'gamma factor fast': '快速子层伽马系数',
    'Gamma factor applied to the fast sub-layer.': '应用于快速子层的伽马系数。',
    'gamma factor slow': '中慢子层伽马系数',
    'Gamma factor applied to the mid and slow sub-layers.': '应用于中速与慢速子层的伽马系数。',
    'gamma factor red': '红通道伽马系数',
    'Per-channel gamma factor applied to the red channel.': '应用于红通道的逐通道伽马系数。',
    'gamma factor green': '绿通道伽马系数',
    'Per-channel gamma factor applied to the green channel.': '应用于绿通道的逐通道伽马系数。',
    'gamma factor blue': '蓝通道伽马系数',
    'Per-channel gamma factor applied to the blue channel.': '应用于蓝通道的逐通道伽马系数。',
    'developer exhaustion': '显影液衰竭',
    'Blend all three print sub-layers toward the matched Gumbel shoulder while preserving midgray via a common horizontal offset.':
        '将三个相纸子层向匹配的 Gumbel 肩部混合,并通过共同的水平偏移保持中灰不变。',

    # --- preflash ---------------------------------------------------
    'exposure': '曝光量',
    'Preflash exposure value in ev for the print': '相纸预闪曝的曝光量(EV)。',
    'y filter shift': 'Y 滤镜偏移',
    'Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units':
        '预闪曝时放大机 Y(黄)滤镜相对中性位置的偏移,柯达 CC 单位,典型值 (−20–20)。',
    'm filter shift': 'M 滤镜偏移',
    'Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units':
        '预闪曝时放大机 M(品红)滤镜相对中性位置的偏移,柯达 CC 单位,典型值 (−20–20)。',

    # --- diffusion filters ---------------------------------------------------
    'Toggle the diffusion filter (Pro-Mist family).': '开关柔光镜(Pro-Mist 系列)。',
    'filter family': '滤镜家族',
    'PSF family. pro_mist / glimmerglass / cinebloom are transparent (energy-preserving); black_pro_mist absorbs a fraction of the deflected light, lifting shadows by reducing local contrast.':
        '点扩散函数家族。pro_mist / glimmerglass / cinebloom 为透明型(能量守恒);black_pro_mist 会吸收一部分偏折光,通过降低局部反差来提亮阴影。',
    'strength': '强度',
    'Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2. Maps internally to the (p_s, p_a) deflected/absorbed photon fractions.':
        '市售滤镜档位:0、1/8=0.125、1/4=0.25、1/2=0.5、1、2。内部映射为偏折/吸收光子比例 (p_s, p_a)。',
    'spatial scale': '空间尺度',
    'Multiplier on the image-plane PSF widths (all per-group lambdas). Adjust for image-format / print-size differences.':
        '像面 PSF 宽度(各组 λ)的倍率。可针对画幅/放印尺寸差异进行调整。',
    'halo warmth': '光晕冷暖偏移',
    'Additive offset on the family\'s halo warmth axis. Positive = warm outer halo / cool inner halo. Energy-preserving per channel. 0 = use family default.':
        '该家族光晕冷暖轴上的附加偏移。正值 = 外圈光晕偏暖 / 内圈偏冷。逐通道能量守恒。0 = 使用家族默认值。',
    'core intensity': '核心强度',
    'Advanced. Multiplier on the core weight; the three group weights are renormalized to sum to 1. 1.0 = use family default.':
        '高级。核心权重的倍率;三组权重会重新归一化为总和 1。1.0 = 使用家族默认值。',
    'core size': '核心尺寸',
    'Advanced. Multiplier on the core lambda. 1.0 = use family default.':
        '高级。核心 λ 的倍率。1.0 = 使用家族默认值。',
    'halo intensity': '光晕强度',
    'Advanced. Multiplier on the halo weight; the three group weights are renormalized to sum to 1. 1.0 = use family default.':
        '高级。光晕权重的倍率;三组权重会重新归一化为总和 1。1.0 = 使用家族默认值。',
    'halo size': '光晕尺寸',
    'Advanced. Multiplier on the halo lambda. 1.0 = use family default.':
        '高级。光晕 λ 的倍率。1.0 = 使用家族默认值。',
    'bloom intensity': '泛光强度',
    'Advanced. Multiplier on the bloom weight; the three group weights are renormalized to sum to 1. 1.0 = use family default.':
        '高级。泛光权重的倍率;三组权重会重新归一化为总和 1。1.0 = 使用家族默认值。',
    'bloom size': '泛光尺寸',
    'Advanced. Multiplier on the bloom lambda. 1.0 = use family default.':
        '高级。泛光 λ 的倍率。1.0 = 使用家族默认值。',

    # --- camera ---------------------------------------------------
    'film format mm': '胶片画幅 (mm)',
    'Long edge of the film format in millimeters, e.g. 8, 16, 35, 60, 120.':
        '胶片画幅的长边,单位毫米,如 8、16、35、60、120。',
    'lens blur um': '镜头模糊 (µm)',
    'Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses; used for sharp input simulations without lens blur.':
        '相机镜头模糊的高斯 σ(微米)。普通镜头约 5 µm,高素质镜头可低至 2–4 µm;用于模拟不带镜头模糊的锐利输入。',
    'auto exposure method': '自动曝光方式',
    'Metering method used by the virtual camera\'s auto-exposure.':
        '虚拟相机自动曝光所用的测光方式。',
    'auto exposure': '自动曝光',
    'Use the auto-exposure feature of the virtual camera.':
        '使用虚拟相机的自动曝光功能。',
    'exposure compensation ev': '曝光补偿 (EV)',
    'Add a bias to the auto-exposure of the camera.':
        '为相机的自动曝光结果添加偏置。',

    # --- scanner ---------------------------------------------------
    'lens blur': '镜头模糊',
    'Sigma of gaussian filter in pixel for the scanner lens blur.':
        '扫描仪镜头模糊的高斯 σ(像素)。',
    'white correction': '白场校正',
    'Enable white point correction applied to the scanner output.':
        '启用作用于扫描输出的白场校正。',
    'white level': '白场电平',
    'Target white level applied when white correction is enabled.':
        '启用白场校正时应用的目标白场电平。',
    'black correction': '黑场校正',
    'Enable black point correction applied to the scanner output.':
        '启用作用于扫描输出的黑场校正。',
    'black level': '黑场电平',
    'Target black level applied when black correction is enabled.':
        '启用黑场校正时应用的目标黑场电平。',
    'unsharp mask': '锐化遮罩',
    'Apply unsharp mask to the scan, [sigma in pixel, amount].':
        '对扫描结果应用锐化遮罩,[σ(像素), 强度]。',

    # --- gamut compression ---------------------------------------------------
    'Compress input chromaticities toward the visible spectral locus before spectral upsampling. Off passes input through unchanged.':
        '在光谱上采样之前,将输入色度坐标向可见光谱轨迹压缩。关闭则原样传递输入。',
    'algorithm': '算法',
    'xy: radial compression in CIE 1931 chromaticity (ACES RGC family, hue-preserving). oklch: perceptual chroma reduction at constant Oklch (L, h).':
        'xy:在 CIE 1931 色品图中沿径向压缩(ACES RGC 系列,保持色相)。oklch:在 Oklch (L, h) 不变的前提下做感知彩度降低。',
    'knee': '拐点 (knee)',
    'Reinhard knee (threshold, limit, power). Defaults (0.0, 1.0, 6.0) apply a soft full-range roll-off that asymptotes at the spectral locus boundary.':
        'Reinhard 拐点(阈值、上限、幂次)。默认 (0.0, 1.0, 6.0) 在整个范围内软滚降,渐近于光谱轨迹边界。',
    'off: disable output gamut compression. cam16ucs (default): CAM16-UCS chroma reduction with the smoothest constant-hue behavior. oklch: perceptual chroma reduction in OkLab, preserves hue + lightness. aces_rgc: per-channel ACES RGC v1.3, matches Resolve/Nuke/OCIO. oklrab/jzazbz: alternative perceptual spaces.':
        'off:关闭输出色域压缩。cam16ucs(默认):CAM16-UCS 彩度降低,恒定色相行为最平滑。oklch:Oklab 感知彩度降低,保持色相与明度。aces_rgc:逐通道 ACES RGC v1.3,与 Resolve/Nuke/OCIO 一致。oklrab/jzazbz:备选感知空间。',
    'Reinhard knee (threshold, limit, power) on normalized chroma C/C_max. Default (0.0, 1.0, 6.0) rolls off smoothly from the center with the asymptote on the output cube edge.':
        '作用于归一化彩度 C/C_max 的 Reinhard 拐点(阈值、上限、幂次)。默认 (0.0, 1.0, 6.0) 从中心平滑滚降,渐近于输出色域立方体边界。',

    # --- input / crop / spectral ---------------------------------------------------
    'input color space': '输入色彩空间',
    'Color space of the input image, will be internally converted to sRGB and negative values clipped':
        '输入图像的色彩空间,内部会转换为 sRGB,负值将被裁剪。',
    'apply cctf decoding': '应用 CCTF 解码',
    'Apply the inverse cctf transfer function of the color space':
        '应用该色彩空间的逆 CCTF 传递函数。',
    'upscale factor': '放大倍数',
    'Scale image size up to increase resolution': '放大图像尺寸以提高分辨率。',
    'crop': '裁剪',
    'Crop image to a fraction of the original size to preview details at full scale':
        '将图像裁剪为原始尺寸的一部分,以全尺寸预览细节。',
    'crop center': '裁剪中心',
    'Center of the crop region in relative coordinates in x, y (0-1)':
        '裁剪区域中心的相对坐标 x, y (0–1)。',
    'crop size': '裁剪尺寸',
    'Normalized size of the crop region in x, y (0,1), as fraction of the long side.':
        '裁剪区域在 x, y 上的归一化尺寸 (0–1),以长边的比例表示。',
    'Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).':
        '图像光谱分辨率的上采样方法。hanatos2025 覆盖完整可见光谱轨迹;mallett2019 仅适用于 sRGB(会裁剪输入)。',
    'hanatos2025 adaptation window': 'hanatos2025 适应窗口',
    'Apply the hanatos2025 bandpass adaptation window when reconstructing spectra.':
        '重建光谱时应用 hanatos2025 带通适应窗口。',
    'hanatos2025 adaptation surface': 'hanatos2025 适应曲面',
    'Apply the hanatos2025 surface adaptation polynomial when reconstructing spectra.':
        '重建光谱时应用 hanatos2025 曲面适应多项式。',
    'spectral gaussian blur': '光谱高斯模糊',
    'Sigma in nm for Gaussian blur applied to reconstructed spectra.':
        '应用于重建光谱的高斯模糊 σ(纳米)。',

    # --- special / tune ---------------------------------------------------
    'film channel swap': '胶片通道交换',
    'print channel swap': '相纸通道交换',
    'film gamma factor': '胶片伽马系数',
    'Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast':
        '底片密度曲线的伽马系数,< 1 降低反差,> 1 提高反差。',

    # --- display ---------------------------------------------------
    'use display transform': '使用显示器色彩变换',
    'Use Pillow.ImageCms to retrive the display transform (only in Windows) and apply it to the napari viewer output, if disabled the output color space is used':
        '使用 Pillow.ImageCms 获取显示器色彩变换(仅限 Windows)并应用到 napari 视图输出;禁用时使用输出色彩空间。',
    'gray 18% canvas': '18% 灰背景',
    'Use neutral 18% gray as backgroung to judge the exposure and neutral colors':
        '使用中性 18% 灰作为背景,便于判断曝光与中性色。',
    'white padding': '白色边距',
    'Expand the white border layer around the normalized preview frame, expressed as a fraction of the image long edge.':
        '扩展归一化预览画面四周的白色边框,以图像长边的比例表示。',
    'preview max size': '预览最大尺寸',
    'max size of the long edge of the preview image in pixels':
        '预览图像长边的最大尺寸(像素)。',
    'output interpolation': '输出插值方式',
    'Napari interpolation mode used to display the output layer in the viewer.':
        '视图中显示输出图层所用的 napari 插值方式。',

    # --- raw import ---------------------------------------------------
    'white balance': '白平衡',
    'Select white balance settings, if custom you can tune temperature and tint':
        '选择白平衡设置;选择 custom(自定义)时可调整色温与色调。',
    'temperature': '色温',
    'Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings':
        '自定义白平衡的色温(开尔文),其他白平衡模式下不使用。',
    'tint': '色调',
    'Tint value for the custom white balance, not used for the other white balance settings':
        '自定义白平衡的色调值,其他白平衡模式下不使用。',
    'lens correction': '镜头校正',
    'Apply lens corrections': '应用镜头校正。',
    'Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state':
        '使用 rawpy 载入并处理 RAW 文件,输出色彩空间与 CCTF 按当前输入界面的设置。',

    # --- simulation panel ---------------------------------------------------
    'film profile': '胶片档案',
    'Film stock to simulate': '要模拟的胶片型号。',
    'print profile': '相纸档案',
    'Print paper to simulate': '要模拟的相纸型号。',
    'print illuminant': '放印光源',
    'Print illuminant to simulate': '要模拟的放印光源。',
    'print auto compensation': '放印自动补偿',
    'Auto adjust the print exposure for the camera exposure compensation ev':
        '根据相机曝光补偿 (EV) 自动调整放印曝光。',
    'print exposure': '放印曝光',
    'Changes the exposure time set in the virtual enlarger':
        '更改虚拟放大机中设置的曝光时间。',
    'print y filter shift': '放印 Y 滤镜偏移',
    'Y filter shift of the color enlarger from a neutral position, in Kodak CC units':
        '彩色放大机 Y(黄)滤镜相对中性位置的偏移,柯达 CC 单位。',
    'print m filter shift': '放印 M 滤镜偏移',
    'M filter shift of the color enlarger from a neutral position, in Kodak CC units':
        '彩色放大机 M(品红)滤镜相对中性位置的偏移,柯达 CC 单位。',
    'output color space': '输出色彩空间',
    'Output color space of the simulation': '仿真的输出色彩空间。',
    'saving color space': '保存色彩空间',
    'Color space of the saved image file': '保存图像文件所用的色彩空间。',
    'saving cctf encoding': '保存时编码 CCTF',
    'Add or not the CCTF to the saved image file': '是否为保存的图像文件编码 CCTF。',
    'auto preview': '自动预览',
    'trigger the preview after every change of gui parameters, use mouse scrollwheel on parameters field, read preview tooltip for details':
        '界面参数每次变化后自动触发预览;可在参数输入框上使用鼠标滚轮微调,详见预览按钮的工具提示。',
    'scan film': '扫描底片',
    'Show a scan of the negative instead of the print':
        '显示底片的扫描结果,而非放印照片。',
    'scan for print': '为放印扫描',
    'Scan the image for print, ie white and black correction of the scanner are active, and glare is deactivated.':
        '以放印为目的扫描图像:启用扫描仪的白场/黑场校正,并停用眩光。',
    'run the simulation on a small preview and deactivates grain, halation, blurs, unsharp mask (diffusion filters are active)':
        '在小尺寸预览上运行仿真,并停用颗粒、光晕、各类模糊和锐化遮罩(柔光镜仍然生效)。',
    'Run the full simulation on the full-resolution input':
        '在原始分辨率的输入上运行完整仿真。',
    'Save the current output layer to an image file':
        '将当前输出图层保存为图像文件。',

    # --- viewer toolbar ---------------------------------------------------
    'Pixel of the screen mapped 1 to 1 to the image pixel':
        '屏幕像素与图像像素 1:1 对应。',
    '2 screen pixels mapped to 1 image pixel':
        '2 个屏幕像素对应 1 个图像像素。',
    '4 screen pixels mapped to 1 image pixel':
        '4 个屏幕像素对应 1 个图像像素。',

    # --- status bar and dialogs ---------------------------------------------------
    'ready': '就绪',
    'Loading raw...': '正在载入 RAW…',
    'Load raw': '载入 RAW',
    'Failed to load RAW image.': '载入 RAW 图像失败。',
    'Load raw failed': '载入 RAW 失败',
    'Loaded raw': '已载入 RAW',
    'Loaded raw, lens correction not applied': '已载入 RAW,未应用镜头校正',
    'Loaded raw and applied lens correction': '已载入 RAW 并应用镜头校正',
    'Display transform: disabled': '显示器色彩变换:已禁用',
    'Display transform: no display profile, using raw preview':
        '显示器色彩变换:未找到显示器配置文件,使用原始预览',
    'Display transform: display profile found': '显示器色彩变换:已找到显示器配置文件',
    'Display transform unavailable: no display profile detected, disabled':
        '显示器色彩变换不可用:未检测到显示器配置文件,已禁用',
    'Simulation already running': '仿真已在运行',
    'Run simulation': '运行仿真',
    'Load an input image before running the simulation.':
        '请先载入输入图像,再运行仿真。',
    'Computing preview...': '正在计算预览…',
    'Computing scan...': '正在计算扫描…',
    'Preview completed.': '预览完成。',
    'Scan completed.': '扫描完成。',
    'Simulation failed.': '仿真失败。',
    'Preview failed': '预览失败',
    'Scan failed': '扫描失败',
    'Simulation failed': '仿真失败',
    'Save output': '保存输出',
    'Run a simulation before saving the output layer.':
        '请先运行仿真,再保存输出图层。',
    'Save output image': '保存输出图像',
    'Failed to save output image.': '保存输出图像失败。',
    'Saved output image to': '输出图像已保存到',
    'but failed to copy metadata': '但复制元数据失败',
    'Save current as default': '保存为默认',
    'Failed to save default GUI state.': '保存默认界面状态失败。',
    'Saved current GUI state as the startup default': '已将当前界面状态保存为启动默认值',
    'Save GUI state': '保存界面状态',
    'Failed to save GUI state.': '保存界面状态失败。',
    'Saved GUI state to': '界面状态已保存到',
    'Load GUI state': '载入界面状态',
    'Failed to load GUI state.': '载入界面状态失败。',
    'Loaded GUI state from': '界面状态已从以下位置载入:',
    'Restore factory default': '恢复出厂默认',
    'Failed to clear the saved startup default.': '清除已保存的启动默认值失败。',
    'Restored factory default GUI state': '已恢复出厂默认界面状态',
}


def _detect_system_language() -> str:
    """Best-effort system language code, e.g. ``'zh_Hans_CN'``.

    On macOS the locale environment variables are unreliable (often unset
    for GUI-launched processes, and the C library may resolve a region
    rather than the UI language), so query the user's AppleLocale
    preference first — the same source Qt consults. Elsewhere honor the
    locale environment.
    """

    if sys.platform == 'darwin':
        import subprocess

        try:
            return subprocess.check_output(
                ['defaults', 'read', '-g', 'AppleLocale'],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass

    import locale

    for getter in (locale.getlocale, locale.getdefaultlocale):
        try:
            code = getter()[0]
        except ValueError:
            code = None
        if code:
            return str(code)
    return ''


_cached_system_language: str | None = None


def _system_language() -> str:
    global _cached_system_language
    if _cached_system_language is None:
        _cached_system_language = _detect_system_language()
    return _cached_system_language


def translations_enabled() -> bool:
    value = os.environ.get('SPEKTRAFILM_LANG', 'auto').strip().lower()
    if value in {'en', 'eng', 'english', 'off', 'false', 'none', '0'}:
        return False
    if value in {'zh', 'zh_cn', 'zh-cn', 'zh_hans', 'chinese', 'cn', 'on', 'true', '1'}:
        return True
    return _system_language().lower().replace('-', '_').startswith('zh')


def tr(text: str) -> str:
    """Return the Chinese rendering of ``text`` when translations are on.

    Unknown strings pass through unchanged, so untranslated additions
    simply stay in English.
    """

    if not text or not translations_enabled():
        return text
    return TRANSLATIONS.get(text, text)
