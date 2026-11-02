# 授权、权利及真实性说明

- 上游 SpektraLab: https://github.com/JamesQiu2005/SpektraLab ， GPL-3.0-or-later。若直接复制、修改或移植其渲染器源码并分发自己的软件，可能需要依GPL提供相应源码、版权及许可证声明等（具体义务取决于你如何集成及分发；请由开发者核查）。
- 上游项目说明部分胶片测色模型、28份实测数据和LUT来自 `spektrafilm`，采用CC BY-SA 4.0，本包**不含这些数据**。
- 本包SVG、PNG、JSON与生成器是围绕公开物理尺寸和已公开的胶片印字信息**重新组织的工程示例**，不是复制 SpektraLab 的二进制或源代码，也不依赖其光谱模拟引擎。
- 数据库型DX号码不是全部通过实片验证，所以本包区分 `read_from_strip` 与 `database_unverified`；**未经核实的DX默认不显示**。
- 品牌型号名称涉及各公司的商标。本包仅在摄影胶片识别与互操作场景作描述性使用，不代表品牌授权或合作。
- 照片显示边框的近似美术样式，不等同于“原厂家每一批次底片的准确字形与位置”。需要真正的100%准确度，应收集特定生产批次的实际扫描，并在使用前核实版权和授权。
- 不在本包中复制 GPL 源码或第三方受 CC BY-SA 保护的实测数据。若后续决定使用上游代码，请另外按授权处理，而不是把这个包的独立示例与上游代码混称为“无版权限制”。

权威链接： https://spektralab.app/ 、 https://github.com/JamesQiu2005/SpektraLab/blob/main/LICENSE.txt 、 https://spektralab.app/user_guide/ 。
