from __future__ import annotations

from datetime import datetime

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.film_frame import (
    DEFAULT_OUTPUT_LONG_EDGE,
    FILM_FRAME_TEMPLATES,
    FilmFrameOptions,
    compose_film_frame,
    detect_image_aspect,
    get_film_frame_template,
    output_size,
    resize_rgb_image,
)

_PREVIEW_LONG_EDGE = 560
_PREVIEW_SOURCE_MAX = 1400
_RESOLUTIONS = (
    ('1900 px（小）', 1900),
    ('2850 px', 2850),
    ('3800 px（模板原始尺寸）', DEFAULT_OUTPUT_LONG_EDGE),
    ('5700 px（放大）', 5700),
)


class FilmFrameDialog(QtWidgets.QDialog):
    def __init__(
        self,
        image: np.ndarray,
        *,
        film_stock_id: str,
        date_label: str | None = None,
        frame_number: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image = np.asarray(image, dtype=np.float32)
        self.setWindowTitle('添加胶片边框')
        self.setMinimumWidth(1060)

        height, width = self._image.shape[:2]
        longest = max(height, width)
        if longest > _PREVIEW_SOURCE_MAX:
            ratio = _PREVIEW_SOURCE_MAX / longest
            self._preview_source = resize_rgb_image(
                self._image,
                (max(1, round(width * ratio)), max(1, round(height * ratio))),
            )
        else:
            self._preview_source = self._image

        aspect = detect_image_aspect(self._image)
        orientation_name = '横向' if aspect.orientation == 'landscape' else '竖向'
        self.orientation_label = QtWidgets.QLabel(
            f'原图：{aspect.width} × {aspect.height} px（{aspect.ratio:.3f}:1），{orientation_name}；'
            f'自动使用套装中的{orientation_name}模板'
        )
        self.orientation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.orientation_label.setWordWrap(True)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setFixedSize(_PREVIEW_LONG_EDGE + 40, _PREVIEW_LONG_EDGE + 40)
        self.preview_label.setStyleSheet('background: #202124; border: 1px solid #444;')

        self.template_combo = QtWidgets.QComboBox()
        for template in FILM_FRAME_TEMPLATES:
            self.template_combo.addItem(template.ui_name, template.template_id)
        self.template_combo.setCurrentIndex(
            self.template_combo.findData(get_film_frame_template(film_stock_id).template_id)
        )
        self.template_status_label = QtWidgets.QLabel()
        self.template_status_label.setWordWrap(True)

        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItem('完整适配（保留整张照片）', 'contain')
        self.fit_mode_combo.addItem('等比裁切（填满照片窗）', 'cover')
        self.fit_mode_combo.setToolTip('非 3:2 / 2:3 照片可选择完整适配或等比裁切，照片不会被拉伸。')

        self.realistic_checkbox = QtWidgets.QCheckBox('真实质感（孔边柔光、片基颗粒）')
        self.realistic_checkbox.setToolTip('仅为扫描质感后处理：不改变模板的布局、文字、齿孔位置和数量。预览与导出一致。')
        self.letterbox_combo = QtWidgets.QComboBox()
        self.letterbox_combo.addItem('胶片黑（默认）', None)
        self.letterbox_combo.addItem('白色', (255, 255, 255))
        self.letterbox_combo.addItem('中性灰', (128, 128, 128))
        self.letterbox_combo.setToolTip('完整适配时照片窗内的留白底色。')

        self.pan_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pan_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for slider in (self.pan_x_slider, self.pan_y_slider):
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setToolTip('等比裁切时，调整照片在照片窗内的裁切位置。')

        self.resolution_combo = QtWidgets.QComboBox()
        for label, long_edge in _RESOLUTIONS:
            self.resolution_combo.addItem(label, long_edge)
        self.resolution_combo.setCurrentIndex(self.resolution_combo.findData(DEFAULT_OUTPUT_LONG_EDGE))
        self.resolution_label = QtWidgets.QLabel()

        self.frame_number_checkbox = QtWidgets.QCheckBox('显示画幅编号（取消勾选即隐藏模板示例编号）')
        self.frame_number_checkbox.setChecked(True)
        self.frame_number_edit = QtWidgets.QLineEdit()
        self.frame_number_edit.setMaxLength(4)
        self.frame_number_edit.setToolTip('模板内的 12 / 12A 只是示例编号；这里可改为自己的画幅编号。带 DX 条码的模板会同步重绘条码帧位（仅 0–63 的数字），隐藏或非数字编号时条码一并隐藏。')

        self.date_checkbox = QtWidgets.QCheckBox('添加日期戳（默认关闭，叠加在照片上，不属于胶片印字）')
        self.date_checkbox.setChecked(False)
        self.date_label_edit = QtWidgets.QLineEdit(
            date_label if date_label is not None else datetime.now().strftime("'%y %m %d"),
        )
        self.date_position_combo = QtWidgets.QComboBox()
        for label, position in (
            ('左上', 'top_left'),
            ('右上', 'top_right'),
            ('左下', 'bottom_left'),
            ('右下', 'bottom_right'),
        ):
            self.date_position_combo.addItem(label, position)
        self.date_position_combo.setCurrentIndex(self.date_position_combo.findData('bottom_right'))

        form = QtWidgets.QFormLayout()
        form.addRow('胶片模板', self.template_combo)
        form.addRow('模板来源', self.template_status_label)
        form.addRow('照片适配', self.fit_mode_combo)
        form.addRow('留白底色', self.letterbox_combo)
        form.addRow('裁切位置 左右', self.pan_x_slider)
        form.addRow('裁切位置 上下', self.pan_y_slider)
        form.addRow('画幅编号', self.frame_number_edit)
        form.addRow('日期内容', self.date_label_edit)
        form.addRow('日期位置', self.date_position_combo)
        form.addRow('导出分辨率', self.resolution_combo)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('导出')
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.template_status_label.setMinimumWidth(360)
        self.template_status_label.setMaximumHeight(110)
        self.orientation_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.preview_label, 0, QtCore.Qt.AlignCenter)
        left.addStretch(1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.orientation_label)
        right.addLayout(form)
        right.addWidget(self.frame_number_checkbox)
        right.addWidget(self.date_checkbox)
        right.addWidget(self.realistic_checkbox)
        right.addWidget(self.resolution_label)
        right.addStretch(1)
        right.addWidget(self.buttons)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left)
        layout.addLayout(right, 1)

        self._initial_frame_number = frame_number
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.fit_mode_combo.currentIndexChanged.connect(self._update_preview)
        self.letterbox_combo.currentIndexChanged.connect(self._update_preview)
        self.realistic_checkbox.toggled.connect(self._update_preview)
        self.date_position_combo.currentIndexChanged.connect(self._update_preview)
        self.resolution_combo.currentIndexChanged.connect(self._update_preview)
        for slider in (self.pan_x_slider, self.pan_y_slider):
            slider.valueChanged.connect(self._update_preview)
        for checkbox in (self.date_checkbox, self.frame_number_checkbox):
            checkbox.toggled.connect(self._update_preview)
        for editor in (self.date_label_edit, self.frame_number_edit):
            editor.textChanged.connect(self._update_preview)

        self._on_template_changed()

    @property
    def template_id(self) -> str:
        return str(self.template_combo.currentData())

    @property
    def frame_number(self) -> str | None:
        template = get_film_frame_template(self.template_id)
        if not template.has_frame_number:
            return None
        return self.frame_number_edit.text().strip()

    @property
    def date_label(self) -> str:
        return self.date_label_edit.text()

    @property
    def options(self) -> FilmFrameOptions:
        return FilmFrameOptions(
            fit_mode=self.fit_mode_combo.currentData(),
            pan_x=self.pan_x_slider.value() / 100.0,
            pan_y=self.pan_y_slider.value() / 100.0,
            show_date=self.date_checkbox.isChecked(),
            date_position=self.date_position_combo.currentData(),
            show_frame_number=self.frame_number_checkbox.isChecked() and bool(self.frame_number_edit.text().strip()),
            output_long_edge=int(self.resolution_combo.currentData()),
            letterbox_color=self.letterbox_combo.currentData(),
            realistic=self.realistic_checkbox.isChecked(),
        )

    def _on_template_changed(self, *_args) -> None:
        template = get_film_frame_template(self.template_id)
        label = f'边缘印字“{template.edge_label}”；' if template.edge_label else '无品牌、无印字；'
        self.template_status_label.setText(
            f'{label}仅为参考版本，不代表该型号所有批次。{template.source}'
        )
        self.frame_number_checkbox.setEnabled(template.has_frame_number)
        self.frame_number_edit.setEnabled(template.has_frame_number)
        with QtCore.QSignalBlocker(self.frame_number_edit):
            self.frame_number_edit.setText(
                self._initial_frame_number or template.default_frame_number
            )
        self._update_preview()

    def _update_preview(self, *_args) -> None:
        cover = self.fit_mode_combo.currentData() == 'cover'
        self.pan_x_slider.setEnabled(cover)
        self.pan_y_slider.setEnabled(cover)
        orientation = detect_image_aspect(self._image).orientation
        width, height = output_size(orientation, int(self.resolution_combo.currentData()))
        self.resolution_label.setText(f'导出尺寸：{width} × {height} px')
        framed = compose_film_frame(
            self._preview_source,
            template_id=self.template_id,
            date_label=self.date_label,
            frame_number=self.frame_number,
            options=self._preview_options(),
        )
        pixels = np.ascontiguousarray(np.clip(framed * 255.0 + 0.5, 0, 255).astype(np.uint8))
        image = QtGui.QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def _preview_options(self) -> FilmFrameOptions:
        options = self.options
        return FilmFrameOptions(
            fit_mode=options.fit_mode,
            pan_x=options.pan_x,
            pan_y=options.pan_y,
            show_date=options.show_date,
            date_position=options.date_position,
            show_frame_number=options.show_frame_number,
            output_long_edge=_PREVIEW_LONG_EDGE,
            letterbox_color=options.letterbox_color,
            realistic=options.realistic,
        )
