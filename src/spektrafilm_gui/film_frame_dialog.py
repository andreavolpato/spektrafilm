from __future__ import annotations

from datetime import datetime

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.film_frame import (
    FILM_FRAME_TEMPLATES,
    FilmFrameOptions,
    compose_film_frame,
    detect_image_aspect,
    get_film_frame_template,
    resize_rgb_image,
)


class FilmFrameDialog(QtWidgets.QDialog):
    def __init__(
        self,
        image: np.ndarray,
        *,
        film_stock_id: str,
        date_label: str | None = None,
        frame_number: str = '74',
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image = np.asarray(image, dtype=np.float32)
        self.setWindowTitle('添加胶片边框')
        self.setMinimumWidth(720)

        aspect = detect_image_aspect(self._image)
        orientation_name = '横向' if aspect.orientation == 'landscape' else '竖向'
        self.orientation_label = QtWidgets.QLabel(
            f'原图：{aspect.width} × {aspect.height} px（{aspect.ratio:.3f}:1），{orientation_name}；'
            '中央照片窗为 3:2，胶片整体随照片方向旋转'
        )
        self.orientation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.orientation_label.setWordWrap(True)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setFixedSize(600, 460)
        self.preview_label.setStyleSheet('background: #202124; border: 1px solid #444;')

        self.border_width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.border_width_slider.setRange(8, 24)
        self.border_width_slider.setValue(15)
        self.border_width_value = QtWidgets.QLabel('15%')
        border_row = QtWidgets.QHBoxLayout()
        border_row.addWidget(self.border_width_slider, 1)
        border_row.addWidget(self.border_width_value)

        self.template_combo = QtWidgets.QComboBox()
        for template in FILM_FRAME_TEMPLATES:
            self.template_combo.addItem(template.display_name, template.template_id)
        template_ids = [self.template_combo.itemData(index) for index in range(self.template_combo.count())]
        selected_template = get_film_frame_template(film_stock_id)
        if selected_template.template_id not in template_ids:
            self.template_combo.addItem(f'当前 profile：{film_stock_id}（边缘标记待核实）', film_stock_id)
        self.template_combo.setCurrentIndex(
            self.template_combo.findData(selected_template.template_id)
            if selected_template.template_id in template_ids
            else self.template_combo.count() - 1
        )
        self.template_status_label = QtWidgets.QLabel()
        self.template_status_label.setWordWrap(True)

        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItem('留白适配（完整保留照片）', 'contain')
        self.fit_mode_combo.addItem('裁切填充（居中裁切）', 'cover')
        self.fit_mode_combo.setToolTip('非 3:2 照片可选择保留完整画面或裁切填满 3:2 照片窗，不会拉伸。')

        self.date_checkbox = QtWidgets.QCheckBox('添加用户日期标记（默认关闭）')
        self.date_checkbox.setChecked(False)
        self.date_label_edit = QtWidgets.QLineEdit(
            date_label if date_label is not None else datetime.now().strftime('%y %m %d'),
        )
        self.date_label_edit.setToolTip('这是后期添加到照片上的日期标记，不属于胶卷制造商边缘印字。')
        self.date_position_combo = QtWidgets.QComboBox()
        for label, position in (
            ('左上', 'top_left'),
            ('右上', 'top_right'),
            ('左下', 'bottom_left'),
            ('右下', 'bottom_right'),
        ):
            self.date_position_combo.addItem(label, position)
        self.date_position_combo.setCurrentIndex(self.date_position_combo.findData('bottom_right'))

        self.frame_number_checkbox = QtWidgets.QCheckBox('显示用户画格编号（浅色）')
        self.frame_number_checkbox.setChecked(True)
        self.frame_number_edit = QtWidgets.QLineEdit(frame_number)
        self.frame_number_edit.setMaxLength(8)
        self.frame_number_edit.setToolTip('仅在具有可辨认参考布局的模板中显示；编号由用户设置。')

        form = QtWidgets.QFormLayout()
        form.addRow('边框宽度', self._widget_with_layout(border_row))
        form.addRow('胶片模板', self.template_combo)
        form.addRow('参考状态', self.template_status_label)
        form.addRow('照片适配', self.fit_mode_combo)
        form.addRow('日期内容', self.date_label_edit)
        form.addRow('日期位置', self.date_position_combo)
        form.addRow('用户画格编号', self.frame_number_edit)

        toggles = QtWidgets.QHBoxLayout()
        toggles.addWidget(self.date_checkbox)
        toggles.addWidget(self.frame_number_checkbox)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('导出')
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.orientation_label)
        layout.addWidget(self.preview_label, 0, QtCore.Qt.AlignCenter)
        layout.addLayout(form)
        layout.addLayout(toggles)
        layout.addWidget(self.buttons)

        self.border_width_slider.valueChanged.connect(self._update_preview)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        for widget in (
            self.fit_mode_combo,
            self.date_position_combo,
        ):
            widget.currentIndexChanged.connect(self._update_preview)
        for checkbox in (self.date_checkbox, self.frame_number_checkbox):
            checkbox.toggled.connect(self._update_preview)
        for editor in (self.date_label_edit, self.frame_number_edit):
            editor.textChanged.connect(self._update_preview)

        self._on_template_changed()

    @staticmethod
    def _widget_with_layout(layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    @property
    def template_id(self) -> str:
        return str(self.template_combo.currentData())

    @property
    def options(self) -> FilmFrameOptions:
        return FilmFrameOptions(
            border_width=self.border_width_slider.value() / 100.0,
            fit_mode=self.fit_mode_combo.currentData(),
            show_date=self.date_checkbox.isChecked(),
            date_position=self.date_position_combo.currentData(),
            show_frame_number=self.frame_number_checkbox.isChecked(),
        )

    def _on_template_changed(self, *_args) -> None:
        template = get_film_frame_template(self.template_id)
        self.template_status_label.setText(
            f'{template.reference_status}：{template.reference_description}'
        )
        self.frame_number_checkbox.setEnabled(template.has_frame_number)
        self.frame_number_edit.setEnabled(template.has_frame_number)
        if not template.has_frame_number:
            self.frame_number_checkbox.setChecked(False)
        self._update_preview()

    def _update_preview(self, *_args) -> None:
        self.border_width_value.setText(f'{self.border_width_slider.value()}%')
        height, width, _ = self._image.shape
        scale = min(420 / width, 280 / height, 1.0)
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        preview_source = resize_rgb_image(self._image, (preview_width, preview_height))
        framed = compose_film_frame(
            preview_source,
            template_id=self.template_id,
            date_label=self.date_label_edit.text(),
            frame_number=self.frame_number_edit.text(),
            options=self.options,
        )
        pixels = np.clip(framed * 255.0 + 0.5, 0, 255).astype(np.uint8)
        image = QtGui.QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)
