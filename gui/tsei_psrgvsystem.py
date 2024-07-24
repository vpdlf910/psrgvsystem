import sys
import os
import re
import numpy as np
from functools import partial
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QInputDialog, QDialog,
    QDialogButtonBox, QCheckBox, QMessageBox, QScrollArea, QDateEdit, QTimeEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QRadioButton, QAbstractItemView, QComboBox, QListWidgetItem,QApplication
)
from PyQt5.QtCore import Qt, QDateTime, QDate, QTimer
from datetime import datetime, timedelta
import pandas as pd
from utils.database import (
    query_injection_conditions, fetch_injection_data, update_injection_data,
    delete_injection_data, insert_injection_data, fetch_manufacturing_data, query_real_time_sensor_data,query_polymer_solvent,
     query_injection_conditions_small, fetch_injection_data_small, update_injection_data_small,
    delete_injection_data_small, insert_injection_data_small, fetch_manufacturing_data_small, query_real_time_sensor_data_small,query_polymer_solvent_small,fetch_chamber_data
)
from utils.plot import (
    plot_data_volt, plot_ratio_data_volt, plot_multi_data_volt, plot_static_combine_volt,
    plot_ratio_combine_volt, plot_data_rs, plot_ratio_data_rs, plot_multi_data_rs, plot_static_combine_rs,
    plot_ratio_combine_rs,plot_data_volt_small, plot_ratio_data_volt_small, plot_multi_data_volt_small,
    plot_data_rs_small, plot_ratio_data_rs_small, plot_multi_data_rs_small,
    
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ChamberSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Chamber Type")
        self.setGeometry(100, 100, 300, 100)

        self.layout = QVBoxLayout()

        self.large_chamber_radio = QRadioButton("Large Chamber")
        self.small_chamber_radio = QRadioButton("Small Chamber")
        self.layout.addWidget(self.large_chamber_radio)
        self.layout.addWidget(self.small_chamber_radio)

        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_selected_chamber(self):
        if self.large_chamber_radio.isChecked():
            return "large"
        elif self.small_chamber_radio.isChecked():
            return "small"
        else:
            return None

class TSEI_PSRGVSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSEI_Polymer Sensor Result Graphic Visualization System")
        self.resize(800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.left_frame = QVBoxLayout()
        self.right_frame = QVBoxLayout()
        self.layout.addLayout(self.left_frame, 3)
        self.layout.addLayout(self.right_frame, 1)
        self.create_menu()
        self.show_intro_text()
        # Timer for real-time analysis
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph)
        # Initialize UI elements
        self.sensor_id_scroll_area = QScrollArea()
        self.sensor_id_list_widget = QListWidget()
    def closeEvent(self, event):
        QApplication.instance().quit()
    def create_menu(self):
        self.menu_bar = self.menuBar()
        graphic_menu_volt = self.menu_bar.addMenu("Volt_Graphic")
        graphic_menu_volt.addAction("Static Graph (Volt)", self.static_graphic_volt)
        graphic_menu_volt.addAction("Ratio Graph (Volt)", self.ratio_graphic_volt)
        graphic_menu_volt.addAction('Multi Graph (Volt)', self.multi_graphic_volt)
        graphic_menu_volt.addAction('Static Combine Graph (Volt)', self.static_combine_graphic_volt)
        graphic_menu_volt.addAction('Ratio Combine Graph (Volt)', self.ratio_combine_graphic_volt)
        graphic_menu_rs = self.menu_bar.addMenu("RS_Graphic")
        graphic_menu_rs.addAction("Static Graph (Rs)", self.static_graphic_rs)
        graphic_menu_rs.addAction("Ratio Graph (Rs)", self.ratio_graphic_rs)
        graphic_menu_rs.addAction('Multi Graph (Rs)', self.multi_graphic_rs)
        graphic_menu_rs.addAction('Static Combine Graph (Rs)', self.static_combine_graphic_rs)
        graphic_menu_rs.addAction('Ratio Combine Graph (Rs)', self.ratio_combine_graphic_rs)
        self.menu_bar.addAction("Reset", self.reset)
        self.menu_bar.addAction("Chamber Information", self.chamber_information_options)
        self.menu_bar.addAction("Manufacturing Process", self.manufacturing_process_options)
        self.menu_bar.addAction("Real-time Analysis", self.real_time_analysis_options)

    def show_intro_text(self):
        intro_text = (
            "TSEI_Polymer Sensor Result Graphic Visualization System은 다음과 같은 기능을 제공합니다.\n\n"
            "1. 그래프 생성\n"
            "Graphic 메뉴를 통해 그래프 설정 인터페이스를 제공하고, 파일 선택, 시작 시간 및 종료 시간 설정, 센서 ID 선택 후, "
            "Visualization 버튼을 클릭하면 설정된 조건에 따라 그래프를 생성하고 표시합니다.\n\n"
            "추가적인 세부 기능이나 UI 개선이 필요하다면 '(주)태성환경연구소' 고객센터로 문의해 주기 바랍니다.\n\n"
            "감사합니다.\n\n"
            "전화: 052-247-8691\n"
            "메일: info@ts-ei.com"
        )
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        label = QLabel(intro_text)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left_frame.addWidget(label)

    def static_graphic_volt(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Visualization (Volt)', self.visualize_static_volt)

    def ratio_graphic_volt(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Ratio Visualization (Volt)', self.visualize_ratio_volt)

    def multi_graphic_volt(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Multi Visualization (Volt)', self.visualize_multi_volt)

    def static_combine_graphic_volt(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Static Combine Visualization (Volt)', self.visualize_static_combine_volt)

    def ratio_combine_graphic_volt(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Ratio Combine Visualization (Volt)', self.visualize_ratio_combine_volt)

    def static_graphic_rs(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Visualization (Rs)', self.visualize_static_rs)

    def ratio_graphic_rs(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Ratio Visualization (Rs)', self.visualize_ratio_rs)

    def multi_graphic_rs(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Multi Visualization (Rs)', self.visualize_multi_rs)

    def static_combine_graphic_rs(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Static Combine Visualization (Rs)', self.visualize_static_combine_rs)

    def ratio_combine_graphic_rs(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.create_graphic_options('Ratio Combine Visualization (Rs)', self.visualize_ratio_combine_rs)

    def reset(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        self.show_intro_text()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def chamber_information_options(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Idx', 'Injection Time', 'Injection Condition', 'Review'])
        self.load_injection_data()
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 350)
        layout.addWidget(self.table)
        button_layout = QHBoxLayout()
        buttons = [
            ("Add", self.add_injection_data),
            ("Update", self.update_injection_data),
            ("Delete", self.delete_injection_data),
            ("Load", self.load_injection_data)
        ]
        for name, func in buttons:
            button = QPushButton(name)
            button.clicked.connect(func)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)
        self.left_frame.addLayout(layout)

    def manufacturing_process_options(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(17)
        self.table.setHorizontalHeaderLabels([
            'Idx', 'Date of Manufacture', 'Sensor Name', 'Applied Polymer', 'Solvent', 'Solvent Usage (ml)',
            'Carbon Black Usage (g)', 'CNT Usage (g)', 'Polymer Usage (g)', 'Stirring Time (min)', 'Processing Conditions',
            'Substrate', 'Pattern Size (mm)', 'Resistance Measurement 1st', 'Resistance Measurement 2nd',
            'Toluene Validation Experiment', 'Review'
        ])
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.table)
        self.left_frame.addLayout(layout)
        self.load_manufacturing_data('sensor_name', True)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 200)
        self.table.setColumnWidth(4, 150)
        self.table.setColumnWidth(5, 150)
        self.table.setColumnWidth(6, 150)
        self.table.setColumnWidth(7, 150)
        self.table.setColumnWidth(8, 150)
        self.table.setColumnWidth(9, 150)
        self.table.setColumnWidth(10, 250)
        self.table.setColumnWidth(11, 200)
        self.table.setColumnWidth(12, 200)
        self.table.setColumnWidth(13, 250)
        self.table.setColumnWidth(14, 250)
        self.table.setColumnWidth(15, 300)
        self.table.setColumnWidth(16, 400)

        right_layout = QVBoxLayout()
        self.column_list = QListWidget()
        self.column_list.addItems([
            'Idx', 'Date of Manufacture', 'Sensor Name', 'Applied Polymer', 'Solvent', 'Solvent Usage (ml)',
            'Carbon Black Usage (g)', 'CNT Usage (g)', 'Polymer Usage (g)', 'Stirring Time (min)', 'Processing Conditions',
            'Substrate', 'Pattern Size (mm)', 'Resistance Measurement 1st', 'Resistance Measurement 2nd',
            'Toluene Validation Experiment', 'Review'
        ])
        self.column_list.setSelectionMode(QAbstractItemView.SingleSelection)
        right_layout.addWidget(self.column_list)
        self.ascending_radio = QRadioButton("Ascending")
        self.descending_radio = QRadioButton("Descending")
        self.ascending_radio.setChecked(True)
        order_layout = QHBoxLayout()
        order_layout.addWidget(self.ascending_radio)
        order_layout.addWidget(self.descending_radio)
        right_layout.addLayout(order_layout)
        fetch_button = QPushButton("Fetch Data")
        fetch_button.clicked.connect(self.fetch_and_load_data)
        right_layout.addWidget(fetch_button)
        self.right_frame.addLayout(right_layout)

    def load_manufacturing_data(self, order_by_column, ascending):
        data = fetch_manufacturing_data(order_by_column, ascending)
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['date_of_manufacture'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['sensor_name'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['applied_polymer'])))
            self.table.setItem(row_index, 4, QTableWidgetItem(str(row_data['solvent'])))
            self.table.setItem(row_index, 5, QTableWidgetItem(str(row_data['solvent_usage_ml'])))
            self.table.setItem(row_index, 6, QTableWidgetItem(str(row_data['carbon_black_usage_g'])))
            self.table.setItem(row_index, 7, QTableWidgetItem(str(row_data['cnt_usage_g'])))
            self.table.setItem(row_index, 8, QTableWidgetItem(str(row_data['polymer_usage_g'])))
            self.table.setItem(row_index, 9, QTableWidgetItem(str(row_data['stirring_time_min'])))
            self.table.setItem(row_index, 10, QTableWidgetItem(str(row_data['processing_conditions'])))
            self.table.setItem(row_index, 11, QTableWidgetItem(str(row_data['substrate'])))
            self.table.setItem(row_index, 12, QTableWidgetItem(str(row_data['pattern_size_mm'])))
            self.table.setItem(row_index, 13, QTableWidgetItem(str(row_data['resistance_measurement_1st'])))
            self.table.setItem(row_index, 14, QTableWidgetItem(str(row_data['resistance_measurement_2nd'])))
            self.table.setItem(row_index, 15, QTableWidgetItem(str(row_data['toluene_validation_experiment'])))
            self.table.setItem(row_index, 16, QTableWidgetItem(str(row_data['review'])))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def fetch_and_load_data(self):
        selected_column = self.column_list.currentItem().text().replace(' ', '_').lower()
        ascending = self.ascending_radio.isChecked()
        data = fetch_manufacturing_data(selected_column, ascending)
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['date_of_manufacture'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['sensor_name'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['applied_polymer'])))
            self.table.setItem(row_index, 4, QTableWidgetItem(str(row_data['solvent'])))
            self.table.setItem(row_index, 5, QTableWidgetItem(str(row_data['solvent_usage_ml'])))
            self.table.setItem(row_index, 6, QTableWidgetItem(str(row_data['carbon_black_usage_g'])))
            self.table.setItem(row_index, 7, QTableWidgetItem(str(row_data['cnt_usage_g'])))
            self.table.setItem(row_index, 8, QTableWidgetItem(str(row_data['polymer_usage_g'])))
            self.table.setItem(row_index, 9, QTableWidgetItem(str(row_data['stirring_time_min'])))
            self.table.setItem(row_index, 10, QTableWidgetItem(str(row_data['processing_conditions'])))
            self.table.setItem(row_index, 11, QTableWidgetItem(str(row_data['substrate'])))
            self.table.setItem(row_index, 12, QTableWidgetItem(str(row_data['pattern_size_mm'])))
            self.table.setItem(row_index, 13, QTableWidgetItem(str(row_data['resistance_measurement_1st'])))
            self.table.setItem(row_index, 14, QTableWidgetItem(str(row_data['resistance_measurement_2nd'])))
            self.table.setItem(row_index, 15, QTableWidgetItem(str(row_data['toluene_validation_experiment'])))
            self.table.setItem(row_index, 16, QTableWidgetItem(str(row_data['review'])))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def load_injection_data(self):
        injection_data = fetch_injection_data()
        self.table.setRowCount(len(injection_data))
        for row_index, row_data in enumerate(injection_data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['injection_time'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['injection_condition'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['review'])))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_injection_data(self):
        self.modify_injection_data(insert_injection_data)

    def update_injection_data(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            idx = self.table.item(row, 0).text()
            self.modify_injection_data(update_injection_data, idx)
        else:
            QMessageBox.warning(self, "No selection", "Please select a row to update.")

    def delete_injection_data(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            for row in selected_rows:
                idx = self.table.item(row.row(), 0).text()
                delete_injection_data(idx)
            self.load_injection_data()
        else:
            QMessageBox.warning(self, "No selection", "Please select a row to delete.")

    def modify_injection_data(self, db_func, idx=None):
        injection_time, ok1 = QInputDialog.getText(self, 'Input Dialog', 'Enter injection time (YYYY-MM-DD HH:MM:SS):')
        if not ok1:
            return
        injection_condition, ok2 = QInputDialog.getText(self, 'Input Dialog', 'Enter injection condition:')
        if not ok2:
            return
        review, ok3 = QInputDialog.getText(self, 'Input Dialog', 'Enter review (optional):')
        if not ok3:
            review = None
        if idx is None:
            db_func(injection_time, injection_condition, review)
        else:
            db_func(idx, injection_time, injection_condition, review)
        self.load_injection_data()

    def create_graphic_options(self, button_text, visualize_func):
        self.selected_sensor_id = []
        self.date_time_pairs = []
        date_selection_layout = QVBoxLayout()
        date_selection_button = QPushButton('날짜 선택')
        date_selection_button.clicked.connect(self.add_date)
        date_selection_layout.addWidget(date_selection_button)
        self.date_label_layout = QVBoxLayout()
        date_scroll_area = QScrollArea()
        date_scroll_area.setWidgetResizable(True)
        date_widget = QWidget()
        date_widget.setLayout(self.date_label_layout)
        date_scroll_area.setWidget(date_widget)
        date_selection_layout.addWidget(date_scroll_area)
        self.left_frame.addLayout(date_selection_layout)
        sensor_id_layout = QHBoxLayout()
        sensor_id_button = QPushButton('Sensor ID 선택')
        sensor_id_button.clicked.connect(self.select_sensor_ids)
        sensor_id_layout.addWidget(sensor_id_button)
        self.selected_sensor_ids_label = QLabel('')
        sensor_id_layout.addWidget(self.selected_sensor_ids_label)
        self.left_frame.addLayout(sensor_id_layout)
        visualize_button = QPushButton(button_text)
        visualize_button.clicked.connect(visualize_func)
        self.left_frame.addWidget(visualize_button)

    def add_date(self):
        selected_date = self.select_date()
        if selected_date:
            self.date_time_pairs.append((selected_date, None, None))
            self.add_date_label(selected_date)
            self.ask_for_time(selected_date)

    def add_date_label(self, date):
        date_layout = QHBoxLayout()
        label = QLabel(f"선택된 날짜: {date.toString('yyyy-MM-dd')}")
        date_layout.addWidget(label)
        container_widget = QWidget()
        container_widget.setLayout(date_layout)
        self.date_label_layout.addWidget(container_widget)

    def ask_for_time(self, date):
        self.current_date = date
        selected_times = self.select_time()
        if selected_times:
            for i in range(len(self.date_time_pairs)):
                if self.date_time_pairs[i][0] == date:
                    self.date_time_pairs[i] = (date, selected_times[0], selected_times[1])
                    self.add_time_label(date, selected_times[0], selected_times[1])
                    break
        else:
            self.date_time_pairs = [dt for dt in self.date_time_pairs if dt[0] != date]
            for i in range(self.date_label_layout.count()):
                item = self.date_label_layout.itemAt(i)
                if isinstance(item.widget(), QWidget) and isinstance(item.widget().layout().itemAt(0).widget(), QLabel) and item.widget().layout().itemAt(0).widget().text() == f"선택된 날짜: {date.toString('yyyy-MM-dd')}":
                    self.date_label_layout.takeAt(i).widget().deleteLater()
                    break

    def add_time_label(self, date, start_time, end_time):
        time_layout = QHBoxLayout()
        label = QLabel(f'{date.toString("yyyy-MM-dd")} {start_time.toString("HH:mm:ss")} - {end_time.toString("HH:mm:ss")}')
        delete_button = QPushButton('삭제')
        delete_button.clicked.connect(lambda: self.remove_date_time(time_layout, date, start_time, end_time, True))
        time_layout.addWidget(label)
        time_layout.addWidget(delete_button)
        container_widget = QWidget()
        container_widget.setLayout(time_layout)
        self.date_label_layout.addWidget(container_widget)

    def remove_date_time(self, layout, date, start_time, end_time, remove_date_label):
        for i in range(self.date_label_layout.count()):
            item = self.date_label_layout.itemAt(i)
            if item.layout() == layout:
                self.date_label_layout.takeAt(i)
                break
        self.date_time_pairs = [dt for dt in self.date_time_pairs if dt != (date, start_time, end_time)]
        self.clear_layout(layout)
        if remove_date_label:
            for i in range(self.date_label_layout.count()):
                item = self.date_label_layout.itemAt(i)
                if item is not None and isinstance(item.widget(), QWidget):
                    widget_layout = item.widget().layout()
                    if widget_layout is not None:
                        first_item = widget_layout.itemAt(0)
                        if first_item is not None and isinstance(first_item.widget(), QLabel) and first_item.widget().text() == f"선택된 날짜: {date.toString('yyyy-MM-dd')}":
                            self.date_label_layout.takeAt(i).widget().deleteLater()
                            break

    def select_date(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Date')
        layout = QVBoxLayout()
        date_edit = QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QDate.currentDate())
        date_edit.setDisplayFormat('yyyy-MM-dd')
        layout.addWidget(date_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        selected_date = None
        if dialog.exec_() == QDialog.Accepted:
            selected_date = date_edit.date()
        return selected_date

    def select_time(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Start and End Time')
        layout = QVBoxLayout()
        start_time_edit = QTimeEdit()
        start_time_edit.setDisplayFormat('HH:mm:ss')
        layout.addWidget(QLabel("Start Time:"))
        layout.addWidget(start_time_edit)
        end_time_edit = QTimeEdit()
        end_time_edit.setDisplayFormat('HH:mm:ss')
        layout.addWidget(QLabel("End Time:"))
        layout.addWidget(end_time_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        selected_times = None
        if dialog.exec_() == QDialog.Accepted:
            selected_times = (start_time_edit.time(), end_time_edit.time())
        else:
            self.date_time_pairs = [dt for dt in self.date_time_pairs if dt[0] != self.current_date]
        return selected_times

    def select_sensor_ids(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Sensor IDs')

        layout = QVBoxLayout()
        self.sensor_id_vars = []

        all_select_checkbox = QCheckBox('All Select')
        all_select_checkbox.stateChanged.connect(lambda: self.toggle_all_sensor_ids(self.sensor_id_vars, all_select_checkbox))
        layout.addWidget(all_select_checkbox)

        range_select_layout = QHBoxLayout()
        for start in range(1, 41, 10):
            end = start + 9
            range_checkbox = QCheckBox(f'{start}-{end}')
            range_checkbox.stateChanged.connect(lambda state, start=start, end=end: self.toggle_range_sensor_ids(self.sensor_id_vars, start, end, state))
            range_select_layout.addWidget(range_checkbox)
        layout.addLayout(range_select_layout)

        interactive_select_layout = QVBoxLayout()
        df_list = []
        for date, start_time, end_time in self.date_time_pairs:
            date_str = date.toString('yyyy-MM-dd')
            try:
                result = query_polymer_solvent(date_str)
                if result:
                    df = pd.DataFrame(result)
                else:
                    df = pd.DataFrame(columns=['sensor_name', 'applied_polymer', 'solvent'])
            except Exception as e:
                print(f"Error: {e}")
                df = pd.DataFrame(columns=['sensor_name', 'applied_polymer', 'solvent'])
            df_list.append(df)
        
        if df_list:
            concated_df = pd.concat(df_list, ignore_index=True)
            polymer_list = concated_df['applied_polymer'].unique()
            solvent_list = concated_df['solvent'].unique()
        else:
            concated_df = pd.DataFrame(columns=['sensor_name', 'applied_polymer', 'solvent'])
            polymer_list = []
            solvent_list = []

        def extract_sensor_id(sensor_name):
            match = re.match(r'.*-S\d+-(\d+)', sensor_name)
            if match:
                return int(match.group(1))
            
            match = re.match(r'.*-S(\d+)', sensor_name)
            if match:
                return int(match.group(1))
            
            return None

        concated_df['sensor_id'] = concated_df['sensor_name'].apply(extract_sensor_id)

        polymer_layout = QVBoxLayout()
        polymer_label = QLabel('Polymers:')
        polymer_layout.addWidget(polymer_label)
        for polymer in polymer_list:
            checkbox = QCheckBox(polymer)
            checkbox.stateChanged.connect(lambda state, polymer=polymer: self.toggle_polymer_sensor_ids(concated_df, polymer, state))
            polymer_layout.addWidget(checkbox)
        interactive_select_layout.addLayout(polymer_layout)

        solvent_layout = QVBoxLayout()
        solvent_label = QLabel('Solvents:')
        solvent_layout.addWidget(solvent_label)
        for solvent in solvent_list:
            checkbox = QCheckBox(solvent)
            checkbox.stateChanged.connect(lambda state, solvent=solvent: self.toggle_solvent_sensor_ids(concated_df, solvent, state))
            solvent_layout.addWidget(checkbox)
        interactive_select_layout.addLayout(solvent_layout)

        layout.addLayout(interactive_select_layout)

        for i in range(4):
            row_layout = QHBoxLayout()
            for j in range(10):
                checkbox = QCheckBox(str(i * 10 + j + 1))
                row_layout.addWidget(checkbox)
                self.sensor_id_vars.append(checkbox)
            layout.addLayout(row_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            self.selected_sensor_id = [i + 1 for i, var in enumerate(self.sensor_id_vars) if var.isChecked()]
            self.sensor_id_list_widget.clear()
            for sensor_id in self.selected_sensor_id:
                item = QListWidgetItem(f"Sensor ID {sensor_id}")
                self.sensor_id_list_widget.addItem(item)

    def toggle_all_sensor_ids(self, sensor_id_vars, all_select_checkbox):
        state = all_select_checkbox.isChecked()
        for checkbox in sensor_id_vars:
            checkbox.setChecked(state)

    def toggle_range_sensor_ids(self, sensor_id_vars, start, end, state):
        for i in range(start - 1, end):
            sensor_id_vars[i].setChecked(state)

    def toggle_polymer_sensor_ids(self, df, polymer, state):
        sensor_ids = df[df['applied_polymer'] == polymer]['sensor_id'].tolist()
        for checkbox in self.sensor_id_vars:
            sensor_id = int(checkbox.text())
            if sensor_id in sensor_ids:
                checkbox.setChecked(state)

    def toggle_solvent_sensor_ids(self, df, solvent, state):
        sensor_ids = df[df['solvent'] == solvent]['sensor_id'].tolist()
        for checkbox in self.sensor_id_vars:
            sensor_id = int(checkbox.text())
            if sensor_id in sensor_ids:
                checkbox.setChecked(state)

    def select_sensor_ids_old(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Sensor IDs')

        layout = QVBoxLayout()
        sensor_id_vars = []

        all_select_checkbox = QCheckBox('All Select')
        all_select_checkbox.stateChanged.connect(lambda: self.toggle_all_sensor_ids(sensor_id_vars, all_select_checkbox))
        layout.addWidget(all_select_checkbox)

        range_select_layout = QHBoxLayout()
        for start in range(1, 41, 10):
            end = start + 9
            range_checkbox = QCheckBox(f'{start}-{end}')
            range_checkbox.stateChanged.connect(lambda state, start=start, end=end: self.toggle_range_sensor_ids(sensor_id_vars, start, end, state))
            range_select_layout.addWidget(range_checkbox)
        layout.addLayout(range_select_layout)

        for i in range(4):
            row_layout = QHBoxLayout()
            for j in range(10):
                checkbox = QCheckBox(str(i * 10 + j + 1))
                row_layout.addWidget(checkbox)
                sensor_id_vars.append(checkbox)
            layout.addLayout(row_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            self.selected_sensor_id = [i + 1 for i, var in enumerate(sensor_id_vars) if var.isChecked()]
            self.sensor_id_list_widget.clear()
            for sensor_id in self.selected_sensor_id:
                item = QListWidgetItem(f"Sensor ID {sensor_id}")
                self.sensor_id_list_widget.addItem(item)


    def visualize_graph(self, plot_func):
        try:
            if not self.date_time_pairs:
                raise ValueError("날짜와 시간을 모두 선택해야 합니다.")
            start_times = []
            end_times = []
            for date, start_time, end_time in self.date_time_pairs:
                if start_time is None or end_time is None:
                    raise ValueError("모든 날짜에 대해 시작 시간과 종료 시간을 설정해야 합니다.")
                start_times.append(QDateTime(date, start_time).toPyDateTime())
                end_times.append(QDateTime(date, end_time).toPyDateTime())
            if len(start_times) != len(end_times):
                raise ValueError("시작 시간과 종료 시간의 개수가 일치해야 합니다.")
            for start_time, end_time in zip(start_times, end_times):
                if start_time >= end_time:
                    raise ValueError("시작 시간은 종료 시간보다 빨라야 합니다.")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid input", str(e))
            return

        if not self.selected_sensor_id:
            QMessageBox.critical(self, "Invalid input", "적어도 하나의 Sensor ID를 선택해 주세요.")
            return

        combined_start_time = min(start_times)
        combined_end_time = max(end_times)
        injection_times = query_injection_conditions(combined_start_time, combined_end_time, self.selected_sensor_id)

        plot_func(self.selected_sensor_id, combined_start_time, combined_end_time, injection_times, self.date_time_pairs)

    def visualize_static_volt(self):
        self.visualize_graph(plot_data_volt)

    def visualize_ratio_volt(self):
        self.visualize_graph(plot_ratio_data_volt)

    def visualize_multi_volt(self):
        self.visualize_graph(plot_multi_data_volt)

    def visualize_static_combine_volt(self):
        self.visualize_graph(plot_static_combine_volt)

    def visualize_ratio_combine_volt(self):
        self.visualize_graph(plot_ratio_combine_volt)

    def visualize_static_rs(self):
        self.visualize_graph(plot_data_rs)

    def visualize_ratio_rs(self):
        self.visualize_graph(plot_ratio_data_rs)

    def visualize_multi_rs(self):
        self.visualize_graph(plot_multi_data_rs)

    def visualize_static_combine_rs(self):
        self.visualize_graph(plot_static_combine_rs)

    def visualize_ratio_combine_rs(self):
        self.visualize_graph(plot_ratio_combine_rs)

    def real_time_analysis_options(self):
        self.clear_layout(self.left_frame)
        self.clear_layout(self.right_frame)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()

        layout.addWidget(self.canvas)
        self.left_frame.addLayout(layout)
        self.selected_sensor_id = []
        self.date_time_pairs = []

        self.sensor_id_button = QPushButton('Sensor ID 선택')
        
        #self.date_time_pairs = [(QDateTime.currentDateTime().date(),0,0)]
        self.sensor_id_button.clicked.connect(self.select_sensor_ids_old)
        self.sensor_id_scroll_area.setWidget(self.sensor_id_list_widget)
        self.sensor_id_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.right_frame.addWidget(self.sensor_id_button)
        self.right_frame.addWidget(self.sensor_id_scroll_area)

        self.interval_selection = QComboBox()
        self.interval_selection.addItems(["5초", "10초", "30초", "60초"])
        self.right_frame.addWidget(self.interval_selection)

        self.time_selection = QComboBox()
        self.time_selection.addItems(['1 Hour Ago', '2 Hours Ago', '6 Hours Ago', '12 Hours Ago'])
        self.right_frame.addWidget(QLabel("Select Time Range:"))
        self.right_frame.addWidget(self.time_selection)

        self.real_time_button = QPushButton('실시간 분석 시작')
        self.real_time_button.clicked.connect(self.start_real_time_analysis)
        self.right_frame.addWidget(self.real_time_button)

        self.stop_real_time_button = QPushButton('실시간 분석 중지')
        self.stop_real_time_button.clicked.connect(self.stop_real_time_analysis)
        self.right_frame.addWidget(self.stop_real_time_button)

        self.real_time_button.setEnabled(True)
        self.stop_real_time_button.setEnabled(False)


    def start_real_time_analysis(self):
        interval_mapping = {"5초": 5000, "10초": 10000, "30초": 30000, "60초": 60000}
        selected_interval = self.interval_selection.currentText()
        if selected_interval in interval_mapping:
            self.timer.start(interval_mapping[selected_interval])
            self.real_time_button.setEnabled(False)
            self.stop_real_time_button.setEnabled(True)
            self.update_graph()

    def stop_real_time_analysis(self):
        self.timer.stop()
        self.real_time_button.setEnabled(True)
        self.stop_real_time_button.setEnabled(False)

    def update_graph(self):
        time_range = self.time_selection.currentText()

        end_time = datetime.now()
        if time_range == '1 Hour Ago':
            start_time = end_time - timedelta(hours=1)
        elif time_range == '2 Hours Ago':
            start_time = end_time - timedelta(hours=2)
        elif time_range == '6 Hours Ago':
            start_time = end_time - timedelta(hours=6)
        elif time_range == '12 Hours Ago':
            start_time = end_time - timedelta(hours=12)

        if not self.selected_sensor_id:
            QMessageBox.critical(self, "Invalid input", "적어도 하나의 Sensor ID를 선택해 주세요.")
            return

        data = query_real_time_sensor_data(start_time, end_time, self.selected_sensor_id)

        self.ax.clear()
        for sensor_id in self.selected_sensor_id:
            sensor_data = [entry for entry in data if entry['sensor_id'] == sensor_id]
            times = [entry['reg_date'] for entry in sensor_data]
            volts = [entry['volt'] for entry in sensor_data]
            self.ax.plot(times, volts, label=f"Sensor ID {sensor_id}")

        self.ax.set_title("Real-time Voltage Data")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Voltage")
        self.canvas.draw()


class TSEI_PSRGVSystem_small(QMainWindow):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("TSEI_Polymer Sensor Result Graphic Visualization System")
        self.resize(800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.left_frame = QVBoxLayout()
        self.right_frame = QVBoxLayout()
        self.layout.addLayout(self.left_frame, 3)
        self.layout.addLayout(self.right_frame, 1)
        self.create_menu_small()
        self.show_intro_text_small()
        # Timer for real-time analysis
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph_small)
        # Initialize UI elements
        self.sensor_id_scroll_area = QScrollArea()
        self.sensor_id_list_widget = QListWidget()
        self.selected_sensor_data = {}
        self.time_pairs = []
        self.date_time_pairs = []
        self.start_time = None  # 초기화
        self.end_time = None  # 초기화
        
        

    def create_menu_small(self):
        self.menu_bar = self.menuBar()
        graphic_menu_volt_small = self.menu_bar.addMenu("Volt_Graphic")
        graphic_menu_volt_small.addAction("Static Graph (Volt)", self.static_graphic_volt_small)
        graphic_menu_volt_small.addAction("Ratio Graph (Volt)", self.ratio_graphic_volt_small)
        graphic_menu_volt_small.addAction('Multi Graph (Volt)', self.multi_graphic_volt_small)

        graphic_menu_rs_small = self.menu_bar.addMenu("RS_Graphic")
        graphic_menu_rs_small.addAction("Static Graph (Rs)", self.static_graphic_rs_small)
        graphic_menu_rs_small.addAction("Ratio Graph (Rs)", self.ratio_graphic_rs_small)
        graphic_menu_rs_small.addAction('Multi Graph (Rs)', self.multi_graphic_rs_small)

        self.menu_bar.addAction("Reset", self.reset_small)
        self.menu_bar.addAction("Chamber Information", self.chamber_information_options_small)
        self.menu_bar.addAction("Manufacturing Process", self.manufacturing_process_options_small)
        self.menu_bar.addAction("Real-time Analysis", self.real_time_analysis_options_small)

    def show_intro_text_small(self):
        intro_text = (
            "TSEI_Polymer Sensor Result Graphic Visualization System은 다음과 같은 기능을 제공합니다.\n\n"
            "1. 그래프 생성\n"
            "Graphic 메뉴를 통해 그래프 설정 인터페이스를 제공하고, 파일 선택, 시작 시간 및 종료 시간 설정, 센서 ID 선택 후, "
            "Visualization 버튼을 클릭하면 설정된 조건에 따라 그래프를 생성하고 표시합니다.\n\n"
            "추가적인 세부 기능이나 UI 개선이 필요하다면 '(주)태성환경연구소' 고객센터로 문의해 주기 바랍니다.\n\n"
            "감사합니다.\n\n"
            "전화: 052-247-8691\n"
            "메일: info@ts-ei.com"
        )
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        label = QLabel(intro_text)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left_frame.addWidget(label)

    def static_graphic_volt_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Visualization (Volt)', self.visualize_static_volt_small)

    def ratio_graphic_volt_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Ratio Visualization (Volt)', self.visualize_ratio_volt_small)

    def multi_graphic_volt_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Multi Visualization (Volt)', self.visualize_multi_volt_small)

    def static_graphic_rs_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Visualization (Rs)', self.visualize_static_rs_small)

    def ratio_graphic_rs_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Ratio Visualization (Rs)', self.visualize_ratio_rs_small)

    def multi_graphic_rs_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.create_graphic_options_small('Multi Visualization (Rs)', self.visualize_multi_rs_small)

    def reset_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        self.show_intro_text_small()

    def clear_layout_small(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout_small(item.layout())

    def chamber_information_options_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Idx', 'Injection Time', 'Injection Condition', 'Review'])
        self.load_injection_data_small()
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 350)
        layout.addWidget(self.table)
        button_layout = QHBoxLayout()
        buttons = [
            ("Add", self.add_injection_data_small),
            ("Update", self.update_injection_data_small),
            ("Delete", self.delete_injection_data_small),
            ("Load", self.load_injection_data_small)
        ]
        for name, func in buttons:
            button = QPushButton(name)
            button.clicked.connect(func)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)
        self.left_frame.addLayout(layout)

    def manufacturing_process_options_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(17)
        self.table.setHorizontalHeaderLabels([
            'Idx', 'Date of Manufacture', 'Sensor Name', 'Applied Polymer', 'Solvent', 'Solvent Usage (ml)',
            'Carbon Black Usage (g)', 'CNT Usage (g)', 'Polymer Usage (g)', 'Stirring Time (min)', 'Processing Conditions',
            'Substrate', 'Pattern Size (mm)', 'Resistance Measurement 1st', 'Resistance Measurement 2nd',
            'Toluene Validation Experiment', 'Review'
        ])
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.table)
        self.left_frame.addLayout(layout)
        self.load_manufacturing_data_small('sensor_name', True)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 200)
        self.table.setColumnWidth(4, 150)
        self.table.setColumnWidth(5, 150)
        self.table.setColumnWidth(6, 150)
        self.table.setColumnWidth(7, 150)
        self.table.setColumnWidth(8, 150)
        self.table.setColumnWidth(9, 150)
        self.table.setColumnWidth(10, 250)
        self.table.setColumnWidth(11, 200)
        self.table.setColumnWidth(12, 200)
        self.table.setColumnWidth(13, 250)
        self.table.setColumnWidth(14, 250)
        self.table.setColumnWidth(15, 300)
        self.table.setColumnWidth(16, 400)

        right_layout = QVBoxLayout()
        self.column_list = QListWidget()
        self.column_list.addItems([
            'Idx', 'Date of Manufacture', 'Sensor Name', 'Applied Polymer', 'Solvent', 'Solvent Usage (ml)',
            'Carbon Black Usage (g)', 'CNT Usage (g)', 'Polymer Usage (g)', 'Stirring Time (min)', 'Processing Conditions',
            'Substrate', 'Pattern Size (mm)', 'Resistance Measurement 1st', 'Resistance Measurement 2nd',
            'Toluene Validation Experiment', 'Review'
        ])
        self.column_list.setSelectionMode(QAbstractItemView.SingleSelection)
        right_layout.addWidget(self.column_list)
        self.ascending_radio = QRadioButton("Ascending")
        self.descending_radio = QRadioButton("Descending")
        self.ascending_radio.setChecked(True)
        order_layout = QHBoxLayout()
        order_layout.addWidget(self.ascending_radio)
        order_layout.addWidget(self.descending_radio)
        right_layout.addLayout(order_layout)
        fetch_button = QPushButton("Fetch Data")
        fetch_button.clicked.connect(self.fetch_and_load_data_small)
        right_layout.addWidget(fetch_button)
        self.right_frame.addLayout(right_layout)

    def load_manufacturing_data_small(self, order_by_column, ascending):
        data = fetch_manufacturing_data_small(order_by_column, ascending)
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['date_of_manufacture'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['sensor_name'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['applied_polymer'])))
            self.table.setItem(row_index, 4, QTableWidgetItem(str(row_data['solvent'])))
            self.table.setItem(row_index, 5, QTableWidgetItem(str(row_data['solvent_usage_ml'])))
            self.table.setItem(row_index, 6, QTableWidgetItem(str(row_data['carbon_black_usage_g'])))
            self.table.setItem(row_index, 7, QTableWidgetItem(str(row_data['cnt_usage_g'])))
            self.table.setItem(row_index, 8, QTableWidgetItem(str(row_data['polymer_usage_g'])))
            self.table.setItem(row_index, 9, QTableWidgetItem(str(row_data['stirring_time_min'])))
            self.table.setItem(row_index, 10, QTableWidgetItem(str(row_data['processing_conditions'])))
            self.table.setItem(row_index, 11, QTableWidgetItem(str(row_data['substrate'])))
            self.table.setItem(row_index, 12, QTableWidgetItem(str(row_data['pattern_size_mm'])))
            self.table.setItem(row_index, 13, QTableWidgetItem(str(row_data['resistance_measurement_1st'])))
            self.table.setItem(row_index, 14, QTableWidgetItem(str(row_data['resistance_measurement_2nd'])))
            self.table.setItem(row_index, 15, QTableWidgetItem(str(row_data['toluene_validation_experiment'])))
            self.table.setItem(row_index, 16, QTableWidgetItem(str(row_data['review'])))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def fetch_and_load_data_small(self):
        selected_column = self.column_list.currentItem().text().replace(' ', '_').lower()
        ascending = self.ascending_radio.isChecked()
        data = fetch_manufacturing_data_small(selected_column, ascending)
        self.table.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['date_of_manufacture'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['sensor_name'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['applied_polymer'])))
            self.table.setItem(row_index, 4, QTableWidgetItem(str(row_data['solvent'])))
            self.table.setItem(row_index, 5, QTableWidgetItem(str(row_data['solvent_usage_ml'])))
            self.table.setItem(row_index, 6, QTableWidgetItem(str(row_data['carbon_black_usage_g'])))
            self.table.setItem(row_index, 7, QTableWidgetItem(str(row_data['cnt_usage_g'])))
            self.table.setItem(row_index, 8, QTableWidgetItem(str(row_data['polymer_usage_g'])))
            self.table.setItem(row_index, 9, QTableWidgetItem(str(row_data['stirring_time_min'])))
            self.table.setItem(row_index, 10, QTableWidgetItem(str(row_data['processing_conditions'])))
            self.table.setItem(row_index, 11, QTableWidgetItem(str(row_data['substrate'])))
            self.table.setItem(row_index, 12, QTableWidgetItem(str(row_data['pattern_size_mm'])))
            self.table.setItem(row_index, 13, QTableWidgetItem(str(row_data['resistance_measurement_1st'])))
            self.table.setItem(row_index, 14, QTableWidgetItem(str(row_data['resistance_measurement_2nd'])))
            self.table.setItem(row_index, 15, QTableWidgetItem(str(row_data['toluene_validation_experiment'])))
            self.table.setItem(row_index, 16, QTableWidgetItem(str(row_data['review'])))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def load_injection_data_small(self):
        injection_data = fetch_injection_data_small()
        self.table.setRowCount(len(injection_data))
        for row_index, row_data in enumerate(injection_data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row_data['idx'])))
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row_data['injection_time'])))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(row_data['injection_condition'])))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(row_data['review'])))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_injection_data_small(self):
        self.modify_injection_data_small(insert_injection_data_small)

    def update_injection_data_small(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            idx = self.table.item(row, 0).text()
            self.modify_injection_data_small(update_injection_data_small, idx)
        else:
            QMessageBox.warning(self, "No selection", "Please select a row to update.")

    def delete_injection_data_small(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            for row in selected_rows:
                idx = self.table.item(row.row(), 0).text()
                delete_injection_data_small(idx)
            self.load_injection_data_small()
        else:
            QMessageBox.warning(self, "No selection", "Please select a row to delete.")

    def modify_injection_data_small(self, db_func, idx=None):
        injection_time, ok1 = QInputDialog.getText(self, 'Input Dialog', 'Enter injection time (YYYY-MM-DD HH:MM:SS):')
        if not ok1:
            return
        injection_condition, ok2 = QInputDialog.getText(self, 'Input Dialog', 'Enter injection condition:')
        if not ok2:
            return
        review, ok3 = QInputDialog.getText(self, 'Input Dialog', 'Enter review (optional):')
        if not ok3:
            review = None
        if idx is None:
            db_func(injection_time, injection_condition, review)
        else:
            db_func(idx, injection_time, injection_condition, review)
        self.load_injection_data_small()

    def create_graphic_options_small(self, button_text, visualize_func):
        self.selected_sensor_data = {}
        self.date_time_pairs = []
        date_selection_layout = QVBoxLayout()
        date_selection_button = QPushButton('날짜 선택')
        date_selection_button.clicked.connect(self.add_date_small)
        date_selection_layout.addWidget(date_selection_button)
        self.date_label_layout = QVBoxLayout()
        date_scroll_area = QScrollArea()
        date_scroll_area.setWidgetResizable(True)
        date_widget = QWidget()
        date_widget.setLayout(self.date_label_layout)
        date_scroll_area.setWidget(date_widget)
        date_selection_layout.addWidget(date_scroll_area)
        self.left_frame.addLayout(date_selection_layout)
        sensor_id_layout = QHBoxLayout()
        sensor_id_button = QPushButton('Chamber ID 선택')
        sensor_id_button.clicked.connect(self.select_sensor_ids_small)
        sensor_id_layout.addWidget(sensor_id_button)
        self.selected_sensor_ids_label = QLabel('')
        sensor_id_layout.addWidget(self.selected_sensor_ids_label)
        self.left_frame.addLayout(sensor_id_layout)
        visualize_button = QPushButton(button_text)
        visualize_button.clicked.connect(visualize_func)
        self.left_frame.addWidget(visualize_button)

    def add_date_small(self):
        selected_date = self.select_date_small()
        if selected_date:
            self.date_time_pairs.append((selected_date, None, None))
            self.add_date_label_small(selected_date)
            self.ask_for_time_small(selected_date)

    def add_date_label_small(self, date):
        date_layout = QHBoxLayout()
        label = QLabel(f"선택된 날짜: {date.toString('yyyy-MM-dd')}")
        date_layout.addWidget(label)
        container_widget = QWidget()
        container_widget.setLayout(date_layout)
        self.date_label_layout.addWidget(container_widget)

    def ask_for_time_small(self, date):
        self.current_date = date
        selected_times = self.select_time_small()
        if selected_times:
            for i in range(len(self.date_time_pairs)):
                if self.date_time_pairs[i][0] == date:
                    self.date_time_pairs[i] = (date, selected_times[0], selected_times[1])
                    self.add_time_label_small(date, selected_times[0], selected_times[1])
                    break
        else:
            self.date_time_pairs = [dt for dt in self.date_time_pairs if dt[0] != date]
            for i in range(self.date_label_layout.count()):
                item = self.date_label_layout.itemAt(i)
                if isinstance(item.widget(), QWidget) and isinstance(item.widget().layout().itemAt(0).widget(), QLabel) and item.widget().layout().itemAt(0).widget().text() == f"선택된 날짜: {date.toString('yyyy-MM-dd')}":
                    self.date_label_layout.takeAt(i).widget().deleteLater()
                    break

    def add_time_label_small(self, date, start_time, end_time):
        time_layout = QHBoxLayout()
        label = QLabel(f'{date.toString("yyyy-MM-dd")} {start_time.toString("HH:mm:ss")} - {end_time.toString("HH:mm:ss")}')
        delete_button = QPushButton('삭제')
        delete_button.clicked.connect(lambda: self.remove_date_time_small(time_layout, date, start_time, end_time, True))
        time_layout.addWidget(label)
        time_layout.addWidget(delete_button)
        container_widget = QWidget()
        container_widget.setLayout(time_layout)
        self.date_label_layout.addWidget(container_widget)

    def remove_date_time_small(self, layout, date, start_time, end_time, remove_date_label):
        for i in range(self.date_label_layout.count()):
            item = self.date_label_layout.itemAt(i)
            if item.layout() == layout:
                self.date_label_layout.takeAt(i)
                break
        self.date_time_pairs = [dt for dt in self.date_time_pairs if dt != (date, start_time, end_time)]
        self.clear_layout_small(layout)
        if remove_date_label:
            for i in range(self.date_label_layout.count()):
                item = self.date_label_layout.itemAt(i)
                if item is not None and isinstance(item.widget(), QWidget):
                    widget_layout = item.widget().layout()
                    if widget_layout is not None:
                        first_item = widget_layout.itemAt(0)
                        if first_item is not None and isinstance(first_item.widget(), QLabel) and first_item.widget().text() == f"선택된 날짜: {date.toString('yyyy-MM-dd')}":
                            self.date_label_layout.takeAt(i).widget().deleteLater()
                            break

    def select_date_small(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Date')
        layout = QVBoxLayout()
        date_edit = QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QDate.currentDate())
        date_edit.setDisplayFormat('yyyy-MM-dd')
        layout.addWidget(date_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        selected_date = None
        if dialog.exec_() == QDialog.Accepted:
            selected_date = date_edit.date()
        return selected_date

    def select_time_small(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Start and End Time')
        layout = QVBoxLayout()
        start_time_edit = QTimeEdit()
        start_time_edit.setDisplayFormat('HH:mm:ss')
        layout.addWidget(QLabel("Start Time:"))
        layout.addWidget(start_time_edit)
        end_time_edit = QTimeEdit()
        end_time_edit.setDisplayFormat('HH:mm:ss')
        layout.addWidget(QLabel("End Time:"))
        layout.addWidget(end_time_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        selected_times = None
        if dialog.exec_() == QDialog.Accepted:
            selected_times = (start_time_edit.time(), end_time_edit.time())
        else:
            self.date_time_pairs = [dt for dt in self.date_time_pairs if dt[0] != self.current_date]
        return selected_times
    def init_default_time_pairs(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        self.start_time = start_time
        self.end_time = end_time
        self.date_time_pairs = [(self.start_time.date(), self.start_time.time(), self.end_time.time())]

    
    def select_sensor_ids_small(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Chamber and Sensor IDs')

        layout = QVBoxLayout()
        self.sensor_id_vars = []

        all_select_checkbox = QCheckBox('All Select')
        all_select_checkbox.stateChanged.connect(lambda: self.toggle_all_sensor_ids_small(self.sensor_id_vars, all_select_checkbox))
        layout.addWidget(all_select_checkbox)

        # 저장된 start_time과 end_time을 사용하여 date_time_pairs를 생성
        if not self.date_time_pairs:
            self.init_default_time_pairs()

        print(f"Generated date_time_pairs: {self.date_time_pairs}")

        # date_time_pairs에서 시작 시간과 종료 시간 가져오기
        time_pairs = []
        for date, start_time, end_time in self.date_time_pairs:
            start_datetime = QDateTime(date, start_time).toPyDateTime()
            end_datetime = QDateTime(date, end_time).toPyDateTime()
            time_pairs.append((start_datetime, end_datetime))
        print(f"Selected time pairs: {time_pairs}")

        # 시간 범위를 사용하여 데이터 조회
        chamber_data = {}
        for start_time, end_time in time_pairs:
            try:
                result = fetch_chamber_data(start_time, end_time)
                print(f"Fetched chamber data for time pair {start_time} - {end_time}: {result}")
                for chamber_id, sensor_ids in result.items():
                    if chamber_id not in chamber_data:
                        chamber_data[chamber_id] = set()
                    chamber_data[chamber_id].update(sensor_ids)
                print(f"Chamber data aggregated: {chamber_data}")
            except Exception as e:
                print(f"Error: {e}")

        # 키를 기준으로 정렬
        sorted_chamber_data = {k: sorted(list(v)) for k, v in sorted(chamber_data.items())}
        print(f"Sorted chamber data: {sorted_chamber_data}")

        chambers_layout = QHBoxLayout()  # 가로 정렬을 위한 레이아웃

        for chamber_id, sensor_ids in sorted_chamber_data.items():
            chamber_layout = QVBoxLayout()
            chamber_label = QLabel(f'Chamber {chamber_id}')
            chamber_layout.addWidget(chamber_label)

            chamber_select_checkbox = QCheckBox(f'Select Chamber {chamber_id}')
            chamber_select_checkbox.stateChanged.connect(partial(self.toggle_chamber_sensor_ids_small, chamber_id))
            chamber_layout.addWidget(chamber_select_checkbox)

            for sensor_id in sensor_ids:
                checkbox = QCheckBox(str(sensor_id))
                checkbox.setProperty('chamber_id', chamber_id)  # checkbox에 chamber_id를 속성으로 추가
                chamber_layout.addWidget(checkbox)
                self.sensor_id_vars.append(checkbox)

            chambers_layout.addLayout(chamber_layout)  # 각 챔버 레이아웃을 가로 정렬 레이아웃에 추가

        layout.addLayout(chambers_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            self.selected_sensor_data = {}
            for checkbox in self.sensor_id_vars:
                if checkbox.isChecked():
                    sensor_id = int(checkbox.text())
                    chamber_id = checkbox.property('chamber_id')  # checkbox의 속성에서 chamber_id를 가져옴
                    if chamber_id is not None:
                        if chamber_id not in self.selected_sensor_data:
                            self.selected_sensor_data[chamber_id] = []
                        self.selected_sensor_data[chamber_id].append(sensor_id)

            self.sensor_id_list_widget.clear()
            for chamber_id, sensor_ids in self.selected_sensor_data.items():
                for sensor_id in sensor_ids:
                    item = QListWidgetItem(f"Chamber {chamber_id} - Sensor ID {sensor_id}")
                    self.sensor_id_list_widget.addItem(item)

    def toggle_chamber_sensor_ids_small(self, chamber_id, state):
        for checkbox in self.sensor_id_vars:
            if checkbox.property('chamber_id') == chamber_id:
                checkbox.setChecked(state)

    def toggle_all_sensor_ids_small(self, sensor_id_vars, all_select_checkbox):
        state = all_select_checkbox.isChecked()
        for checkbox in sensor_id_vars:
            checkbox.setChecked(state)

    def toggle_polymer_sensor_ids_small(self, df, polymer, state):
        sensor_ids = df[df['applied_polymer'] == polymer]['sensor_id'].tolist()
        for checkbox in self.sensor_id_vars:
            sensor_id = int(checkbox.text())
            if sensor_id in sensor_ids:
                checkbox.setChecked(state)

    def toggle_solvent_sensor_ids_small(self, df, solvent, state):
        sensor_ids = df[df['solvent'] == solvent]['sensor_id'].tolist()
        for checkbox in self.sensor_id_vars:
            sensor_id = int(checkbox.text())
            if sensor_id in sensor_ids:
                checkbox.setChecked(state)


    def visualize_graph_small(self, plot_func):
        try:
            if not self.date_time_pairs:
                raise ValueError("날짜와 시간을 모두 선택해야 합니다.")
            start_times = []
            end_times = []
            for date, start_time, end_time in self.date_time_pairs:
                if start_time is None or end_time is None:
                    raise ValueError("모든 날짜에 대해 시작 시간과 종료 시간을 설정해야 합니다.")
                start_times.append(QDateTime(date, start_time).toPyDateTime())
                end_times.append(QDateTime(date, end_time).toPyDateTime())
            if len(start_times) != len(end_times):
                raise ValueError("시작 시간과 종료 시간의 개수가 일치해야 합니다.")
            for start_time, end_time in zip(start_times, end_times):
                if start_time >= end_time:
                    raise ValueError("시작 시간은 종료 시간보다 빨라야 합니다.")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid input", str(e))
            return

        if not self.selected_sensor_data:
            QMessageBox.critical(self, "Invalid input", "적어도 하나의 Sensor ID를 선택해 주세요.")
            return

        combined_start_time = min(start_times)
        combined_end_time = max(end_times)
        sensor_ids = [sensor_id for sensor_list in self.selected_sensor_data.values() for sensor_id in sensor_list]
        injection_times = query_injection_conditions_small(combined_start_time, combined_end_time, sensor_ids)

        plot_func(self.selected_sensor_data, combined_start_time, combined_end_time, injection_times, self.date_time_pairs)


    def visualize_static_volt_small(self):
        self.visualize_graph_small(plot_data_volt_small)

    def visualize_ratio_volt_small(self):
        self.visualize_graph_small(plot_ratio_data_volt_small)

    def visualize_multi_volt_small(self):
        self.visualize_graph_small(plot_multi_data_volt_small)

    def visualize_static_rs_small(self):
        self.visualize_graph_small(plot_data_rs_small)

    def visualize_ratio_rs_small(self):
        self.visualize_graph_small(plot_ratio_data_rs_small)

    def visualize_multi_rs_small(self):
        self.visualize_graph_small(plot_multi_data_rs_small)


    def real_time_analysis_options_small(self):
        self.clear_layout_small(self.left_frame)
        self.clear_layout_small(self.right_frame)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()

        layout.addWidget(self.canvas)
        self.left_frame.addLayout(layout)
        self.selected_sensor_id = []
        self.date_time_pairs = []

        self.sensor_id_button = QPushButton('Sensor ID 선택')
        
        #self.date_time_pairs = [(QDateTime.currentDateTime().date(),0,0)]
        self.sensor_id_button.clicked.connect(self.select_sensor_ids_small)
        self.sensor_id_scroll_area.setWidget(self.sensor_id_list_widget)
        self.sensor_id_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.right_frame.addWidget(self.sensor_id_button)
        self.right_frame.addWidget(self.sensor_id_scroll_area)

        self.interval_selection = QComboBox()
        self.interval_selection.addItems(["5초", "10초", "30초", "60초"])
        self.right_frame.addWidget(self.interval_selection)

        self.time_selection = QComboBox()
        self.time_selection.addItems(['1 Hour Ago', '2 Hours Ago', '6 Hours Ago', '12 Hours Ago'])
        self.right_frame.addWidget(QLabel("Select Time Range:"))
        self.right_frame.addWidget(self.time_selection)

        self.real_time_button = QPushButton('실시간 분석 시작')
        self.real_time_button.clicked.connect(self.start_real_time_analysis_small)
        self.right_frame.addWidget(self.real_time_button)

        self.stop_real_time_button = QPushButton('실시간 분석 중지')
        self.stop_real_time_button.clicked.connect(self.stop_real_time_analysis_small)
        self.right_frame.addWidget(self.stop_real_time_button)

        self.real_time_button.setEnabled(True)
        self.stop_real_time_button.setEnabled(False)


    def start_real_time_analysis_small(self):
        interval_mapping = {"5초": 5000, "10초": 10000, "30초": 30000, "60초": 60000}
        selected_interval = self.interval_selection.currentText()
        if selected_interval in interval_mapping:
            self.timer.start(interval_mapping[selected_interval])
            self.real_time_button.setEnabled(False)
            self.stop_real_time_button.setEnabled(True)
            self.update_graph_small()

    def stop_real_time_analysis_small(self):
        self.timer.stop()
        self.real_time_button.setEnabled(True)
        self.stop_real_time_button.setEnabled(False)

    def update_graph_small(self):
        time_range = self.time_selection.currentText()

        end_time = datetime.now()
        if time_range == '1 Hour Ago':
            start_time = end_time - timedelta(hours=1)
        elif time_range == '2 Hours Ago':
            start_time = end_time - timedelta(hours=2)
        elif time_range == '6 Hours Ago':
            start_time = end_time - timedelta(hours=6)
        elif time_range == '12 Hours Ago':
            start_time = end_time - timedelta(hours=12)

        if not self.selected_sensor_data:
            QMessageBox.critical(self, "Invalid input", "적어도 하나의 Sensor ID를 선택해 주세요.")
            return

        data = query_real_time_sensor_data_small(start_time, end_time, self.selected_sensor_data)
        if not data:
            print("No data available for plotting.")
            return

        df = pd.DataFrame(data)
        if df.empty:
            print("DataFrame is empty after loading data.")
            return

        df['reg_date'] = pd.to_datetime(df['reg_date'])
        df['time_dt'] = pd.to_datetime(df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

        num_chambers = len(self.selected_sensor_data)
        num_cols = 2
        num_rows = (num_chambers + 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols)

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (chamber_id, sensor_list) in enumerate(self.selected_sensor_data.items()):
            if i >= len(axes):
                break
            ax = axes[i]
            chamber_df = df[df['chamber_id'] == chamber_id]

            for sensor_id in sensor_list:
                sensor_df = chamber_df[chamber_df['sensor_id'] == sensor_id]
                if sensor_df.empty:
                    print(f"No data for sensor ID {sensor_id} in chamber ID {chamber_id}.")
                    continue

                if 'volt' not in sensor_df.columns:
                    print(f"Column 'volt' not found in sensor data for sensor ID {sensor_id} in chamber ID {chamber_id}.")
                    continue

                sensor_df = sensor_df.sort_values(by='time_dt')
                ax.plot(sensor_df['time_dt'], sensor_df['volt'], label=f'Sensor ID {sensor_id}')

            ax.set_title(f'Chamber {chamber_id} Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Voltage')
            ax.legend(loc='best', fontsize='small')
            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        self.canvas.figure = fig
        self.canvas.draw()

        self.start_time = start_time
        self.end_time = end_time