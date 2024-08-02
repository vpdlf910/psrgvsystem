import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Qt5Agg 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from PyQt5.QtCore import QDateTime
from datetime import datetime,timedelta
from utils.database import query_sensor_data, query_sensor_data_small
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from matplotlib.collections import PathCollection
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
from scipy.signal import find_peaks, savgol_filter
from utils.configure_ax import configure_ax

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

# 데이터베이스 쿼리 호출 싱글톤
class DatabaseQuery:
    """싱글톤 패턴을 사용한 데이터베이스 쿼리 클래스"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseQuery, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def query_sensor_data(self, start_time, end_time, sensor_id):
        return query_sensor_data(start_time, end_time, sensor_id)

db_query = DatabaseQuery()

def format_timedelta(td):
        """Convert timedelta to formatted string HH:MM:SS"""
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

# 데이터를 가져오고 데이터를 평균 값으로 전처리 하는 과정
def prepare_data(selected_sensor_ids, date_time_pairs):
    """센서 데이터를 준비하고 평균값으로 전처리"""
    sensor_data = []
    for date, start_time, end_time in date_time_pairs:
        data = db_query.query_sensor_data(QDateTime(date, start_time).toPyDateTime(), QDateTime(date, end_time).toPyDateTime(), selected_sensor_ids)
        sensor_data.extend(data)

    if not sensor_data:
        print("센서 데이터가 비어 있습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(sensor_data)
    if df.empty:
        print("DataFrame is empty after conversion from sensor_data.")
        return pd.DataFrame()

    df['date'] = df['reg_date'].dt.date
    result_df = pd.DataFrame(columns=['sensor_id', 'reg_date', 'avg_volt', 'avg_rs'])

    for (sensor_id, date), group in df.groupby(['sensor_id', 'date']):
        group = group.sort_values('reg_date').reset_index(drop=True)
        group['avg_volt'] = group['volt'].rolling(window=201, min_periods=1, center=True).mean()
        group['avg_rs'] = group['rs'].rolling(window=201, min_periods=1, center=True).mean()
        # group['avg_volt'] = savgol_filter(group['avg_volt'], window_length=501, polyorder=3)
        # group['avg_rs'] = savgol_filter(group['avg_rs'], window_length=501, polyorder=3)
        
        result_df = pd.concat([result_df, group[['sensor_id', 'reg_date', 'avg_volt', 'avg_rs']]], ignore_index=True)

    return result_df

class PlotWindow(QMainWindow):
    """데이터 플롯을 위한 메인 윈도우"""
    def __init__(self, parent=None, title="Plot Window"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas)
        # Add NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.button_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.next_button = QPushButton('Next')
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.next_button)
        self.layout.addLayout(self.button_layout)

        self.text_obj = None
        self.saved_y_value = None  # To store the first clicked y value

class MultiPlotWindow(PlotWindow):
    def __init__(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, transformer, chamber_type, parent=None):
        super().__init__(parent, title="Multi Data Plot")
        self.selected_sensor_ids = selected_sensor_ids
        self.filtered_df = filtered_df
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.date_time_pairs = date_time_pairs
        self.y_col = y_col
        self.transformer = transformer
        self.chamber_type = chamber_type
        self.current_pair_index = 0
        self.plot(self.selected_sensor_ids, self.filtered_df, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs[self.current_pair_index], self.chamber_type)
        self.back_button.clicked.connect(self.prev_pair)
        self.next_button.clicked.connect(self.next_pair)

    def plot(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, current_pair, chamber_type):
        self.canvas.figure.clf()
        date, start_time, end_time = current_pair
        date_filtered_df = filtered_df[filtered_df['reg_date'].dt.date == date.toPyDate()]
        num_subplots = (len(selected_sensor_ids) - 1) // 5 + 1
        num_subplots = min(num_subplots, 8)
        num_cols = 2
        num_rows = (num_subplots + 1) // num_cols
        fig = self.canvas.figure
        axes = []
        for i in range(1, num_subplots + 1):
            ax = fig.add_subplot(num_rows, num_cols, i)
            axes.append(ax)

        def plot_lines(ax, sensor_ids):
            for sensor_id in sensor_ids:
                sensor_df = date_filtered_df[date_filtered_df['sensor_id'] == sensor_id]
                sensor_df.loc[:, f'{self.y_col}_minmax'] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                sensor_df = sensor_df.copy()
                sensor_df.reset_index(drop=True, inplace=True)
                max_idx = sensor_df[f'{self.y_col}_minmax'].idxmax()
                min_idx = sensor_df[f'{self.y_col}_minmax'].idxmin()
                ax.plot(sensor_df['time_dt'], sensor_df[f'{self.y_col}_minmax'], label=f'sensor_id {sensor_id}', picker=5)
                ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, f'{self.y_col}_minmax'], color='red', s=100, picker=5)
                ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, f'{self.y_col}_minmax'], color='blue', s=100, picker=5)

        for i in range(num_subplots):
            sensor_ids = selected_sensor_ids[i * 5:(i + 1) * 5]
            plot_lines(axes[i], sensor_ids)
            y_min, y_max = axes[i].get_ylim()
            date_str = date.toString("yyyy-MM-dd")
            configure_ax(axes[i], combined_start_time, combined_end_time, injection_times, y_max, date_str, specific_chamber_type=chamber_type)
            axes[i].set_title(f'Combined Data for Sensors {sensor_ids}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(self.y_col)
            axes[i].legend(loc='best', fontsize='small')
            axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axes[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[i].set_xlim(combined_start_time, combined_end_time)
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(num_subplots, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Plots for {date.toString("yyyy-MM-dd")}')
        plt.tight_layout()
        self.canvas.draw()

    def prev_pair(self):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.plot(self.selected_sensor_ids, self.filtered_df, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs[self.current_pair_index], self.chamber_type)

    def next_pair(self):
        if self.current_pair_index < len(self.date_time_pairs) - 1:
            self.current_pair_index += 1
            self.plot(self.selected_sensor_ids, self.filtered_df, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs[self.current_pair_index], self.chamber_type)

class SensorDataPlotWindow(PlotWindow):
    def __init__(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, chamber_type, parent=None):
        self.selected_sensor_ids = selected_sensor_ids
        self.filtered_df = filtered_df
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.date_time_pairs = date_time_pairs
        self.current_index = 0
        self.y_col = y_col
        self.minmax_transform = minmax_transform
        self.chamber_type = chamber_type
        self.transformer = MinMaxScaler() if minmax_transform else None
        super().__init__(parent, title="Sensor Data Plot")
        self.plot_current()

        self.next_button.clicked.connect(self.next_sensor)
        self.back_button.clicked.connect(self.prev_sensor)

        self.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        """데이터 포인트 클릭 이벤트 핸들러"""
        artist = event.artist
        if isinstance(artist, plt.Line2D):
            xdata, ydata = artist.get_xdata(), artist.get_ydata()
            ind = event.ind
            
            if self.saved_y_value is None:
                # 세 번째 클릭 시 기존 점과 텍스트 제거
                if hasattr(self, 'first_point') and self.first_point:
                    self.first_point.remove()
                    self.first_point = None
                if hasattr(self, 'second_point') and self.second_point:
                    self.second_point.remove()
                    self.second_point = None
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                    self.text_obj = None
                self.canvas.draw()

                self.saved_y_value = ydata[ind][0]
                self.saved_x_value = xdata[ind][0]
                print(f"First click: x = {self.saved_x_value}, y = {self.saved_y_value}")
                
                # 첫 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.first_point, = ax.plot(self.saved_x_value, self.saved_y_value, 'ro')
                self.canvas.draw()

            else:
                current_y_value = ydata[ind][0]
                current_x_value = xdata[ind][0]
                y_diff = current_y_value - self.saved_y_value

                # Convert x_diff to timedelta if it's a numpy.timedelta64 object
                x_diff = current_x_value - self.saved_x_value
                if isinstance(x_diff, np.timedelta64):
                    x_diff = timedelta(seconds=x_diff / np.timedelta64(1, 's'))
                    x_diff_formatted = format_timedelta(x_diff)  # 시:분:초 형식으로 변환
                else:
                    x_diff_formatted = str(x_diff)

                print(f"Second click: x = {current_x_value}, y = {current_y_value}, Δx = {x_diff_formatted}, Δy = {y_diff}")

                # 두 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.second_point, = ax.plot(current_x_value, current_y_value, 'go')
                self.canvas.draw()

                self.saved_y_value = None
                self.saved_x_value = None

                # Δx, Δy 값을 캔버스에 표시
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                self.text_obj = ax.text(
                    0.5, 0.95, f'Δx = {x_diff_formatted}, Δy = {y_diff:.4f}', 
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
                )
                self.canvas.draw()

            # 세 번째 클릭을 위한 상태 초기화
            self.third_click = not hasattr(self, 'third_click')

        elif isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            ind = event.ind
            for i in ind:
                x_value = offsets[i][0]
                y_value = offsets[i][1]
                if isinstance(x_value, datetime):
                    x_value = x_value.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Clicked on point at: x = {x_value}, y = {y_value}")


    def plot_current(self):
        sensor_id = self.selected_sensor_ids[self.current_index]
        self.plot(sensor_id, self.filtered_df, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs, self.y_col, self.minmax_transform, self.chamber_type)

    def next_sensor(self):
        if self.current_index < len(self.selected_sensor_ids) - 1:
            self.current_index += 1
            self.canvas.figure.clear()
            self.plot_current()

    def prev_sensor(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.canvas.figure.clear()
            self.plot_current()

    def plot_peaks_and_valleys(self, ax, sensor_df, y_col_to_plot):
        # 피크 및 저점 탐지
        num_points = len(sensor_df)
        distance_value = num_points // 100

        median_value = sensor_df[y_col_to_plot].median()
        min_value = sensor_df[y_col_to_plot].min()
        max_value = sensor_df[y_col_to_plot].max()
        midpoint_value = (min_value + max_value) / 2
        prominence_value = midpoint_value * 0.01  # 중간값의 %로 prominence 설정

        # 피크와 저점 찾기
        peaks, _ = find_peaks(sensor_df[y_col_to_plot], distance=distance_value, prominence=prominence_value)
        inverted_data = -sensor_df[y_col_to_plot]
        valleys, _ = find_peaks(inverted_data, distance=distance_value, prominence=prominence_value)

        change_threshold_relative = sensor_df[y_col_to_plot].median() * 0.1  # 중앙값에서 1% 변동 기준
        change_threshold_absolute = 0.0004  # 절대적 변화 기준 설정 (예: 0.0004)

        # 각 피크에 대해 가장 가까운 이전 저점을 찾아 연결
        for peak_idx in peaks:
            # 가장 가까운 이전 저점 찾기
            previous_valleys = valleys[valleys < peak_idx]
            if len(previous_valleys) == 0:
                continue  # 이전 저점이 없으면 건너뜀
            closest_valley_idx = previous_valleys[-1]

            # 저항 변화 계산
            peak_resistance = sensor_df[y_col_to_plot].iloc[peak_idx]
            valley_resistance = sensor_df[y_col_to_plot].iloc[closest_valley_idx]
            resistance_diff = peak_resistance - valley_resistance  # ΔΩ을 계산

            # 저항 변화값 기준을 초과하는 경우에만 표시
            if resistance_diff >= max(change_threshold_relative, change_threshold_absolute):
                try:
                    # 피크 최고점에서 수직으로 저점까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[peak_idx], sensor_df['time_dt'].iloc[peak_idx]], [sensor_df[y_col_to_plot].iloc[peak_idx], valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    # 저점에서 수평으로 피크 정점 시간까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[closest_valley_idx], sensor_df['time_dt'].iloc[peak_idx]], [valley_resistance, valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    
                    ax.scatter(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], color='red', zorder=5)
                    ax.text(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], f'ΔΩ: {resistance_diff:.5f}', color='red', fontsize=9)
                except Exception as e:
                    print(f"Error plotting peak: {e}")

    def plot(self, sensor_id, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, chamber_type):
        for col, (date, start_time, end_time) in enumerate(date_time_pairs):
            ax = self.canvas.figure.add_subplot(1, len(date_time_pairs), col + 1)
            sensor_df = filtered_df[(filtered_df['sensor_id'] == sensor_id) & (filtered_df['reg_date'].dt.date == date.toPyDate())]
            if sensor_df.empty:
                continue

            y_col_transformed = y_col  # Default to the original column name
            if minmax_transform:
                y_col_transformed = f'{y_col}_minmax'
                sensor_df[y_col_transformed] = self.transformer.fit_transform(sensor_df[[y_col]])

            sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
            sensor_df = sensor_df.copy()
            sensor_df.reset_index(drop=True, inplace=True)
            
            y_col_to_plot = y_col_transformed

            max_idx = sensor_df[y_col_to_plot].idxmax()
            min_idx = sensor_df[y_col_to_plot].idxmin()

            ax.plot(sensor_df['time_dt'], sensor_df[y_col_to_plot], label=f'sensor_id {sensor_id}', picker=5)
            ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_to_plot], label=f'max: {sensor_df.loc[max_idx, y_col_to_plot]}',color='red', s=100, picker=5)
            ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_to_plot], label=f'min: {sensor_df.loc[min_idx, y_col_to_plot]}',color='blue', s=10, picker=5)
            
            # 피크 및 저점 찾기 및 시각화 함수 호출
            self.plot_peaks_and_valleys(ax, sensor_df, y_col_to_plot)
            
            y_min, y_max = ax.get_ylim()
            date_str = date.toString("yyyy-MM-dd")
            configure_ax(ax, combined_start_time, combined_end_time, injection_times, y_max, date_str, specific_chamber_type=chamber_type)
            ax.set_title(f'Separate Data for {date_str}')
            ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left', ncol=1)
            ax.set_ylabel(y_col_to_plot)

            self.tooltip = QLabel('', self)
            self.tooltip.setStyleSheet("background-color: white; border: 1px solid black;")
            self.tooltip.setVisible(False)
        self.canvas.draw()


class SensorCombinePlotWindow(PlotWindow):
    def __init__(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, chamber_type, parent=None):
        self.selected_sensor_ids = selected_sensor_ids
        self.filtered_df = filtered_df
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.date_time_pairs = date_time_pairs
        self.minmax_transform = minmax_transform
        self.y_col = y_col
        self.chamber_type = chamber_type
        self.transformer = MinMaxScaler() if minmax_transform else None
        super().__init__(parent, title="Combined Sensor Data Plot")
        self.plot()

        self.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        """데이터 포인트 클릭 이벤트 핸들러"""
        artist = event.artist
        if isinstance(artist, plt.Line2D):
            xdata, ydata = artist.get_xdata(), artist.get_ydata()
            ind = event.ind
            
            if self.saved_y_value is None:
                # 세 번째 클릭 시 기존 점과 텍스트 제거
                if hasattr(self, 'first_point') and self.first_point:
                    self.first_point.remove()
                    self.first_point = None
                if hasattr(self, 'second_point') and self.second_point:
                    self.second_point.remove()
                    self.second_point = None
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                    self.text_obj = None
                self.canvas.draw()

                self.saved_y_value = ydata[ind][0]
                self.saved_x_value = xdata[ind][0]
                print(f"First click: x = {self.saved_x_value}, y = {self.saved_y_value}")
                
                # 첫 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.first_point, = ax.plot(self.saved_x_value, self.saved_y_value, 'ro')
                self.canvas.draw()

            else:
                current_y_value = ydata[ind][0]
                current_x_value = xdata[ind][0]
                y_diff = current_y_value - self.saved_y_value

                # Convert x_diff to timedelta if it's a numpy.timedelta64 object
                x_diff = current_x_value - self.saved_x_value
                if isinstance(x_diff, np.timedelta64):
                    x_diff = timedelta(seconds=x_diff / np.timedelta64(1, 's'))
                    x_diff_formatted = format_timedelta(x_diff)  # 시:분:초 형식으로 변환
                else:
                    x_diff_formatted = str(x_diff)

                print(f"Second click: x = {current_x_value}, y = {current_y_value}, Δx = {x_diff_formatted}, Δy = {y_diff}")

                # 두 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.second_point, = ax.plot(current_x_value, current_y_value, 'go')
                self.canvas.draw()

                self.saved_y_value = None
                self.saved_x_value = None

                # Δx, Δy 값을 캔버스에 표시
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                self.text_obj = ax.text(
                    0.5, 0.95, f'Δx = {x_diff_formatted}, Δy = {y_diff:.4f}', 
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
                )
                self.canvas.draw()

            # 세 번째 클릭을 위한 상태 초기화
            self.third_click = not hasattr(self, 'third_click')

        elif isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            ind = event.ind
            for i in ind:
                x_value = offsets[i][0]
                y_value = offsets[i][1]
                if isinstance(x_value, datetime):
                    x_value = x_value.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Clicked on point at: x = {x_value}, y = {y_value}")

    def plot_peaks_and_valleys(self, ax, sensor_df, y_col_to_plot):
        # 피크 및 저점 탐지
        num_points = len(sensor_df)
        distance_value = num_points // 100

        median_value = sensor_df[y_col_to_plot].median()
        min_value = sensor_df[y_col_to_plot].min()
        max_value = sensor_df[y_col_to_plot].max()
        midpoint_value = (min_value + max_value) / 2
        prominence_value = midpoint_value * 0.01  # 중간값의 %로 prominence 설정

        # 피크와 저점 찾기
        peaks, _ = find_peaks(sensor_df[y_col_to_plot], distance=distance_value, prominence=prominence_value)
        inverted_data = -sensor_df[y_col_to_plot]
        valleys, _ = find_peaks(inverted_data, distance=distance_value, prominence=prominence_value)

        change_threshold_relative = sensor_df[y_col_to_plot].median() * 0.1  # 중앙값에서 1% 변동 기준
        change_threshold_absolute = 0.0004  # 절대적 변화 기준 설정 (예: 0.0004)

        # 각 피크에 대해 가장 가까운 이전 저점을 찾아 연결
        for peak_idx in peaks:
            # 가장 가까운 이전 저점 찾기
            previous_valleys = valleys[valleys < peak_idx]
            if len(previous_valleys) == 0:
                continue  # 이전 저점이 없으면 건너뜀
            closest_valley_idx = previous_valleys[-1]

            # 저항 변화 계산
            peak_resistance = sensor_df[y_col_to_plot].iloc[peak_idx]
            valley_resistance = sensor_df[y_col_to_plot].iloc[closest_valley_idx]
            resistance_diff = peak_resistance - valley_resistance  # ΔΩ을 계산

            # 저항 변화값 기준을 초과하는 경우에만 표시
            if resistance_diff >= max(change_threshold_relative, change_threshold_absolute):
                try:
                    # 피크 최고점에서 수직으로 저점까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[peak_idx], sensor_df['time_dt'].iloc[peak_idx]], [sensor_df[y_col_to_plot].iloc[peak_idx], valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    # 저점에서 수평으로 피크 정점 시간까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[closest_valley_idx], sensor_df['time_dt'].iloc[peak_idx]], [valley_resistance, valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    
                    ax.scatter(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], color='red', zorder=5)
                    ax.text(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], f'ΔΩ: {resistance_diff:.5f}', color='red', fontsize=9)
                except Exception as e:
                    print(f"Error plotting peak: {e}")

    def plot(self):
        ax = self.canvas.figure.add_subplot(1, 1, 1)
        for sensor_id in self.selected_sensor_ids:
            for date, start_time, end_time in self.date_time_pairs:
                sensor_df = self.filtered_df[(self.filtered_df['sensor_id'] == sensor_id) & (self.filtered_df['reg_date'].dt.date == date.toPyDate())]
                if sensor_df.empty:
                    continue
                if self.minmax_transform:
                    sensor_df.loc[:, f'{self.y_col}_minmax'] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                    y_col_transformed = f'{self.y_col}_minmax'
                else:
                    y_col_transformed = self.y_col
                sensor_df.loc[:, 'time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
                sensor_df = sensor_df.copy()
                sensor_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
                y_col_to_plot = y_col_transformed
                sns.lineplot(data=sensor_df, x='time_dt', y=y_col_transformed, label=f'sensor_id {sensor_id}', ax=ax, picker=5)
                
                # 피크 및 저점 찾기 및 시각화 함수 호출
                self.plot_peaks_and_valleys(ax, sensor_df, y_col_to_plot)

                y_min, y_max = ax.get_ylim()
                for date, _, _ in self.date_time_pairs:
                    date_str = date.toString("yyyy-MM-dd")
                    configure_ax(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str, specific_chamber_type=self.chamber_type)
                ax.set_title('Combined Sensor Data')
                ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left', ncol=1)
                ax.set_ylabel(y_col_transformed)
                
        self.canvas.draw()

def plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, chamber_type=None, parent=None):
    global sensor_data_plot_window
    sensor_data_plot_window = SensorDataPlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, chamber_type, parent)
    sensor_data_plot_window.show()

def plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, chamber_type=None, parent=None):
    global sensor_combine_plot_window
    sensor_combine_plot_window = SensorCombinePlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, chamber_type, parent)
    sensor_combine_plot_window.show()

def plot_static_combine_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, chamber_type=chamber_type, parent=parent)

def plot_static_combine_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, chamber_type=chamber_type, parent=parent)

def plot_ratio_combine_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, chamber_type=chamber_type, parent=parent)

def plot_ratio_combine_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, chamber_type=chamber_type, parent=parent)

def plot_data_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, chamber_type=chamber_type, parent=parent)

def plot_data_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, chamber_type=chamber_type, parent=parent)

def plot_ratio_data_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, chamber_type=chamber_type, parent=parent)
    
def plot_ratio_data_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, chamber_type=chamber_type, parent=parent)

def plot_multi_data_volt(selected_sensor_ids, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    global multi_plot_window
    multi_plot_window = MultiPlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', transformer=transformer, chamber_type=chamber_type, parent=parent)
    multi_plot_window.show()

def plot_multi_data_rs(selected_sensor_ids, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type='40channel', parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    global multi_plot_window
    multi_plot_window = MultiPlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', transformer=transformer, chamber_type=chamber_type, parent=parent)
    multi_plot_window.show()

class DatabaseQuerySmall:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseQuerySmall, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def query_sensor_data(self, start_time, end_time, chamber_sensor_data):
        # 실제 데이터베이스 쿼리 함수 호출
        return query_sensor_data_small(start_time, end_time, chamber_sensor_data)

# 사용 예제
db_query_small = DatabaseQuerySmall()

def prepare_data_small(chamber_sensor_data, date_time_pairs):
    print(f"prepare_data_small 호출: chamber_sensor_data={chamber_sensor_data}, date_time_pairs={date_time_pairs}")  # 디버깅 출력
    sensor_data = []
    for date, start_time, end_time in date_time_pairs:
        data = db_query_small.query_sensor_data(QDateTime(date, start_time).toPyDateTime(), QDateTime(date, end_time).toPyDateTime(), chamber_sensor_data)
        sensor_data.extend(data)

    if not sensor_data:
        print("센서 데이터가 비어 있습니다.")
        return {}

    df = pd.DataFrame(sensor_data)
    if df.empty:
        print("DataFrame is empty after conversion from sensor_data.")
        return {}

    df['chamber_id'] = df['chamber_id'].astype(int)
    df['sensor_id'] = df['sensor_id'].astype(int)
    df['date'] = df['reg_date'].dt.date  # Extract date for grouping
    chamber_dataframes = {}

    for chamber_id, group in df.groupby('chamber_id'):
        group = group.sort_values('reg_date').reset_index(drop=True)
        group['avg_volt'] = group.groupby('sensor_id')['volt'].transform(lambda x: x.rolling(window=201, min_periods=1, center=True).mean())
        group['avg_rs'] = group.groupby('sensor_id')['rs'].transform(lambda x: x.rolling(window=201, min_periods=1, center=True).mean())
        # group['avg_volt'] = savgol_filter(group['avg_volt'], window_length=501, polyorder=3)
        # group['avg_rs'] = savgol_filter(group['avg_rs'], window_length=501, polyorder=3)
        chamber_dataframes[chamber_id] = group

    return chamber_dataframes

class PlotWindowSmall(QMainWindow):
    def __init__(self, parent=None, title="Plot Window"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas)

        # Add NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.text_obj = None
        self.saved_y_value = None  # To store the first clicked y value

        # 버튼 추가
        self.button_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.next_button = QPushButton('Next')
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.next_button)
        self.layout.addLayout(self.button_layout)

class SensorDataPlotWindowSmall(PlotWindowSmall):
    def __init__(self, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, separate=False, parent=None):
        self.chamber_dataframes = chamber_dataframes
        self.chamber_ids = list(chamber_dataframes.keys())
        self.date_time_pairs = date_time_pairs
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.chamber_type = chamber_type
        self.current_chamber_index = 0
        self.current_date_index = 0
        self.y_col = y_col
        self.minmax_transform = minmax_transform
        self.separate = separate
        self.transformer = MinMaxScaler() if minmax_transform else None
        super().__init__(parent, title="Sensor Data Plot")
        self.plot_current()

        self.next_button.clicked.connect(self.next_plot)
        self.back_button.clicked.connect(self.prev_plot)

    def on_pick(self, event):
        """데이터 포인트 클릭 이벤트 핸들러"""
        artist = event.artist
        if isinstance(artist, plt.Line2D):
            xdata, ydata = artist.get_xdata(), artist.get_ydata()
            ind = event.ind
            
            if self.saved_y_value is None:
                # 세 번째 클릭 시 기존 점과 텍스트 제거
                if hasattr(self, 'first_point') and self.first_point:
                    self.first_point.remove()
                    self.first_point = None
                if hasattr(self, 'second_point') and self.second_point:
                    self.second_point.remove()
                    self.second_point = None
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                    self.text_obj = None
                self.canvas.draw()

                self.saved_y_value = ydata[ind][0]
                self.saved_x_value = xdata[ind][0]
                print(f"First click: x = {self.saved_x_value}, y = {self.saved_y_value}")
                
                # 첫 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.first_point, = ax.plot(self.saved_x_value, self.saved_y_value, 'ro')
                self.canvas.draw()

            else:
                current_y_value = ydata[ind][0]
                current_x_value = xdata[ind][0]
                y_diff = current_y_value - self.saved_y_value

                # Convert x_diff to timedelta if it's a numpy.timedelta64 object
                x_diff = current_x_value - self.saved_x_value
                if isinstance(x_diff, np.timedelta64):
                    x_diff = timedelta(seconds=x_diff / np.timedelta64(1, 's'))
                    x_diff_formatted = format_timedelta(x_diff)  # 시:분:초 형식으로 변환
                else:
                    x_diff_formatted = str(x_diff)

                print(f"Second click: x = {current_x_value}, y = {current_y_value}, Δx = {x_diff_formatted}, Δy = {y_diff}")

                # 두 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.second_point, = ax.plot(current_x_value, current_y_value, 'go')
                self.canvas.draw()

                self.saved_y_value = None
                self.saved_x_value = None

                # Δx, Δy 값을 캔버스에 표시
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                self.text_obj = ax.text(
                    0.5, 0.95, f'Δx = {x_diff_formatted}, Δy = {y_diff:.4f}', 
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
                )
                self.canvas.draw()

            # 세 번째 클릭을 위한 상태 초기화
            self.third_click = not hasattr(self, 'third_click')

        elif isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            ind = event.ind
            for i in ind:
                x_value = offsets[i][0]
                y_value = offsets[i][1]
                if isinstance(x_value, datetime):
                    x_value = x_value.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Clicked on point at: x = {x_value}, y = {y_value}")

    def plot_current(self):
        current_chamber_id = self.chamber_ids[self.current_chamber_index]
        current_date_pair = self.date_time_pairs[self.current_date_index]
        current_df = self.chamber_dataframes[current_chamber_id]
        if self.separate:
            self.plot_chamber_data_separate(current_df, current_chamber_id, current_date_pair, self.chamber_type)
        else:
            self.plot_chamber_data(current_df, current_chamber_id, current_date_pair, self.chamber_type)

    def next_plot(self):
        if self.current_date_index < len(self.date_time_pairs) - 1:
            self.current_date_index += 1
        elif self.current_chamber_index < len(self.chamber_ids) - 1:
            self.current_date_index = 0
            self.current_chamber_index += 1
        else:
            return
        self.canvas.figure.clear()
        self.plot_current()

    def prev_plot(self):
        if self.current_date_index > 0:
            self.current_date_index -= 1
        elif self.current_chamber_index > 0:
            self.current_chamber_index -= 1
            self.current_date_index = len(self.date_time_pairs) - 1
        else:
            return
        self.canvas.figure.clear()
        self.plot_current()

    def plot_peaks_and_valleys(self, ax, sensor_df, y_col_to_plot):
        # 피크 및 저점 탐지
        num_points = len(sensor_df)
        distance_value = num_points // 100

        median_value = sensor_df[y_col_to_plot].median()
        min_value = sensor_df[y_col_to_plot].min()
        max_value = sensor_df[y_col_to_plot].max()
        midpoint_value = (min_value + max_value) / 2
        prominence_value = midpoint_value * 0.01  # 중간값의 %로 prominence 설정

        # 피크와 저점 찾기
        peaks, _ = find_peaks(sensor_df[y_col_to_plot], distance=distance_value, prominence=prominence_value)
        inverted_data = -sensor_df[y_col_to_plot]
        valleys, _ = find_peaks(inverted_data, distance=distance_value, prominence=prominence_value)

        change_threshold_relative = sensor_df[y_col_to_plot].median() * 0.1  # 중앙값에서 1% 변동 기준
        change_threshold_absolute = 0.0004  # 절대적 변화 기준 설정 (예: 0.0004)

        # 각 피크에 대해 가장 가까운 이전 저점을 찾아 연결
        for peak_idx in peaks:
            # 가장 가까운 이전 저점 찾기
            previous_valleys = valleys[valleys < peak_idx]
            if len(previous_valleys) == 0:
                continue  # 이전 저점이 없으면 건너뜀
            closest_valley_idx = previous_valleys[-1]

            # 저항 변화 계산
            peak_resistance = sensor_df[y_col_to_plot].iloc[peak_idx]
            valley_resistance = sensor_df[y_col_to_plot].iloc[closest_valley_idx]
            resistance_diff = peak_resistance - valley_resistance  # ΔΩ을 계산

            # 저항 변화값 기준을 초과하는 경우에만 표시
            if resistance_diff >= max(change_threshold_relative, change_threshold_absolute):
                try:
                    # 피크 최고점에서 수직으로 저점까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[peak_idx], sensor_df['time_dt'].iloc[peak_idx]], [sensor_df[y_col_to_plot].iloc[peak_idx], valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    # 저점에서 수평으로 피크 정점 시간까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[closest_valley_idx], sensor_df['time_dt'].iloc[peak_idx]], [valley_resistance, valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    
                    ax.scatter(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], color='red', zorder=5)
                    ax.text(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], f'ΔΩ: {resistance_diff:.5f}', color='red', fontsize=9)
                except Exception as e:
                    print(f"Error plotting peak: {e}")

    def plot_chamber_data(self, df, chamber_id, date_pair, chamber_type, show_legend=True):
        sensor_ids = df['sensor_id'].unique()
        num_sensors = len(sensor_ids)
        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax = fig.add_subplot(1, 1, 1)
        self.canvas.mpl_connect('pick_event', self.on_pick)

        qdate, qstart_time, qend_time = date_pair
        start_datetime = QDateTime(qdate, qstart_time).toPyDateTime()
        end_datetime = QDateTime(qdate, qend_time).toPyDateTime()

        for sensor_id in sensor_ids:
            sensor_df = df[(df['sensor_id'] == sensor_id) & (df['reg_date'] >= start_datetime) & (df['reg_date'] <= end_datetime)]
            if sensor_df.empty:
                continue

            y_col_transformed = self.y_col
            if self.minmax_transform:
                y_col_transformed = f'{self.y_col}_minmax'
                sensor_df[y_col_transformed] = self.transformer.fit_transform(sensor_df[[self.y_col]])

            sensor_df = sensor_df.copy()
            sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

            y_col_to_plot = y_col_transformed

            sns.lineplot(data=sensor_df, x='time_dt', y=y_col_to_plot, label=f'Sensor {sensor_id}', ax=ax, picker=5)
            max_idx = sensor_df[y_col_to_plot].idxmax()
            min_idx = sensor_df[y_col_to_plot].idxmin()
            ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_to_plot],label=f'max: {sensor_df.loc[max_idx, y_col_to_plot]}' ,color='red', s=100, picker=5)
            ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_to_plot],label=f'min: {sensor_df.loc[min_idx, y_col_to_plot]}' ,color='blue', s=100, picker=5)

            # 피크 및 저점 찾기 및 시각화 함수 호출
            self.plot_peaks_and_valleys(ax, sensor_df, y_col_to_plot)

        y_min, y_max = ax.get_ylim()
        date_str = qdate.toString("yyyy-MM-dd")
        configure_ax(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str, specific_chamber_id=chamber_id, specific_chamber_type=chamber_type, show_legend=show_legend)
        ax.set_title(f'Chamber {chamber_id} - Sensor Data - {date_str}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0.5)
        ax.set_ylabel('Normalized Voltage')
        ax.set_xlabel('Time')

        self.canvas.draw()

    def plot_chamber_data_separate(self, df, chamber_id, date_pair, chamber_type):
        sensor_ids = df['sensor_id'].unique()
        num_sensors = len(sensor_ids)
        num_cols = 2
        num_rows = (num_sensors + num_cols - 1) // num_cols
        self.canvas.figure.clear()
        fig = self.canvas.figure

        qdate, qstart_time, qend_time = date_pair
        start_datetime = QDateTime(qdate, qstart_time).toPyDateTime()
        end_datetime = QDateTime(qdate, qend_time).toPyDateTime()

        axes = []
        for i in range(num_sensors):
            row, col = divmod(i, num_cols)
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            axes.append(ax)
            sensor_id = sensor_ids[i]
            sensor_df = df[(df['sensor_id'] == sensor_id) & (df['reg_date'] >= start_datetime) & (df['reg_date'] <= end_datetime)]
            if sensor_df.empty:
                ax.axis('off')
                continue

            y_col_transformed = self.y_col
            if self.minmax_transform:
                y_col_transformed = f'{self.y_col}_minmax'
                sensor_df[y_col_transformed] = self.transformer.fit_transform(sensor_df[[self.y_col]])

            sensor_df = sensor_df.copy()
            sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

            y_col_to_plot = y_col_transformed

            sns.lineplot(data=sensor_df, x='time_dt', y=y_col_to_plot, label=f'Sensor {sensor_id}', ax=ax, picker=5, legend=False)
            max_idx = sensor_df[y_col_to_plot].idxmax()
            min_idx = sensor_df[y_col_to_plot].idxmin()
            ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_to_plot],label=f'max: {sensor_df.loc[max_idx, y_col_to_plot]}' ,color='red', s=100, picker=5)
            ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_to_plot],label=f'min: {sensor_df.loc[min_idx, y_col_to_plot]}' ,color='blue', s=100, picker=5)

            # 피크 및 저점 찾기 및 시각화 함수 호출
            self.plot_peaks_and_valleys(ax, sensor_df, y_col_to_plot)

            y_min, y_max = ax.get_ylim()
            date_str = qdate.toString("yyyy-MM-dd")
            configure_ax(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str, specific_chamber_id=chamber_id, specific_chamber_type=chamber_type, show_legend=False)
            ax.set_title(f'Chamber {chamber_id} - Sensor {sensor_id} - {date_str}')
            ax.set_ylabel(y_col_to_plot)
            ax.set_xlabel('Time')
            

        self.canvas.draw()


class MultiPlotWindowSmall(PlotWindowSmall):
    def __init__(self, chamber_sensor_data, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, parent=None):
        self.chamber_sensor_data = chamber_sensor_data
        self.chamber_dataframes = chamber_dataframes
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.date_time_pairs = date_time_pairs
        self.chamber_type = chamber_type
        self.current_pair_index = 0
        self.y_col = y_col
        self.minmax_transform = minmax_transform
        self.transformer = MinMaxScaler() if minmax_transform else None
        super().__init__(parent, title="Multi Data Plot")
        self.plot_current()
        self.back_button.clicked.connect(self.prev_pair)
        self.next_button.clicked.connect(self.next_pair)

    def plot_current(self):
        self.plot(self.chamber_sensor_data, self.chamber_dataframes, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs[self.current_pair_index], self.chamber_type)

    def plot(self, chamber_sensor_data, chamber_dataframes, combined_start_time, combined_end_time, injection_times, current_pair, chamber_type):
        self.canvas.figure.clf()
        date, start_time, end_time = current_pair
        num_chambers = len(chamber_sensor_data)
        num_cols = 2
        num_rows = (num_chambers + 1) // num_cols
        fig = self.canvas.figure
        axes = []

        for i in range(num_chambers):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            axes.append(ax)

        def plot_chamber(ax, chamber_id, sensor_ids):
            for sensor_id in sensor_ids:
                df = chamber_dataframes[chamber_id]
                sensor_df = df[(df['sensor_id'] == sensor_id) & (df['reg_date'].dt.date == date.toPyDate())]
                if sensor_df.empty:
                    continue
                if self.minmax_transform:
                    sensor_df.loc[:, f'{self.y_col}_minmax'] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                    y_col_transformed = f'{self.y_col}_minmax'
                else:
                    y_col_transformed = self.y_col
                sensor_df = sensor_df.copy()
                sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
                sensor_df.reset_index(drop=True, inplace=True)
                max_idx = sensor_df[y_col_transformed].idxmax()
                min_idx = sensor_df[y_col_transformed].idxmin()
                ax.plot(sensor_df['time_dt'], sensor_df[y_col_transformed], label=f'sensor_id {sensor_id}', picker=5)
                ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_transformed], color='red', s=100, picker=5)
                ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_transformed], color='blue', s=100, picker=5)
            y_min, y_max = ax.get_ylim()
            date_str = date.toString("yyyy-MM-dd")
            configure_ax(ax, combined_start_time, combined_end_time, injection_times, y_max, date_str, specific_chamber_id=chamber_id, specific_chamber_type=chamber_type, show_legend=False)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)  # 범례를 그래프 위쪽에 배치

        for i, chamber_id in enumerate(chamber_sensor_data.keys()):
            plot_chamber(axes[i], chamber_id, chamber_sensor_data[chamber_id])

        fig.suptitle(f'Plots for {date.toString("yyyy-MM-dd")}')
        plt.tight_layout()
        self.canvas.draw()

    def prev_pair(self):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.plot_current()

    def next_pair(self):
        if self.current_pair_index < len(self.date_time_pairs) - 1:
            self.current_pair_index += 1
            self.plot_current()
    
class SensorDataPlotWindowSmall_Enumerate(PlotWindowSmall):
    def __init__(self, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, separate=False, parent=None):
        self.chamber_dataframes = chamber_dataframes
        self.chamber_ids = list(chamber_dataframes.keys())
        self.date_time_pairs = date_time_pairs
        self.combined_start_time = combined_start_time
        self.combined_end_time = combined_end_time
        self.injection_times = injection_times
        self.chamber_type = chamber_type
        self.current_chamber_index = 0
        self.current_date_index = 0
        self.current_sensor_index = 0
        self.y_col = y_col
        self.minmax_transform = minmax_transform
        self.separate = separate
        self.transformer = MinMaxScaler() if minmax_transform else None
        super().__init__(parent, title="Sensor Data Plot")
        self.plot_current()

        self.next_button.clicked.connect(self.next_plot)
        self.back_button.clicked.connect(self.prev_plot)

    def on_pick(self, event):
        """데이터 포인트 클릭 이벤트 핸들러"""
        artist = event.artist
        if isinstance(artist, plt.Line2D):
            xdata, ydata = artist.get_xdata(), artist.get_ydata()
            ind = event.ind
            
            if self.saved_y_value is None:
                # 세 번째 클릭 시 기존 점과 텍스트 제거
                if hasattr(self, 'first_point') and self.first_point:
                    self.first_point.remove()
                    self.first_point = None
                if hasattr(self, 'second_point') and self.second_point:
                    self.second_point.remove()
                    self.second_point = None
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                    self.text_obj = None
                self.canvas.draw()

                self.saved_y_value = ydata[ind][0]
                self.saved_x_value = xdata[ind][0]
                print(f"First click: x = {self.saved_x_value}, y = {self.saved_y_value}")
                
                # 첫 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.first_point, = ax.plot(self.saved_x_value, self.saved_y_value, 'ro')
                self.canvas.draw()

            else:
                current_y_value = ydata[ind][0]
                current_x_value = xdata[ind][0]
                y_diff = current_y_value - self.saved_y_value

                # Convert x_diff to timedelta if it's a numpy.timedelta64 object
                x_diff = current_x_value - self.saved_x_value
                if isinstance(x_diff, np.timedelta64):
                    x_diff = timedelta(seconds=x_diff / np.timedelta64(1, 's'))
                    x_diff_formatted = format_timedelta(x_diff)  # 시:분:초 형식으로 변환
                else:
                    x_diff_formatted = str(x_diff)

                print(f"Second click: x = {current_x_value}, y = {current_y_value}, Δx = {x_diff_formatted}, Δy = {y_diff}")

                # 두 번째 클릭 시 점 표시
                ax = self.canvas.figure.gca()
                self.second_point, = ax.plot(current_x_value, current_y_value, 'go')
                self.canvas.draw()

                self.saved_y_value = None
                self.saved_x_value = None

                # Δx, Δy 값을 캔버스에 표시
                if hasattr(self, 'text_obj') and self.text_obj:
                    self.text_obj.remove()
                self.text_obj = ax.text(
                    0.5, 0.95, f'Δx = {x_diff_formatted}, Δy = {y_diff:.4f}', 
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
                )
                self.canvas.draw()

            # 세 번째 클릭을 위한 상태 초기화
            self.third_click = not hasattr(self, 'third_click')

        elif isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            ind = event.ind
            for i in ind:
                x_value = offsets[i][0]
                y_value = offsets[i][1]
                if isinstance(x_value, datetime):
                    x_value = x_value.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Clicked on point at: x = {x_value}, y = {y_value}")

    def plot_current(self):
        current_chamber_id = self.chamber_ids[self.current_chamber_index]
        current_date_pair = self.date_time_pairs[self.current_date_index]
        current_df = self.chamber_dataframes[current_chamber_id]
        current_sensor_id = current_df['sensor_id'].unique()[self.current_sensor_index]

        if self.separate:
            self.plot_chamber_data_separate(current_df, current_chamber_id, current_date_pair, self.chamber_type, current_sensor_id)
        else:
            self.plot_chamber_data(current_df, current_chamber_id, current_date_pair, self.chamber_type, current_sensor_id)

    def next_plot(self):
        current_chamber_id = self.chamber_ids[self.current_chamber_index]
        current_df = self.chamber_dataframes[current_chamber_id]
        sensor_ids = current_df['sensor_id'].unique()

        if self.current_sensor_index < len(sensor_ids) - 1:
            self.current_sensor_index += 1
        else:
            self.current_sensor_index = 0
            if self.current_date_index < len(self.date_time_pairs) - 1:
                self.current_date_index += 1
            elif self.current_chamber_index < len(self.chamber_ids) - 1:
                self.current_date_index = 0
                self.current_chamber_index += 1
            else:
                return

        self.canvas.figure.clear()
        self.plot_current()

    def prev_plot(self):
        current_chamber_id = self.chamber_ids[self.current_chamber_index]
        current_df = self.chamber_dataframes[current_chamber_id]
        sensor_ids = current_df['sensor_id'].unique()

        if self.current_sensor_index > 0:
            self.current_sensor_index -= 1
        else:
            self.current_sensor_index = len(sensor_ids) - 1
            if self.current_date_index > 0:
                self.current_date_index -= 1
            elif self.current_chamber_index > 0:
                self.current_chamber_index -= 1
                self.current_date_index = len(self.date_time_pairs) - 1
            else:
                return

        self.canvas.figure.clear()
        self.plot_current()

    def plot_peaks_and_valleys(self, ax, sensor_df, y_col_to_plot):
        # 피크 및 저점 탐지
        num_points = len(sensor_df)
        distance_value = num_points // 100

        median_value = sensor_df[y_col_to_plot].median()
        min_value = sensor_df[y_col_to_plot].min()
        max_value = sensor_df[y_col_to_plot].max()
        midpoint_value = (min_value + max_value) / 2
        prominence_value = midpoint_value * 0.01  # 중간값의 %로 prominence 설정

        # 피크와 저점 찾기
        peaks, _ = find_peaks(sensor_df[y_col_to_plot], distance=distance_value, prominence=prominence_value)
        inverted_data = -sensor_df[y_col_to_plot]
        valleys, _ = find_peaks(inverted_data, distance=distance_value, prominence=prominence_value)

        change_threshold_relative = sensor_df[y_col_to_plot].median() * 0.1  # 중앙값에서 1% 변동 기준
        change_threshold_absolute = 0.0004  # 절대적 변화 기준 설정 (예: 0.0004)

        # 각 피크에 대해 가장 가까운 이전 저점을 찾아 연결
        for peak_idx in peaks:
            # 가장 가까운 이전 저점 찾기
            previous_valleys = valleys[valleys < peak_idx]
            if len(previous_valleys) == 0:
                continue  # 이전 저점이 없으면 건너뜀
            closest_valley_idx = previous_valleys[-1]

            # 저항 변화 계산
            peak_resistance = sensor_df[y_col_to_plot].iloc[peak_idx]
            valley_resistance = sensor_df[y_col_to_plot].iloc[closest_valley_idx]
            resistance_diff = peak_resistance - valley_resistance  # ΔΩ을 계산

            # 저항 변화값 기준을 초과하는 경우에만 표시
            if resistance_diff >= max(change_threshold_relative, change_threshold_absolute):
                try:
                    # 피크 최고점에서 수직으로 저점까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[peak_idx], sensor_df['time_dt'].iloc[peak_idx]], [sensor_df[y_col_to_plot].iloc[peak_idx], valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    # 저점에서 수평으로 피크 정점 시간까지 실선
                    ax.plot([sensor_df['time_dt'].iloc[closest_valley_idx], sensor_df['time_dt'].iloc[peak_idx]], [valley_resistance, valley_resistance], color='red', linestyle='-', linewidth=0.5)
                    
                    ax.scatter(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], color='red', zorder=5)
                    ax.text(sensor_df['time_dt'].iloc[peak_idx], sensor_df[y_col_to_plot].iloc[peak_idx], f'ΔΩ: {resistance_diff:.5f}', color='red', fontsize=9)
                except Exception as e:
                    print(f"Error plotting peak: {e}")

    def plot_chamber_data(self, df, chamber_id, date_pair, chamber_type, sensor_id, show_legend=True):
        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax = fig.add_subplot(1, 1, 1)
        self.canvas.mpl_connect('pick_event', self.on_pick)

        qdate, qstart_time, qend_time = date_pair
        start_datetime = QDateTime(qdate, qstart_time).toPyDateTime()
        end_datetime = QDateTime(qdate, qend_time).toPyDateTime()

        sensor_df = df[(df['sensor_id'] == sensor_id) & (df['reg_date'] >= start_datetime) & (df['reg_date'] <= end_datetime)]
        if sensor_df.empty:
            ax.set_title(f'Chamber {chamber_id} - Sensor {sensor_id} - No Data')
            self.canvas.draw()
            return

        y_col_transformed = self.y_col
        if self.minmax_transform:
            y_col_transformed = f'{self.y_col}_minmax'
            sensor_df[y_col_transformed] = self.transformer.fit_transform(sensor_df[[self.y_col]])

        sensor_df = sensor_df.copy()
        sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

        y_col_to_plot = y_col_transformed

        sns.lineplot(data=sensor_df, x='time_dt', y=y_col_to_plot, label=f'Sensor {sensor_id}', ax=ax, picker=5)
        max_idx = sensor_df[y_col_to_plot].idxmax()
        min_idx = sensor_df[y_col_to_plot].idxmin()
        ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_to_plot],label=f'max: {sensor_df.loc[max_idx, y_col_to_plot]}' ,color='red', s=100, picker=5)
        ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_to_plot],label=f'min: {sensor_df.loc[min_idx, y_col_to_plot]}' ,color='blue', s=100, picker=5)

        # 피크 및 저점 찾기 및 시각화 함수 호출
        self.plot_peaks_and_valleys(ax, sensor_df, y_col_to_plot)

        y_min, y_max = ax.get_ylim()
        date_str = qdate.toString("yyyy-MM-dd")
        configure_ax(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str, specific_chamber_id=chamber_id, specific_chamber_type=chamber_type, show_legend=show_legend)
        ax.set_title(f'Chamber {chamber_id} - Sensor Data - {date_str}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0.5)
        ax.set_ylabel('Normalized Voltage')
        ax.set_xlabel('Time')

        self.canvas.draw()


def plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=False, separate=False, parent=None):
    global sensor_data_plot_window_small
    sensor_data_plot_window_small = SensorDataPlotWindowSmall(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, separate, parent)
    sensor_data_plot_window_small.show()

def plot_sensor_data_by_chamber_separate(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=False, separate=True, parent=None):
    global sensor_data_plot_window_small
    sensor_data_plot_window_small = SensorDataPlotWindowSmall(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, separate, parent)
    sensor_data_plot_window_small.show()

def plot_sensor_data_by_chamber_enumerate(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=False, separate=False, parent=None):
    global sensor_data_plot_window_small
    sensor_data_plot_window_small = SensorDataPlotWindowSmall_Enumerate(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col, minmax_transform, separate, parent)
    sensor_data_plot_window_small.show()

def plot_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=False, separate=separate, parent=parent)

def plot_data_rs_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_rs', minmax_transform=False, separate=separate, parent=parent)

def plot_ratio_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=True, separate=separate, parent=parent)

def plot_ratio_data_rs_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_rs', minmax_transform=True, separate=separate, parent=parent)

def plot_data_volt_small_separate(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=True, parent=parent)

def plot_data_rs_small_separate(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)
    plot_data_rs_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=True, parent=parent)

def plot_multi_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return
    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    global multi_plot_window_small
    multi_plot_window_small = MultiPlotWindowSmall(chamber_sensor_data, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=True, parent=parent)
    multi_plot_window_small.show()

def plot_multi_data_rs_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return
    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    global multi_plot_window_small
    multi_plot_window_small = MultiPlotWindowSmall(chamber_sensor_data, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_rs', minmax_transform=True, parent=parent)
    multi_plot_window_small.show()

def plot_data_volt_small_enumerate(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber_enumerate(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_volt', minmax_transform=False, separate=separate, parent=parent)

def plot_data_rs_small_enumerate(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, separate=False, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber_enumerate(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, chamber_type, y_col='avg_rs', minmax_transform=False, separate=separate, parent=parent)
