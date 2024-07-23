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
from datetime import datetime
from utils.database import query_sensor_data,query_sensor_data_small
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,QLabel
from matplotlib.collections import PathCollection

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

# 데이터베이스 쿼리 호출 싱글톤 
class DatabaseQuery:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseQuery, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def query_sensor_data(self, start_time, end_time, sensor_id):
        # 실제 데이터베이스 쿼리 함수 호출
        return query_sensor_data(start_time, end_time, sensor_id)

# 사용 예제
db_query = DatabaseQuery()

def on_pick(event):
    artist = event.artist
    if isinstance(artist, plt.Line2D):
        xdata, ydata = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print(f"Clicked on line: {artist.get_label()} at index: {ind}")
        print(f"Data: x = {xdata[ind]}, y = {ydata[ind]}")
    elif isinstance(artist, PathCollection):
        offsets = artist.get_offsets()
        ind = event.ind
        for i in ind:
            print(f"Clicked on point at: x = {offsets[i][0]}, y = {offsets[i][1]}")

# 데이터를 가져오고 데이터를 평균 값으로 전처리 하는 과정
def prepare_data(selected_sensor_ids, date_time_pairs):
    print(f"prepare_data 호출: selected_sensor_ids={selected_sensor_ids}, date_time_pairs={date_time_pairs}")  # 디버깅 출력
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

    df['date'] = df['reg_date'].dt.date  # Extract date for grouping
    result_df = pd.DataFrame(columns=['sensor_id', 'reg_date', 'avg_volt', 'avg_rs'])

    for (sensor_id, date), group in df.groupby(['sensor_id', 'date']):
        group = group.sort_values('reg_date').reset_index(drop=True)
        group['avg_volt'] = group['volt'].rolling(window=101, min_periods=1, center=True).mean()
        group['avg_rs'] = group['rs'].rolling(window=101, min_periods=1, center=True).mean()

        if not group[['sensor_id', 'reg_date', 'avg_volt', 'avg_rs']].isna().all().all():
            result_df = pd.concat([result_df, group[['sensor_id', 'reg_date', 'avg_volt', 'avg_rs']]], ignore_index=True)

    return result_df

# injection data를 표에 그리는 과정
def configure_ax(ax, combined_start_time, combined_end_time, injection_times, y_max, date_str):
    for injection_type, times in injection_times.items():
        for time in times:
            if time.strftime("%Y-%m-%d") == date_str:
                injection_time = time.replace(year=1970, month=1, day=1)
                if combined_start_time <= injection_time <= combined_end_time:
                    ax.axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in ax.get_legend_handles_labels()[1] else "")
                    ax.text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')

    ax.set_xlabel('Time')
    ax.set_ylabel('avg_volt')
    ax.legend(loc='best', fontsize='small')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 시간 간격 설정
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # 15분 간격 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim(combined_start_time, combined_end_time)
    ax.tick_params(axis='x', rotation=45)

class PlotWindow(QMainWindow):
    def __init__(self, parent=None, title="Plot Window"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas)

        # 버튼 추가
        self.button_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.next_button = QPushButton('Next')
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.next_button)
        self.layout.addLayout(self.button_layout)

def plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=None):
    class SensorDataPlotWindow(PlotWindow):
        def __init__(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent=None):
            self.selected_sensor_ids = selected_sensor_ids
            self.filtered_df = filtered_df
            self.combined_start_time = combined_start_time
            self.combined_end_time = combined_end_time
            self.injection_times = injection_times
            self.date_time_pairs = date_time_pairs
            self.current_index = 0
            self.y_col = y_col
            self.minmax_transform = minmax_transform
            self.transformer = MinMaxScaler() if minmax_transform else None
            super().__init__(parent, title="Sensor Data Plot")
            self.plot_current()

            self.next_button.clicked.connect(self.next_sensor)
            self.back_button.clicked.connect(self.prev_sensor)

        def plot_current(self):
            sensor_id = self.selected_sensor_ids[self.current_index]
            self.plot(sensor_id, self.filtered_df, self.combined_start_time, self.combined_end_time, self.injection_times, self.date_time_pairs, self.y_col, self.minmax_transform)

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

        def plot(self, sensor_id, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform):
            for col, (date, start_time, end_time) in enumerate(date_time_pairs):
                ax = self.canvas.figure.add_subplot(1, len(date_time_pairs), col + 1)
                sensor_df = filtered_df[(filtered_df['sensor_id'] == sensor_id) & (filtered_df['reg_date'].dt.date == date.toPyDate())]
                if sensor_df.empty:
                    continue
                
                y_col_transformed = y_col  # Default to the original column name
                if minmax_transform:
                    # Create a unique minmax column name for this date
                    y_col_transformed = f'{y_col}_minmax'
                    sensor_df[y_col_transformed] = self.transformer.fit_transform(sensor_df[[y_col]])
                
                sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
                sensor_df = sensor_df.copy()
                sensor_df.reset_index(drop=True, inplace=True)
                
                max_idx = sensor_df[y_col_transformed].idxmax()
                min_idx = sensor_df[y_col_transformed].idxmin()
                
                sns.lineplot(data=sensor_df, x='time_dt', y=y_col_transformed, label=f'sensor_id {sensor_id}', ax=ax)
                ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_transformed], color='red', label=f'Max {y_col_transformed}:{sensor_df["reg_date"].iloc[max_idx]}', s=100,picker=5)
                ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_transformed], color='blue', label=f'Min {y_col_transformed}:{sensor_df["reg_date"].iloc[min_idx]}', s=100)
                
                y_min, y_max = ax.get_ylim()
                date_str = date.toString("yyyy-MM-dd")
                configure_ax(ax, combined_start_time, combined_end_time, injection_times, y_max, date_str)
                ax.set_title(f'Separate Data for {date_str}')
                ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left', ncol=1)
                ax.set_ylabel(y_col_transformed)
                self.canvas.mpl_connect('pick_event', on_pick)


                self.tooltip = QLabel('', self)
                self.tooltip.setStyleSheet("background-color: white; border: 1px solid black;")
                self.tooltip.setVisible(False)
            self.canvas.draw()
        def on_motion(self, event):
            if event.inaxes == self.ax:
                visible = False
                for line in [self.line1, self.line2]:
                    cont, ind = line.contains(event)
                    if cont:
                        self.tooltip.setText(line.get_label())
                        self.tooltip.move(event.x + 10, event.y + 10)  # 오프셋 추가
                        self.tooltip.setVisible(True)
                        visible = True
                        break
                if not visible:
                    self.tooltip.setVisible(False)
            else:
                self.tooltip.setVisible(False)


    global sensor_data_plot_window
    sensor_data_plot_window = SensorDataPlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent)
    sensor_data_plot_window.show()

def plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=None):
    class SensorCombinePlotWindow(PlotWindow):
        def __init__(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent=None):
            self.selected_sensor_ids = selected_sensor_ids
            self.filtered_df = filtered_df
            self.combined_start_time = combined_start_time
            self.combined_end_time = combined_end_time
            self.injection_times = injection_times
            self.date_time_pairs = date_time_pairs
            self.minmax_transform = minmax_transform
            self.y_col = y_col
            self.transformer = MinMaxScaler() if minmax_transform else None
            super().__init__(parent, title="Combined Sensor Data Plot")
            self.plot()

        def plot(self):
            ax = self.canvas.figure.add_subplot(1, 1, 1)
            for sensor_id in self.selected_sensor_ids:
                for date, start_time, end_time in self.date_time_pairs:
                    sensor_df = self.filtered_df[(self.filtered_df['sensor_id'] == sensor_id) & (self.filtered_df['reg_date'].dt.date == date.toPyDate())]
                    if sensor_df.empty:
                        continue
                    if self.minmax_transform:
                        sensor_df.loc[:, f'{self.y_col}_minmax'] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                        y_col = f'{self.y_col}_minmax'
                    else:
                        y_col = self.y_col
                    sensor_df.loc[:, 'time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
                    sns.lineplot(data=sensor_df, x='time_dt', y=y_col, label=f'sensor_id {sensor_id}', ax=ax,picker=5)
                    y_min, y_max = ax.get_ylim()
                    for date, _, _ in self.date_time_pairs:
                        date_str = date.toString("yyyy-MM-dd")
                        configure_ax(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str)
                    ax.set_title('Combined Sensor Data')
                    ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left', ncol=1)
                    ax.set_ylabel(y_col)
                    
            self.canvas.draw()

    global sensor_combine_plot_window
    sensor_combine_plot_window = SensorCombinePlotWindow(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent)
    sensor_combine_plot_window.show()

def plot_static_combine_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=parent)

def plot_static_combine_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, parent=parent)

def plot_ratio_combine_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, parent=parent)

def plot_ratio_combine_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, parent=parent)

def plot_data_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=parent)

def plot_data_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, parent=parent)

def plot_ratio_data_volt(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, parent=parent)
    
def plot_ratio_data_rs(selected_sensor_ids: list, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, parent=parent)

def plot_multi_data_volt(selected_sensor_ids, combined_start_time, combined_end_time, injection_times, date_time_pairs, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    class MultiPlotWindow(PlotWindow):
        def __init__(self, parent=None):
            super().__init__(parent, title="Multi Data Plot")
            self.current_pair_index = 0
            self.date_time_pairs = date_time_pairs
            self.plot(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])
            self.back_button.clicked.connect(self.prev_pair)
            self.next_button.clicked.connect(self.next_pair)

        def plot(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, current_pair):
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
                    sensor_df.loc[:, 'avg_volt_minmax'] = transformer.fit_transform(sensor_df[['avg_volt']])
                    sensor_df = sensor_df.copy()
                    sensor_df.reset_index(drop=True, inplace=True)
                    max_volt_idx = sensor_df['avg_volt_minmax'].idxmax()
                    min_volt_idx = sensor_df['avg_volt_minmax'].idxmin()
                    ax.plot(sensor_df['time_dt'], sensor_df['avg_volt_minmax'], label=f'sensor_id {sensor_id}')
                    ax.scatter(sensor_df.loc[max_volt_idx, 'time_dt'], sensor_df.loc[max_volt_idx, 'avg_volt_minmax'], color='red', s=100)
                    ax.scatter(sensor_df.loc[min_volt_idx, 'time_dt'], sensor_df.loc[min_volt_idx, 'avg_volt_minmax'], color='blue', s=100)

            for i in range(num_subplots):
                sensor_ids = selected_sensor_ids[i * 5:(i + 1) * 5]
                plot_lines(axes[i], sensor_ids)
                y_min, y_max = axes[i].get_ylim()
                for injection_type, times in injection_times.items():
                    for time in times:
                        injection_time = time.replace(year=1970, month=1, day=1)
                        if combined_start_time <= injection_time <= combined_end_time:
                            axes[i].axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in axes[i].get_legend_handles_labels()[1] else "")
                            axes[i].text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')
                axes[i].set_title(f'Combined Data for Sensors {sensor_ids}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('avg_volt')
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
                self.plot(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])

        def next_pair(self):
            if self.current_pair_index < len(self.date_time_pairs) - 1:
                self.current_pair_index += 1
                self.plot(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])

    global multi_plot_window
    multi_plot_window = MultiPlotWindow(parent)
    multi_plot_window.show()

def plot_multi_data_rs(selected_sensor_ids, combined_start_time, combined_end_time, injection_times, date_time_pairs, parent=None):
    result_df = prepare_data(selected_sensor_ids, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin(selected_sensor_ids)]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    class MultiPlotWindow(PlotWindow):
        def __init__(self, parent=None):
            super().__init__(parent, title="Multi Data Plot")
            self.plot(selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs)

        def plot(self, selected_sensor_ids, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs):
            dates = filtered_df['reg_date'].dt.date.unique()
            for current_date in dates:
                date_filtered_df = filtered_df[filtered_df['reg_date'].dt.date == current_date]
                num_subplots = (len(selected_sensor_ids) - 1) // 5 + 1
                num_subplots = min(num_subplots, 8)
                num_cols = 2
                num_rows = (num_subplots + 1) // num_cols
                fig = self.canvas.figure
                axes = []
                for i in range(1, num_subplots + 1):
                    ax = fig.add_subplot(num_rows, num_cols, i)
                    axes.append(ax)
                fig.set_size_inches(15, 5 * num_rows)
                fig.subplots_adjust(wspace=0.3)

                def plot_lines(ax, sensor_ids):
                    for sensor_id in sensor_ids:
                        sensor_df = date_filtered_df[date_filtered_df['sensor_id'] == sensor_id]
                        sensor_df.loc[:, 'avg_rs_minmax'] = transformer.fit_transform(sensor_df[['avg_rs']])
                        sensor_df = sensor_df.copy()
                        sensor_df.reset_index(drop=True, inplace=True)
                        max_rs_idx = sensor_df['avg_rs_minmax'].idxmax()
                        min_rs_idx = sensor_df['avg_rs_minmax'].idxmin()
                        ax.plot(sensor_df['time_dt'], sensor_df['avg_rs_minmax'], label=f'sensor_id {sensor_id}',picker=5)
                        ax.scatter(sensor_df.loc[max_rs_idx, 'time_dt'], sensor_df.loc[max_rs_idx, 'avg_rs_minmax'], color='red', s=100)
                        ax.scatter(sensor_df.loc[min_rs_idx, 'time_dt'], sensor_df.loc[min_rs_idx, 'avg_rs_minmax'], color='blue', s=100)

                for i in range(num_subplots):
                    sensor_ids = selected_sensor_ids[i * 5:(i + 1) * 5]
                    plot_lines(axes[i], sensor_ids)
                    y_min, y_max = axes[i].get_ylim()
                    for injection_type, times in injection_times.items():
                        for time in times:
                            injection_time = time.replace(year=1970, month=1, day=1)
                            if combined_start_time <= injection_time <= combined_end_time:
                                axes[i].axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in axes[i].get_legend_handles_labels()[1] else "")
                                axes[i].text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')
                    axes[i].set_title(f'Combined Data for Sensors {sensor_ids}')
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel('avg_rs')
                    axes[i].legend(loc='best', fontsize='small')
                    axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    axes[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    axes[i].set_xlim(combined_start_time, combined_end_time)
                    axes[i].tick_params(axis='x', rotation=45)

                for j in range(num_subplots, len(axes)):
                    fig.delaxes(axes[j])

                fig.suptitle(f'Plots for {current_date}')
                plt.tight_layout()
                self.canvas.draw()

    global multi_plot_window
    multi_plot_window = MultiPlotWindow(parent)
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
        group['avg_volt'] = group.groupby('sensor_id')['volt'].transform(lambda x: x.rolling(window=101, min_periods=1, center=True).mean())
        group['avg_rs'] = group.groupby('sensor_id')['rs'].transform(lambda x: x.rolling(window=101, min_periods=1, center=True).mean())
        chamber_dataframes[chamber_id] = group

    return chamber_dataframes

def configure_ax_small(ax, combined_start_time, combined_end_time, injection_times, y_max, date_str):
    for injection_type, times in injection_times.items():
        for time in times:
            if time.strftime("%Y-%m-%d") == date_str:
                injection_time = time.replace(year=1970, month=1, day=1)
                if combined_start_time <= injection_time <= combined_end_time:
                    ax.axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in ax.get_legend_handles_labels()[1] else "")
                    ax.text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')

    ax.set_xlabel('Time')
    ax.set_ylabel('avg_volt')
    ax.legend(loc='best', fontsize='small')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 시간 간격 설정
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # 15분 간격 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim(combined_start_time, combined_end_time)
    ax.tick_params(axis='x', rotation=45)

class PlotWindowSmall(QMainWindow):
    def __init__(self, parent=None, title="Plot Window"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas)

        # 버튼 추가
        self.button_layout = QHBoxLayout()
        self.back_button = QPushButton('Back')
        self.next_button = QPushButton('Next')
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.next_button)
        self.layout.addLayout(self.button_layout)

def plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=None):
    class SensorDataPlotWindowSmall(PlotWindowSmall):
        def __init__(self, chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent=None):
            self.chamber_dataframes = chamber_dataframes
            self.chamber_ids = list(chamber_dataframes.keys())
            self.combined_start_time = combined_start_time
            self.combined_end_time = combined_end_time
            self.injection_times = injection_times
            self.date_time_pairs = date_time_pairs
            self.current_index = 0
            self.y_col = y_col
            self.minmax_transform = minmax_transform
            self.transformer = MinMaxScaler() if minmax_transform else None
            super().__init__(parent, title="Sensor Data Plot")
            self.plot_current()

            self.next_button.clicked.connect(self.next_chamber)
            self.back_button.clicked.connect(self.prev_chamber)

        def plot_current(self):
            current_chamber_id = self.chamber_ids[self.current_index]
            current_df = self.chamber_dataframes[current_chamber_id]
            self.plot_chamber_data(current_df, current_chamber_id)

        def next_chamber(self):
            if self.current_index < len(self.chamber_ids) - 1:
                self.current_index += 1
                self.canvas.figure.clear()
                self.plot_current()

        def prev_chamber(self):
            if self.current_index > 0:
                self.current_index -= 1
                self.canvas.figure.clear()
                self.plot_current()

        def plot_chamber_data(self, df, chamber_id):
            sensor_ids = df['sensor_id'].unique()
            print(f"Plotting for sensors: {sensor_ids} in chamber_id: {chamber_id}")  # 디버깅 메시지 추가
            legend_added = False

            transformed_data = {}

            for col, (date, start_time, end_time) in enumerate(self.date_time_pairs):
                ax = self.canvas.figure.add_subplot(1, len(self.date_time_pairs), col + 1)
                for sensor_id in sensor_ids:
                    sensor_df = df[
                        (df['sensor_id'] == sensor_id) & 
                        (df['reg_date'].dt.date == date.toPyDate())
                    ]
                    print(f"Sensor ID: {sensor_id}, Filtered Data:\n{sensor_df.head()}")  # 디버깅 메시지 추가
                    if sensor_df.empty:
                        print(f"No data for Sensor ID: {sensor_id} on {date.toPyDate()} in Chamber ID: {chamber_id}")
                        continue

                    y_col_transformed = self.y_col
                    if self.minmax_transform:
                        y_col_transformed = f'{self.y_col}_minmax'
                        if sensor_id not in transformed_data:
                            transformed_data[sensor_id] = sensor_df.copy()
                            transformed_data[sensor_id][y_col_transformed] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                        sensor_df = transformed_data[sensor_id]

                    sensor_df['time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

                    print(f"Plotting sensor data for Sensor ID {sensor_id}:\n{sensor_df[['time_dt', y_col_transformed]].head()}")  # 디버깅 메시지 추가

                    max_idx = sensor_df[y_col_transformed].idxmax()
                    min_idx = sensor_df[y_col_transformed].idxmin()

                    sns.lineplot(data=sensor_df, x='time_dt', y=y_col_transformed, label=f'sensor_id {sensor_id}', ax=ax)
                    ax.scatter(sensor_df.loc[max_idx, 'time_dt'], sensor_df.loc[max_idx, y_col_transformed], color='red', s=100, picker=5)
                    ax.scatter(sensor_df.loc[min_idx, 'time_dt'], sensor_df.loc[min_idx, y_col_transformed], color='blue', s=100)

                y_min, y_max = ax.get_ylim()
                date_str = date.toString("yyyy-MM-dd")
                configure_ax_small(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str)
                ax.set_title(f'Chamber {chamber_id} - Data for {date_str}')
                if not legend_added:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
                    legend_added = True
                ax.set_ylabel(y_col_transformed)
                # self.canvas.mpl_connect('pick_event', self.on_pick_small)

                self.tooltip = QLabel('', self)
                self.tooltip.setStyleSheet("background-color: white; border: 1px solid black;")
                self.tooltip.setVisible(False)
            self.canvas.draw()

        def on_pick_small(event):
            artist = event.artist
            if isinstance(artist, plt.Line2D):
                xdata, ydata = artist.get_xdata(), artist.get_ydata()
                ind = event.ind
                print(f"Clicked on line: {artist.get_label()} at index: {ind}")
                print(f"Data: x = {xdata[ind]}, y = {ydata[ind]}")
            elif isinstance(artist, PathCollection):
                offsets = artist.get_offsets()
                ind = event.ind
                for i in ind:
                    print(f"Clicked on point at: x = {offsets[i][0]}, y = {offsets[i][1]}")
        
        def on_motion(self, event):
            if event.inaxes == self.ax:
                visible = False
                for line in [self.line1, self.line2]:
                    cont, ind = line.contains(event)
                    if cont:
                        self.tooltip.setText(line.get_label())
                        self.tooltip.move(event.x + 10, event.y + 10)
                        self.tooltip.setVisible(True)
                        visible = True
                        break
                if not visible:
                    self.tooltip.setVisible(False)
            else:
                self.tooltip.setVisible(False)

    global sensor_data_plot_window_small
    sensor_data_plot_window_small = SensorDataPlotWindowSmall(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent)
    sensor_data_plot_window_small.show()

def plot_sensor_combine_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=None):
    class SensorCombinePlotWindowSmall(PlotWindowSmall):
        def __init__(self, chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent=None):
            self.chamber_sensor_data = chamber_sensor_data
            self.filtered_df = filtered_df
            self.combined_start_time = combined_start_time
            self.combined_end_time = combined_end_time
            self.injection_times = injection_times
            self.date_time_pairs = date_time_pairs
            self.minmax_transform = minmax_transform
            self.y_col = y_col
            self.transformer = MinMaxScaler() if minmax_transform else None
            super().__init__(parent, title="Combined Sensor Data Plot")
            self.plot()

        def plot(self):
            ax = self.canvas.figure.add_subplot(1, 1, 1)
            for chamber_id, sensor_ids in self.chamber_sensor_data.items():
                for sensor_id in sensor_ids:
                    for date, start_time, end_time in self.date_time_pairs:
                        sensor_df = self.filtered_df[(self.filtered_df['sensor_id'] == sensor_id) & (self.filtered_df['reg_date'].dt.date == date.toPyDate())]
                        if sensor_df.empty:
                            continue
                        if self.minmax_transform:
                            sensor_df.loc[:, f'{self.y_col}_minmax'] = self.transformer.fit_transform(sensor_df[[self.y_col]])
                            y_col = f'{self.y_col}_minmax'
                        else:
                            y_col = self.y_col
                        sensor_df.loc[:, 'time_dt'] = pd.to_datetime(sensor_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))
                        sns.lineplot(data=sensor_df, x='time_dt', y=y_col, label=f'Chamber {chamber_id} - Sensor ID {sensor_id}', ax=ax, picker=5)
                        y_min, y_max = ax.get_ylim()
                        for date, _, _ in self.date_time_pairs:
                            date_str = date.toString("yyyy-MM-dd")
                            configure_ax_small(ax, self.combined_start_time, self.combined_end_time, self.injection_times, y_max, date_str)
                        ax.set_title('Combined Sensor Data')
                        ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left', ncol=1)
                        ax.set_ylabel(y_col)
                    
            self.canvas.draw()

    global sensor_combine_plot_window_small
    sensor_combine_plot_window_small = SensorCombinePlotWindowSmall(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col, minmax_transform, parent)
    sensor_combine_plot_window_small.show()

def plot_static_combine_volt_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=parent)

def plot_static_combine_rs_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, parent=parent)

def plot_ratio_combine_volt_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, parent=parent)

def plot_ratio_combine_rs_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_combine_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, parent=parent)

def plot_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, parent=None):
    chamber_dataframes = prepare_data_small(chamber_sensor_data, date_time_pairs)
    if not chamber_dataframes:
        print("No data available for plotting.")
        return
    
    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_by_chamber(chamber_dataframes, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=False, parent=parent)

def plot_data_rs_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=False, parent=parent)

def plot_ratio_data_volt_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_volt', minmax_transform=True, parent=parent)
    
def plot_ratio_data_rs_small(chamber_sensor_data: dict, combined_start_time: datetime, combined_end_time: datetime, injection_times: dict, date_time_pairs: list, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    filtered_df.loc[:, 'time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    plot_sensor_data_small(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs, y_col='avg_rs', minmax_transform=True, parent=parent)

def plot_multi_data_volt_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    class MultiPlotWindowSmall(PlotWindowSmall):
        def __init__(self, parent=None):
            super().__init__(parent, title="Multi Data Plot")
            self.current_pair_index = 0
            self.date_time_pairs = date_time_pairs
            self.plot(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])
            self.back_button.clicked.connect(self.prev_pair)
            self.next_button.clicked.connect(self.next_pair)

        def plot(self, chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, current_pair):
            self.canvas.figure.clf()
            date, start_time, end_time = current_pair
            date_filtered_df = filtered_df[filtered_df['reg_date'].dt.date == date.toPyDate()]
            num_subplots = (len([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids]) - 1) // 5 + 1
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
                    sensor_df.loc[:, 'avg_volt_minmax'] = transformer.fit_transform(sensor_df[['avg_volt']])
                    sensor_df = sensor_df.copy()
                    sensor_df.reset_index(drop=True, inplace=True)
                    max_volt_idx = sensor_df['avg_volt_minmax'].idxmax()
                    min_volt_idx = sensor_df['avg_volt_minmax'].idxmin()
                    ax.plot(sensor_df['time_dt'], sensor_df['avg_volt_minmax'], label=f'sensor_id {sensor_id}')
                    ax.scatter(sensor_df.loc[max_volt_idx, 'time_dt'], sensor_df.loc[max_volt_idx, 'avg_volt_minmax'], color='red', s=100)
                    ax.scatter(sensor_df.loc[min_volt_idx, 'time_dt'], sensor_df.loc[min_volt_idx, 'avg_volt_minmax'], color='blue', s=100)

            for i in range(num_subplots):
                sensor_ids = [sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids][i * 5:(i + 1) * 5]
                plot_lines(axes[i], sensor_ids)
                y_min, y_max = axes[i].get_ylim()
                for injection_type, times in injection_times.items():
                    for time in times:
                        injection_time = time.replace(year=1970, month=1, day=1)
                        if combined_start_time <= injection_time <= combined_end_time:
                            axes[i].axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in axes[i].get_legend_handles_labels()[1] else "")
                            axes[i].text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')
                axes[i].set_title(f'Combined Data for Sensors {sensor_ids}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('avg_volt')
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
                self.plot(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])

        def next_pair(self):
            if self.current_pair_index < len(self.date_time_pairs) - 1:
                self.current_pair_index += 1
                self.plot(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, self.date_time_pairs[self.current_pair_index])

    global multi_plot_window_small
    multi_plot_window_small = MultiPlotWindowSmall(parent)
    multi_plot_window_small.show()

def plot_multi_data_rs_small(chamber_sensor_data, combined_start_time, combined_end_time, injection_times, date_time_pairs, parent=None):
    result_df = prepare_data_small(chamber_sensor_data, date_time_pairs)
    filtered_df = result_df[result_df['sensor_id'].isin([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids])]
    transformer = MinMaxScaler()
    filtered_df['time_dt'] = pd.to_datetime(filtered_df['reg_date'].dt.strftime('1970-01-01 %H:%M:%S'))

    combined_start_time = combined_start_time.replace(year=1970, month=1, day=1)
    combined_end_time = combined_end_time.replace(year=1970, month=1, day=1)

    class MultiPlotWindowSmall(PlotWindowSmall):
        def __init__(self, parent=None):
            super().__init__(parent, title="Multi Data Plot")
            self.plot(chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs)

        def plot(self, chamber_sensor_data, filtered_df, combined_start_time, combined_end_time, injection_times, date_time_pairs):
            dates = filtered_df['reg_date'].dt.date.unique()
            for current_date in dates:
                date_filtered_df = filtered_df[filtered_df['reg_date'].dt.date == current_date]
                num_subplots = (len([sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids]) - 1) // 5 + 1
                num_subplots = min(num_subplots, 8)
                num_cols = 2
                num_rows = (num_subplots + 1) // num_cols
                fig = self.canvas.figure
                axes = []
                for i in range(1, num_subplots + 1):
                    ax = fig.add_subplot(num_rows, num_cols, i)
                    axes.append(ax)
                fig.set_size_inches(15, 5 * num_rows)
                fig.subplots_adjust(wspace=0.3)

                def plot_lines(ax, sensor_ids):
                    for sensor_id in sensor_ids:
                        sensor_df = date_filtered_df[date_filtered_df['sensor_id'] == sensor_id]
                        sensor_df.loc[:, 'avg_rs_minmax'] = transformer.fit_transform(sensor_df[['avg_rs']])
                        sensor_df = sensor_df.copy()
                        sensor_df.reset_index(drop=True, inplace=True)
                        max_rs_idx = sensor_df['avg_rs_minmax'].idxmax()
                        min_rs_idx = sensor_df['avg_rs_minmax'].idxmin()
                        ax.plot(sensor_df['time_dt'], sensor_df['avg_rs_minmax'], label=f'sensor_id {sensor_id}', picker=5)
                        ax.scatter(sensor_df.loc[max_rs_idx, 'time_dt'], sensor_df.loc[max_rs_idx, 'avg_rs_minmax'], color='red', s=100)
                        ax.scatter(sensor_df.loc[min_rs_idx, 'time_dt'], sensor_df.loc[min_rs_idx, 'avg_rs_minmax'], color='blue', s=100)

                for i in range(num_subplots):
                    sensor_ids = [sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids][i * 5:(i + 1) * 5]
                    plot_lines(axes[i], sensor_ids)
                    y_min, y_max = axes[i].get_ylim()
                    for injection_type, times in injection_times.items():
                        for time in times:
                            injection_time = time.replace(year=1970, month=1, day=1)
                            if combined_start_time <= injection_time <= combined_end_time:
                                axes[i].axvline(x=injection_time, linestyle='--', color='r', label=f'{injection_type}' if injection_type not in axes[i].get_legend_handles_labels()[1] else "")
                                axes[i].text(injection_time, y_max, f'{injection_type}', rotation=45, verticalalignment='bottom', color='r')
                    axes[i].set_title(f'Combined Data for Sensors {sensor_ids}')
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel('avg_rs')
                    axes[i].legend(loc='best', fontsize='small')
                    axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    axes[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    axes[i].set_xlim(combined_start_time, combined_end_time)
                    axes[i].tick_params(axis='x', rotation=45)

                for j in range(num_subplots, len(axes)):
                    fig.delaxes(axes[j])

                fig.suptitle(f'Plots for {current_date}')
                plt.tight_layout()
                self.canvas.draw()

    global multi_plot_window_small
    multi_plot_window_small = MultiPlotWindowSmall(parent)
    multi_plot_window_small.show()