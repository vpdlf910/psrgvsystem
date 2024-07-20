import sys
import os
from PyQt5.QtWidgets import QApplication, QDialog
from gui.login import LoginWindow
from gui.tsei_psrgvsystem import TSEI_PSRGVSystem
import matplotlib
matplotlib.use('Qt5Agg')  # Qt5Agg 백엔드 사용

base_dir = os.path.dirname(os.path.abspath(__file__))
qt_plugin_path = os.path.join(base_dir, 'venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

def main():
    app = QApplication.instance()  # 이미 QApplication 인스턴스가 있는지 확인
    if not app:
        app = QApplication(sys.argv)
    else:
        print("QApplication instance already exists")

    login_window = LoginWindow(lambda: TSEI_PSRGVSystem().show())
    if login_window.exec_() == QDialog.Accepted:
        main_window = TSEI_PSRGVSystem()
        main_window.show()
        sys.exit(app.exec_())  # main_window.show() 다음에 sys.exit(app.exec_())를 호출

if __name__ == "__main__":
    main()
