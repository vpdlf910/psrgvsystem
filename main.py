import sys
import os
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from gui.login import LoginWindow
from gui.tsei_psrgvsystem import TSEI_PSRGVSystem, ChamberSelectionDialog
from gui.tsei_psrgvsystem import TSEI_PSRGVSystem_small
import matplotlib
matplotlib.use('Qt5Agg')  # Qt5Agg 백엔드 사용

base_dir = os.path.dirname(os.path.abspath(__file__))
qt_plugin_path = os.path.join(base_dir, 'venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

class MainApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.login_window = LoginWindow(self.on_login_success)

    def on_login_success(self):
        dialog = ChamberSelectionDialog()
        if dialog.exec_() == QDialog.Accepted:
            selected_chamber = dialog.get_selected_chamber()
            if selected_chamber == "large":
                self.main_window = TSEI_PSRGVSystem()
            elif selected_chamber == "small":
                self.main_window = TSEI_PSRGVSystem_small()
            else:
                QMessageBox.critical(None, "Error", "Invalid chamber selection, exiting application.")
                self.app.quit()
                return

            self.main_window.show()
        else:
            QMessageBox.information(None, "Info", "Chamber selection canceled, exiting application.")
            self.app.quit()
            return

    def run(self):
        if self.login_window.exec_() == QDialog.Accepted:
            sys.exit(self.app.exec_())
        else:
            QMessageBox.information(None, "Info", "Login canceled, exiting application.")
            sys.exit()

def main():
    app = MainApp()
    app.run()

if __name__ == "__main__":
    main()