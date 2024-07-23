from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

USER_ID = os.getenv('USER_ID')
USER_PASSWORD = os.getenv('USER_PASSWORD')
EXPIRY_DATE = os.getenv('EXPIRY_DATE')

# 사용자 정보 설정
valid_id = USER_ID
valid_pw = USER_PASSWORD
expiry_date = datetime.strptime(EXPIRY_DATE, '%Y-%m-%d')

class LoginWindow(QDialog):
    def __init__(self, on_login_success):
        super().__init__()
        self.setWindowTitle("Login")
        self.on_login_success = on_login_success

        layout = QVBoxLayout()

        layout.addWidget(QLabel("ID"))
        self.id_entry = QLineEdit()
        layout.addWidget(self.id_entry)

        layout.addWidget(QLabel("Password"))
        self.pw_entry = QLineEdit()
        self.pw_entry.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.pw_entry)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.check_login)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

    def check_login(self):
        user_id = self.id_entry.text()
        user_pw = self.pw_entry.text()

        if user_id == valid_id and user_pw == valid_pw:
            if datetime.now() > expiry_date:
                QMessageBox.critical(self, "Error", "프로그램 사용 유효기간이 만료되었습니다.")
            else:
                self.accept()
                self.on_login_success()
        else:
            QMessageBox.critical(self, "Error", "Invalid ID or Password")
