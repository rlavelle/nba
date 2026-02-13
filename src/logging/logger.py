import configparser
import datetime
import os

from src.config import CONFIG_PATH
from src.logging.email_sender import EmailSender
from src.utils.date import date_to_dint


class Logger:
    def __init__(self, name: str | None = None, daily_cron: bool = False, admin: bool = False):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)
        self.admin = admin

        if daily_cron:
            today = date_to_dint(datetime.date.today())
            log_folder = self.config.get('LOG_PATH', 'logs_folder')
            self.filename = os.path.join(log_folder, f'log_{name}_{today}.txt' if name else f'log_{today}.txt')
        else:
            self.filename = self.config.get('LOG_PATH', 'path')

    def log(self, message: str, email=False):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} - {message}\n'

        with open(self.filename, 'a') as log_file:
            log_file.write(log_entry)

        if email:
            self.send_error(log_entry)

    def email_log(self):
        email_sender = EmailSender()
        email_sender.read_recipients_from_file()
        email_sender.set_subject('NBA Log')
        email_sender.set_body('full log')
        email_sender.add_attachment(self.filename)
        email_sender.send_email(admin=self.admin)

    def send_error(self, message: str):
        email_sender = EmailSender()
        email_sender.read_recipients_from_file()
        email_sender.set_subject('Error Log')
        email_sender.set_body(message)
        email_sender.send_email(admin=self.admin)


if __name__ == "__main__":
    logger = Logger()
    logger.log('test')
