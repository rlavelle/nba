import datetime
import configparser
from src.config import CONFIG_PATH
from src.logging.email_sender import EmailSender


class Logger:
    def __init__(self, fpath='path'):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)

        self.filename = self.config.get('LOG_PATH', fpath)

    def log(self, message, email=False):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} - {message}\n'

        with open(self.filename, 'a') as log_file:
            log_file.write(log_entry)

        if email:
            self.send_error()

    def email_log(self):
        email_sender = EmailSender()
        email_sender.read_recipients_from_file()
        email_sender.set_subject('Error Log')
        email_sender.set_body('full error log')
        email_sender.add_attachment(self.filename)
        email_sender.send_email()

    @staticmethod
    def send_error(message):
        email_sender = EmailSender()
        email_sender.read_recipients_from_file()
        email_sender.set_subject('Error Log')
        email_sender.set_body(message)
        email_sender.send_email()


if __name__ == "__main__":
    logger = Logger()
    logger.log('test')

