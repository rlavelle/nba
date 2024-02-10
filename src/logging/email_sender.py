import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import configparser
from src.config import CONFIG_PATH


class EmailSender:
    def __init__(self):
        self.smtp_server = ""
        self.smtp_port = 0
        self.sender_email = ""
        self.sender_password = ""
        self.recipients_file_path = ""
        self.recipients = []
        self.subject = ""
        self.body = ""
        self.attachments = []

        self.load_config(CONFIG_PATH)

    def load_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        self.smtp_server = config.get('EMAIL', 'SMTP_SERVER')
        self.smtp_port = config.getint('EMAIL', 'SMTP_PORT')
        self.sender_email = config.get('EMAIL', 'SENDER_EMAIL')
        self.sender_password = config.get('EMAIL', 'SENDER_PASSWORD')
        self.recipients_file_path = config.get('EMAIL', 'RECIPIENTS_FILE_PATH')

    def read_recipients_from_file(self):
        with open(self.recipients_file_path, 'r') as file:
            self.recipients = [line.strip() for line in file]

    def set_subject(self, subject):
        self.subject = subject

    def set_body(self, body):
        self.body = body

    def add_attachment(self, file_path):
        self.attachments.append(file_path)

    def send_email(self):
        if not self.recipients:
            raise ValueError("No recipients. Please add recipients using read_recipients_from_file or add_recipient.")

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.recipients)
        msg['Subject'] = self.subject

        msg.attach(MIMEText(self.body, 'plain'))

        for attachment in self.attachments:
            with open(attachment, 'rb') as file:
                part = MIMEApplication(file.read(), Name=attachment.split('/')[-1])
                part['Content-Disposition'] = f'attachment; filename="{attachment.split("/")[-1]}"'
                msg.attach(part)

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.recipients, msg.as_string())


if __name__ == "__main__":
    email_sender = EmailSender()
    email_sender.read_recipients_from_file()
    email_sender.set_subject('Test Subject')
    email_sender.set_body('This is a test email.')
    email_sender.add_attachment('/Users/rowanlavelle/Documents/Projects/nba/logs/log.txt')

    email_sender.send_email()
