#!/usr/bin/env python

import locale
import sys
import smtplib

from contextlib import contextmanager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@contextmanager
def localized(code):
    old_code, old_encoding = locale.getlocale()
    locale.setlocale(locale.LC_ALL, code)
    yield
    locale.setlocale(locale.LC_ALL, f"{old_code}.{old_encoding}")


def progress_bar(count, total, status=''):
    bar_len = 100
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def progress_bar_clear_line():
    sys.stdout.write("\033[K") 
    
def progress_bar_close():
    sys.stdout.write('\n')
  

def send_email(from_addr, password, to_addrs: list, subject, body):
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = from_addr
    message['To'] = ','.join(to_addrs)

    body_content = body
    message.attach(MIMEText(body_content, "html"))
    msg_body = message.as_string()

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(message['From'], password)
    server.sendmail(message['From'], message['To'], msg_body)
    server.quit()
