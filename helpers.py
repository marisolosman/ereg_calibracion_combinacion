#!/usr/bin/env python

import locale
import smtplib

from contextlib import contextmanager
from email.message import EmailMessage


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

    
def close_progress_bar():
    sys.stdout.write('\n')
  

def send_email(from_addr, password, to_addrs: list, subject, message):
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = ','.join(to_addrs)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_addr, password)
    server.sendmail(msg)
    server.quit()
