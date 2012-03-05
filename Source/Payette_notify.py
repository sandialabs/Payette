#!/usr/bin/env python2
import os
import sys
import smtplib
if sys.version_info < (3,0):
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEBase import MIMEBase
    from email.MIMEText import MIMEText
    from email import Encoders
else:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

gmail_user = "csm.uofu@gmail.com"
gmail_pwd = "thisiscsm"

mailing_list = ["scot.swan@gmail.com"]

def notify(subject, text):
    try:
        for recipient in mailing_list:
            msg = MIMEMultipart()

            msg['From'] = gmail_user
            msg['To'] = recipient

            #  uncomment this next line if you want a subject
            #  when sending an email (leave commented for texts)
            msg['Subject'] = subject
 
            msg.attach(MIMEText(text))
            mailServer = smtplib.SMTP("smtp.gmail.com", 587)
            mailServer.ehlo()
            mailServer.starttls()
            mailServer.ehlo()
            mailServer.login(gmail_user, gmail_pwd)
            mailServer.sendmail(gmail_user, recipient, msg.as_string())
            # Should be mailServer.quit(), but that crashes...
            mailServer.close()
    except:
        print("Payette_notify.py: Unable to send messages.")
