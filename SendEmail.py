import smtplib
from email.mime.text import MIMEText

class MailSender:
    def __init__(self, gmail_email="aureliano.firouzi@gmail.com", gmail_password="syae ijyb anfi leaf"):
        self.email = gmail_email
        self.password = gmail_password
    
    def send_confirmation(self, patient_info, receiver_email):
        """Skicka bekräftelse-email till patient"""
        
        subject = "Bokningsbekräftelse - Vårdcentral"
        
        body = f"""Hej {patient_info['namn']}!

Din bokning är registrerad:

Namn: {patient_info['namn']}
Adress: {patient_info['adress']}
Ålder: {patient_info['ålder']}
Besvär: {patient_info['besvär']}

Vi återkommer med bokad tid.

Mvh Vårdcentralen"""

        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = self.email
            msg['To'] = receiver_email
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            print(f"✅ Email skickat till {receiver_email}")
            return True
            
        except Exception as e:
            print(f"❌ Email-fel: {e}")
            return False