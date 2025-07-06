import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart  # 需要改为 MIMEMultipart
from email.mime.text import MIMEText

import cv2

email_sender = ''  # 填入发送邮箱地址，可以是你自己的
email_receiver = ['']  # 接收邮箱，同上

smtp_host = 'smtp.xx.com'  # 设置为你的发送邮箱smtp地址
smtp_password = ''  # smtp验证码
smtp_port = 465  # 端口，自行查阅


def send_email(ret, image, text, pconf):
    """
    发送邮件
    :param ret: 检测结果
    :param image: 图片数据
    """
    if not ret:
        print('没有检测到车辆')
        return False

    try:
        # 创建邮件对象 - 必须使用 MIMEMultipart 才能同时包含文本和图片
        msg = MIMEMultipart()
        msg['Subject'] = '发现车道堵塞'
        msg['From'] = email_sender
        msg['To'] = ', '.join(email_receiver)  # 多个收件人需要用逗号分隔

        # HTML正文
        body = f'''
        <html>
            <body>
                <h1>车牌：{text}</h1>
                <h2>置信度：{pconf:.2f}</h2>
                <h3>检测到车道堵塞，请及时处理。</h3>
                <p>以下是检测到的图片：</p>
                <img src="cid:image1" alt="检测到的图片" style="max-width: 100%; height: auto;">
                <p>感谢您对此产品信赖和使用</p>
            </body>
        </html>
        '''
        msg.attach(MIMEText(body, 'html', 'utf-8'))

        # 添加图片
        _, img_encoded = cv2.imencode('.jpeg', image)
        img_bytes = img_encoded.tobytes()

        image_part = MIMEImage(img_bytes)
        image_part.add_header('Content-ID', '<image1>')
        image_part.add_header('Content-Disposition', 'attachment', filename='detection.jpeg')
        msg.attach(image_part)

        # 发送邮件
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(email_sender, smtp_password)
            server.sendmail(email_sender, email_receiver, msg.as_string())
            print('邮件发送成功')
            return True

    except Exception as e:
        print(f'邮件发送失败: {e}')
        return False
