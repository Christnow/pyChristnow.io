import smtplib
import poplib
import imaplib
from email.mime.text import MIMEText
from email.header import Header


class operate_email:
    def __init__(self, your_email_address, your_email_passwd):
        self.address = your_email_address
        self.passwd = your_email_passwd

    def send_email_by_smtp(self, receive_email_address, message_subject,
                           message_context):
        smtp_server_host = "smtp.exmail.qq.com"
        semtp_server_post = "465"
        message = MIMEText(message_context, 'plain', 'utf-8')
        message['From'] = Header(self.address, 'utf-8')
        message['To'] = Header(receive_email_address, 'utf-8')
        message['Subject'] = Header(message_subject, 'utf-8')
        email_client = smtplib.SMTP_SSL(smtp_server_host, semtp_server_post)
        try:
            email_client.login(self.address, self.password)
            print("smtp----login sucess,now will send email to {}".format(
                reveice_email_address))
        except Exception as e:
            print(e)
            try:
                print('smtp----login again.....')
                email_client.login(self.address, self.password)
                print("smtp----login sucess,now will send email to {}".format(
                    reveice_email_address))
            except Exception as e:
                print(e)
                print(
                    'smtp----sorry,check yourusername or yourpassword not correct or another problem occur'
                )
        else:
            email_client.sendmail(self.address, receive_email_address,
                                  message.as_string())
            print('smtp----send email to {} finish'.format(
                receive_email_address))
        finally:
            email_client.close()

    def recv_email_by_imap(self):
        imap_server_host = "imap.exmail.qq.com"
        imap_server_port = 993
        try:
            email_server = imaplib.IMAP4_SSL(host=imap_server_host,
                                             port=imap_server_port)
            print("imap----connect server success, now will check username")
        except:
            print(
                "imap----sorry the given email server address connect time out"
            )
            exit(1)
        try:
            # 验证邮箱及密码是否正确
            email_server.login(self.address, self.passwd)
            print("imap----username exist, now will check password")
        except:
            print(
                "imap----sorry the given email address or password seem do not correct"
            )
            exit(1)

        # 邮箱中其收到的邮件的数量
        email_server.select()
        email_count = len(email_server.search(None, 'ALL')[1][0].split())
        # 通过fetch(index)读取第index封邮件的内容；这里读取最后一封，也即最新收到的那一封邮件
        typ, email_content = email_server.fetch(
            f'{email_count}'.encode('utf-8'), '(RFC822)')
        # 将邮件内存由byte转成str
        email_content = email_content[0][1].decode('utf-8')
        # 关闭select
        email_server.close()
        # 关闭连接
        email_server.logout()
        return email_content

    # 此函数通过使用poplib实现接收邮件
    def recv_email_by_pop3(self):
        # 要进行邮件接收的邮箱。改成自己的邮箱
        email_address = "your_email@qq.com"
        # 要进行邮件接收的邮箱的密码。改成自己的邮箱的密码
        email_password = "your_email_password"
        # 邮箱对应的pop服务器，也可以直接是IP地址
        # 改成自己邮箱的pop服务器；qq邮箱不需要修改此值
        pop_server_host = "pop.qq.com"
        # 邮箱对应的pop服务器的监听端口。改成自己邮箱的pop服务器的端口；qq邮箱不需要修改此值
        pop_server_port = 995

        try:
            # 连接pop服务器。如果没有使用SSL，将POP3_SSL()改成POP3()即可其他都不需要做改动
            email_server = poplib.POP3_SSL(host=pop_server_host,
                                           port=pop_server_port,
                                           timeout=10)
            print("pop3----connect server success, now will check username")
        except:
            print(
                "pop3----sorry the given email server address connect time out"
            )
            exit(1)
        try:
            # 验证邮箱是否存在
            email_server.user(email_address)
            print("pop3----username exist, now will check password")
        except:
            print("pop3----sorry the given email address seem do not exist")
            exit(1)
        try:
            # 验证邮箱密码是否正确
            email_server.pass_(email_password)
            print("pop3----password correct,now will list email")
        except:
            print("pop3----sorry the given username seem do not correct")
            exit(1)

        # 邮箱中其收到的邮件的数量
        email_count = len(email_server.list()[1])
        # 通过retr(index)读取第index封邮件的内容；这里读取最后一封，也即最新收到的那一封邮件
        resp, lines, octets = email_server.retr(email_count)
        # lines是邮件内容，列表形式使用join拼成一个byte变量
        email_content = b'\r\n'.join(lines)
        # 再将邮件内容由byte转成str类型
        email_content = email_content.decode()
        print(email_content)

        # 关闭连接
        email_server.close()


if __name__ == '__main__':

    my_email_address = 'your_email@qq.com'
    my_email_passwd = 'your_email_password'
    client = operate_email(my_email_address, my_email_passwd)
    email_content = client.recv_email_by_imap()
    with open('recv_email.txt', 'w', encoding='utf-8') as f:
        f.write(email_content)
