import re
import os
import requests
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
import html
from typing import List, Dict, Optional

# 从环境变量读取机密配置
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MAILEROO_API_KEY = os.getenv('MAILEROO_API_KEY')
MAIL_TO = os.getenv('MAIL_TO')
MAIL_FROM = os.getenv('MAIL_FROM')


def fetch_atom_feed(url: str) -> str:
    """获取 Atom feed XML 文本"""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_atom_feed_yesterday(xml_content: str) -> List[Dict]:
    """
    解析 Atom feed，获取“前一天（UTC）”的所有 entry：
    返回列表，每个元素包含 title, link, updated, summary, summary_type
    """
    root = ET.fromstring(xml_content)
    NS = {'atom': 'http://www.w3.org/2005/Atom'}

    now_utc = datetime.now(timezone.utc)
    yesterday_utc = now_utc - timedelta(days=1)

    entries = []
    for e in root.findall('atom:entry', NS):
        title_elem = e.find('atom:title', NS)
        link_elem = e.find('atom:link', NS)
        updated_elem = e.find('atom:updated', NS) or e.find('atom:published', NS)
        summary_elem = e.find('atom:summary', NS)

        if not (title_elem is not None and link_elem is not None and updated_elem is not None and summary_elem is not None):
            continue

        title = title_elem.text or ''
        link = link_elem.get('href') or ''
        updated_str = updated_elem.text
        updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
        summary_type = summary_elem.get('type') or ''
        summary_content = summary_elem.text or ''
        
        summary_content = re.sub(
            r'\s*<p>\s*Tags:\s*.*?</p>',
            '',
            summary_content,
            flags=re.DOTALL
        ).strip()
        
        # 只保留前一天 UTC 的 entry
        if yesterday_utc.date() <= updated.date() < now_utc.date():
            entries.append({
                'title': title,
                'link': link,
                'updated': updated_str,
                'summary': summary_content,
                'summary_type': summary_type,
            })

    return entries


def groq_translate_html(summary_html: str, to_lang: str = 'zh') -> Optional[str]:
    """
    使用 Groq API 将 HTML 文本翻译成中文，保持 HTML 结构。
    如果失败则返回 None。
    """
    if not summary_html or not GROQ_API_KEY:
        return None

    try:
        # 使用 Groq OpenAI 兼容接口
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        # 这里示例用 mixtral-8x7b-32768，可视为 groq/compound 中的一个模型
        payload = {
            "model": "groq/compound-mini",
            "messages": [
                {
                    "role": "system",
                    "content": f"你是一个专业的翻译助手，请将下面的英文 HTML 内容准确翻译成{to_lang}，必须保持 HTML 标签结构不变。"
                },
                {
                    "role": "user",
                    "content": summary_html
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq] 翻译失败：{e}")
        return None


def build_email_html(entries: List[Dict], with_translation: bool) -> str:
    """
    根据 entries 构建完整 HTML 邮件。
    - with_translation=True 时：尝试翻译每条 summary；
    - False 时：只包含英文原文。
    """
    beijing_tz = timezone(timedelta(hours=8))
    now_bj = datetime.now(beijing_tz)
    subject_date = now_bj.strftime("%Y-%m-%d")
    subject = f"🍈 Simon atom - {subject_date}"

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html>")
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append(f"<title>{html.escape(subject)}</title>")
    parts.append("</head>")
    parts.append('<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background-color:#F9FAFB;">')

    # HEADER
    parts.append(
        '<div style="background:linear-gradient(135deg,#0F172A,#1E293B);padding:20px 16px;text-align:center;color:#FFFFFF;">'
        f'<h1 style="margin:0;font-size:30px;line-height:1.4;">Simon Willison\'s atom</h1>'
        f'<p style="margin:8px 0 0 0;font-size:14px;color:#CBD5E1;">Generated on {now_bj.strftime("%Y-%m-%d %H:%M")} UTC+8</p>'
        "</div>"
    )

    # 容器
    parts.append('<div style="max-width:600px;margin:0 auto;padding:16px;background-color:#FFFFFF;">')

    if not entries:
        parts.append(
            '<p style="font-size:14px;color:#64748B;">'
            "没有检测到前一天（UTC）的新内容。"
            "</p>"
        )
    else:
        for entry in entries:
            title = entry["title"]
            link = entry["link"]
            updated_str = entry["updated"]
            summary_html = entry["summary"]

            # 转换时间到北京
            try:
                updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                updated_bj = updated_dt.astimezone(beijing_tz)
                updated_display = updated_bj.strftime("%Y-%m-%d %H:%M %Z")
            except Exception:
                updated_display = updated_str

            title_safe = html.escape(title)
            link_safe = html.escape(link, quote=True)

            # 每条 entry 容器
            parts.append(
                '<div style="margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #E2E8F0;">'
                f'<h2 style="margin:0 0 6px 0;font-size:18px;color:#0F172A;line-height:1.4;">'
                f'<a href="{link_safe}" style="color:#1D4ED8;text-decoration:none;">{title_safe}</a>'
                "</h2>"
                f'<p style="margin:0 0 12px 0;font-size:12px;color:#64748B;">Published: {updated_display}</p>'
            )

            # 英文原文
            parts.append(
                '<div style="margin-top:10px;">'
                '<h3 style="margin:0 0 6px 0;font-size:14px;color:#0F172A;">English Original:</h3>'
                # summary 是原本的 HTML 片段，直接嵌入
                f'{summary_html}'
                "</div>"
            )

            if with_translation:
                zh_html = groq_translate_html(summary_html)
                if zh_html:
                    parts.append(
                        '<div style="margin-top:12px;">'
                        '<h3 style="margin:0 0 6px 0;font-size:14px;color:#0F172A;">中文翻译:</h3>'
                        f"{zh_html}"
                        "</div>"
                    )
                else:
                    parts.append(
                        '<div style="margin-top:12px;">'
                        '<p style="margin:0;font-size:12px;color:#9CA3AF;font-style:italic;">'
                        "翻译失败，已保留英文原文。"
                        "</p>"
                        "</div>"
                    )

            parts.append("</div>")  # 结束单条 entry

    parts.append("</div>")  # 结束主体容器

    # FOOTER
    parts.append(
        '<div style="background:linear-gradient(135deg,#0F172A,#1E293B);padding:16px;text-align:center;color:#FFFFFF;">'
        f'<p style="margin:0;font-size:12px;color:#CBD5E1;">'
        f'Source: simonwillison.net'
        "</p>"
        "</div>"
    )

    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def send_via_maileroo(to_list: List[str], subject: str, html_body: str) -> None:
    """通过 Maileroo 的 Email API 发送 HTML 邮件"""
    if not MAILEROO_API_KEY or not MAIL_FROM:
        print("Maileroo API key 或 MAIL_FROM 未配置，跳过发送。")
        return

    url = "https://smtp.maileroo.com/send"
    headers = {
        "X-API-Key": MAILEROO_API_KEY,
    }

    for addr in to_list:
        payload = {
            "to": addr,
            "from": MAIL_FROM,
            "subject": subject,
            "html": html_body,
        }
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=30)
            print(f"响应状态码: {resp.status_code}")
            print(f"响应内容: {resp.text}")

            resp.raise_for_status()
            print(f"邮件发送成功: {addr}")
        except requests.exceptions.HTTPError as e:
            print(f"邮件发送失败 {addr}: HTTP {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"邮件发送失败 {addr}: {e}")


def generate_github_pages_html(entries: List[Dict]) -> str:
    """
    生成 GitHub Pages 用的 index.html，只显示最近 10 条 title/link。
    布局简洁，头尾也带渐变 HEADER/FOOTER。
    """
    latest = entries[:10]
    beijing_tz = timezone(timedelta(hours=8))
    now_bj = datetime.now(beijing_tz)

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html>")
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append("<title>Simon Willison&#39;s Atom Feed - Latest 10 entries</title>")
    parts.append(
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
        "margin:0;padding:0;background:#0F172A;color:#0F172A;}"
        ".wrap{max-width:800px;margin:0 auto;padding:16px 16px 32px;background:#F9FAFB;}"
        ".header{background:linear-gradient(135deg,#0F172A,#1E293B);padding:20px 16px;text-align:center;color:#FFFFFF;margin-bottom:24px;}"
        ".header h1{margin:0;font-size:22px;}"
        ".header p{margin:8px 0 0 0;font-size:13px;color:#CBD5E1;}"
        ".entry{padding:12px 0;border-bottom:1px solid #E5E7EB;}"
        ".entry a{color:#1D4ED8;text-decoration:none;font-size:16px;}"
        ".entry a:hover{text-decoration:underline;}"
        ".date{font-size:12px;color:#6B7280;margin-top:4px;}"
        ".footer{text-align:center;font-size:12px;color:#9CA3AF;margin-top:24px;}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<div class="header">')
    parts.append("<h1>Simon Willison&#39;s Atom Feed</h1>")
    parts.append("<p>Latest 10 entries (Beijing Time)</p>")
    parts.append("</div>")
    parts.append('<div class="wrap">')

    if not latest:
        parts.append('<p style="font-size:14px;color:#64748B;">当前没有可显示的条目。</p>')
    else:
        for e in latest:
            title = html.escape(e["title"])
            link = html.escape(e["link"], quote=True)
            updated_str = e["updated"]
            try:
                updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                updated_bj = updated_dt.astimezone(beijing_tz)
                date_str = updated_bj.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = updated_str

            parts.append('<div class="entry">')
            parts.append(f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>')
            parts.append(f'<div class="date">{date_str}</div>')
            parts.append("</div>")

    parts.append(
        f'<div class="footer">Data updated at {now_bj.strftime("%Y-%m-%d %H:%M")} UTC+8</div>'
    )
    parts.append("</div>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def main():
    FEED_URL = "https://simonwillison.net/atom/everything"
    try:
        xml_text = fetch_atom_feed(FEED_URL)
    except Exception as e:
        print(f"获取 Atom feed 失败: {e}")
        return

    entries = parse_atom_feed_yesterday(xml_text)
    print(f"前一天（UTC）共匹配到 {len(entries)} 条 entry")

    # 始终生成 GitHub Pages index.html（即便 entries 为空也会生成一份简单页面）
    pages_html = generate_github_pages_html(entries)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(pages_html)
    print("index.html 已生成（用于 GitHub Pages）")

    # 如果没有前一天的内容，不发邮件（可按需修改为发送“空日报”）
    if not entries:
        print("无前一天内容，跳过邮件发送。")
        return

    # 准备多收件人
    if not MAIL_TO:
        print("MAIL_TO 未配置，无法发送邮件。")
        return
    to_list = [x.strip() for x in MAIL_TO.split(",") if x.strip()]
    if not to_list:
        print("MAIL_TO 中没有有效邮箱，跳过邮件发送。")
        return

    # 先尝试带翻译版本，如 Groq 整体出错，再降级为英文版
    beijing_tz = timezone(timedelta(hours=8))
    now_bj = datetime.now(beijing_tz)
    subject_date = now_bj.strftime("%Y-%m-%d")
    subject = f"🍈 Simon atom - {subject_date}"

    try:
        email_html = build_email_html(entries, with_translation=True)
        send_via_maileroo(to_list, subject, email_html)
    except Exception as e:
        print(f"生成或发送带翻译邮件失败，尝试发送纯英文版本。错误: {e}")
        email_html_fallback = build_email_html(entries, with_translation=False)
        send_via_maileroo(to_list, subject, email_html_fallback)


if __name__ == "__main__":
    main()
