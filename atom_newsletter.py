import re
import os
import time
import requests
import json
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
import html
from typing import List, Dict, Optional

# 从环境变量读取机密配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
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


# 不可重试的状态码
_NO_RETRY = {400, 401, 403, 404}


def deepseek_translate_html(summary_html: str, to_lang: str = "zh") -> Optional[str]:
    """
    使用 DeepSeek 翻译 HTML 内容为指定语言。
    - 长文本自动分段翻译，避免截断
    - 失败自动重试 3 次（指数退避）
    """
    if not summary_html or not DEEPSEEK_API_KEY:
        return None

    # ── 短文本：直接翻译 ──
    if len(summary_html) <= 6000:
        return _call_deepseek(summary_html, to_lang)

    # ── 长文本：分段翻译 ──
    chunks = _split_html(summary_html, max_chars=3000)
    print(f"[DeepSeek] 📄 长文本拆分为 {len(chunks)} 段")

    results: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[DeepSeek] 翻译第 {i}/{len(chunks)} 段…")
        result = _call_deepseek(chunk, to_lang)
        if result is None:
            return None
        results.append(result)

    return "\n".join(results)


# ─────────────────────────────────────────
#  核心 API 调用（含重试）
# ─────────────────────────────────────────
def _call_deepseek(
    html: str,
    to_lang: str,
    max_retries: int = 3,
    max_tokens: int = 8000,
) -> Optional[str]:

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    f"你是专业翻译助手。将以下英文 HTML 翻译成{to_lang}，"
                    "只翻译文本，保持所有 HTML 标签和属性原样不动。"
                    "不要添加任何解释或 markdown 代码块标记。"
                ),
            },
            {"role": "user", "content": html},
        ],
        "temperature": 1.3,
        "max_tokens": max_tokens,     # ← 关键
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            # 不可重试的错误，立即退出
            if resp.status_code in _NO_RETRY:
                print(f"[DeepSeek] ❌ HTTP {resp.status_code}，不可重试：{resp.text[:300]}")
                return None

            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]

            # ── 检测截断 ──
            if choice.get("finish_reason") == "length":
                print("[DeepSeek] ⚠️ 输出被截断（finish_reason=length），"
                      "建议减小分段大小或增大 max_tokens")

            return choice["message"]["content"].strip()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            tag = {
                requests.exceptions.Timeout: "⏰ 超时",
                requests.exceptions.ConnectionError: "🔌 连接错误",
            }.get(type(e), "⚠️ HTTP 错误")
            print(f"[DeepSeek] {tag}（{attempt}/{max_retries}）：{e}")

        except Exception as e:
            print(f"[DeepSeek] ❌ 未知错误：{type(e).__name__}: {e}")
            return None                    # 未知异常不重试

        if attempt < max_retries:
            wait = min(5 * 2 ** (attempt - 1), 180)
            print(f"[DeepSeek] ⏳ {wait}s 后重试…")
            time.sleep(wait)

    print("[DeepSeek] 🚫 重试耗尽，翻译放弃")
    return None


# ─────────────────────────────────────────
#  HTML 智能分段
# ─────────────────────────────────────────
_BLOCK_RE = re.compile(
    r"(<(?:p|div|h[1-6]|ul|ol|li|tr|blockquote|section|article|figure)"
    r"[\s>].*?</(?:p|div|h[1-6]|ul|ol|li|tr|blockquote|section|article|figure)>)",
    re.DOTALL | re.IGNORECASE,
)

def _split_html(html: str, max_chars: int = 3000) -> List[str]:
    """
    按块级 HTML 标签边界拆分，尽量让每段 ≤ max_chars。
    保证不会从标签中间切断。
    """
    parts = _BLOCK_RE.split(html)           # 交替：间隔文本 / 匹配标签
    chunks: list[str] = []
    buf = ""

    for part in parts:
        if not part:
            continue
        # 单个 part 就超长 → 单独成段（总比截断好）
        if len(part) > max_chars and not buf:
            chunks.append(part)
            continue
        if len(buf) + len(part) > max_chars and buf:
            chunks.append(buf)
            buf = part
        else:
            buf += part

    if buf:
        chunks.append(buf)

    return chunks or [html]


def _clamp_images(html_content: str) -> str:
    """给 summary 里的每个 <img> 注入 max-width:100% 的 inline style，
       同时移除可能存在的 width/height 属性，防止撑破布局。"""
    # 移除 <img> 上的 width="..." / height="..." 属性
    html_content = re.sub(
        r'(<img\b[^>]*?)\s+width\s*=\s*["\']?\d+["\']?',
        r'\1',
        html_content,
        flags=re.IGNORECASE,
    )
    html_content = re.sub(
        r'(<img\b[^>]*?)\s+height\s*=\s*["\']?\d+["\']?',
        r'\1',
        html_content,
        flags=re.IGNORECASE,
    )

    # 如果 <img> 已有 style="..."，在里面追加；否则插入新的 style
    def _inject_style(m: re.Match) -> str:
        tag = m.group(0)
        inject = "max-width:100%!important;height:auto!important;display:block;"
        if re.search(r'style\s*=\s*["\']', tag, re.IGNORECASE):
            # 已有 style，追加到末尾
            tag = re.sub(
                r'(style\s*=\s*["\'])',
                rf'\1{inject}',
                tag,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            # 没有 style，插入一个
            tag = tag.replace("<img", f'<img style="{inject}"', 1)
        return tag

    html_content = re.sub(r'<img\b[^>]*?/?>', _inject_style, html_content, flags=re.IGNORECASE)
    return html_content


def build_email_html(entries: List[Dict], with_translation: bool) -> str:
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

    # ────────── 新增：全局样式，约束图片/表格宽度 ──────────
    parts.append("""<style>
      img {
        max-width: 100% !important;
        height: auto !important;
        display: block;
      }
      table {
        max-width: 100% !important;
        width: auto !important;
      }
      pre, code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
      }
    </style>""")
    # ─────────────────────────────────────────────────────

    parts.append("</head>")
    parts.append(
        '<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;'
        'background-color:#F9FAFB;'
        # ▼ 防止出现横向滚动条
        'width:100%!important;-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%;">'
    )

    # HEADER
    parts.append(
        '<div style="background:linear-gradient(135deg,#0F172A,#1E293B);padding:20px 16px;text-align:center;color:#FFFFFF;">'
        f'<h1 style="margin:0;font-size:30px;line-height:1.4;">Simon Willison\'s atom</h1>'
        f'<p style="margin:8px 0 0 0;font-size:14px;color:#CBD5E1;">Updated at {now_bj.strftime("%Y-%m-%d %H:%M")} UTC+8</p>'
        "</div>"
    )

    # ▼ 外层容器加 overflow:hidden，兜底防溢出
    parts.append(
        '<div style="max-width:600px;margin:0 auto;padding:16px;'
        'background-color:#FFFFFF;overflow:hidden;word-wrap:break-word;">'
    )

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

            # ────────── 关键：处理 summary 中的图片 ──────────
            summary_html = _clamp_images(summary_html)

            try:
                updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                updated_bj = updated_dt.astimezone(beijing_tz)
                updated_display = updated_bj.strftime("%Y-%m-%d %H:%M %Z")
            except Exception:
                updated_display = updated_str

            title_safe = html.escape(title)
            link_safe = html.escape(link, quote=True)

            parts.append(
                '<div style="margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #E2E8F0;">'
                f'<h2 style="margin:0 0 6px 0;font-size:18px;color:#0F172A;line-height:1.4;">'
                f'<a href="{link_safe}" style="color:#1D4ED8;text-decoration:none;">{title_safe}</a>'
                "</h2>"
                f'<p style="margin:0 0 12px 0;font-size:12px;color:#64748B;">🕖Published: {updated_display}</p>'
            )

            # ▼ 内容区也加 overflow:hidden
            parts.append(
                '<div style="margin-top:10px;font-size:14px;line-height:1.6;color:#111827;overflow:hidden;">'
                '<h3 style="margin:0 0 6px 0;font-size:14px;color:#0F172A;">📄English:</h3>'
                f'{summary_html}'
                "</div>"
            )

            if with_translation:
                zh_html = deepseek_translate_html(summary_html)
                if zh_html:
                    zh_html = _clamp_images(zh_html)
                    parts.append(
                        '<div style="margin-top:12px;font-size:14px;line-height:1.6;color:#374151;overflow:hidden;">'
                        '<h3 style="margin:0 0 6px 0;font-size:14px;color:#0F172A;">🤖中文翻译:</h3>'
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

            parts.append("</div>")

    parts.append("</div>")

    # FOOTER
    parts.append(
        '<div style="background:linear-gradient(135deg,#0F172A,#1E293B);padding:16px;text-align:center;color:#FFFFFF;">'
        '<p style="margin:0;font-size:12px;color:#CBD5E1;">'
        'Source: simonwillison.net'
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

    url = "https://smtp.maileroo.com/api/v2/emails"
    headers = {
        "X-API-Key": MAILEROO_API_KEY,
        "Content-Type": "application/json", 
    }

    for addr in to_list:
        payload = {
            "from": {
                "address": MAIL_FROM,
                "display_name": "Newsletter"
            },
            "to": [
                {"address": addr}
            ],
            "subject": subject,
            "html": html_body,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"响应状态码: {resp.status_code}")
            print(f"响应内容: {resp.text}")

            resp.raise_for_status()
            print(f"邮件发送成功: {addr}")
        except requests.exceptions.HTTPError as e:
            print(f"邮件发送失败 {addr}: HTTP {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"邮件发送失败 {addr}: {e}")


ENTRIES_JSON = "entries.json"


def load_saved_entries() -> List[Dict]:
    """从 entries.json 加载已保存的历史条目。"""
    if not os.path.exists(ENTRIES_JSON):
        return []
    try:
        with open(ENTRIES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 {ENTRIES_JSON} 失败：{e}")
        return []


def save_entries(entries: List[Dict]) -> None:
    """将条目列表写入 entries.json 持久化。"""
    with open(ENTRIES_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def merge_entries(
    new_entries: List[Dict],
    saved_entries: List[Dict],
    max_count: int = 10,
) -> List[Dict]:
    """
    合并新旧条目 → 按 link 去重 → 按 updated 降序 → 取前 max_count 条。
    """
    seen: set = set()
    merged: List[Dict] = []

    # 新条目优先（同 link 保留新版本）
    for e in new_entries + saved_entries:
        key = e.get("link", e.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            merged.append(e)

    # 按 updated 时间降序排序
    def sort_key(entry: Dict) -> datetime:
        try:
            return datetime.fromisoformat(
                entry["updated"].replace("Z", "+00:00")
            )
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    merged.sort(key=sort_key, reverse=True)
    return merged[:max_count]


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
    parts.append("<p>Latest 10 entries</p>")
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
        f'<div class="footer">Data updated at {now_bj.strftime("%Y-%m-%d %H:%M")} +08:00</div>'
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

    saved_entries = load_saved_entries()
    all_entries = merge_entries(entries, saved_entries, max_count=10)
    save_entries(all_entries)
    print(f"合并后共 {len(all_entries)} 条条目（最多保留 10 条）")

    pages_html = generate_github_pages_html(all_entries)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(pages_html)
    print("index.html 已生成（用于 GitHub Pages）")

    if not entries:
        print("无前一天内容，跳过邮件发送。")
        return

    if not MAIL_TO:
        print("MAIL_TO 未配置，无法发送邮件。")
        return
    to_list = [x.strip() for x in MAIL_TO.split(",") if x.strip()]
    if not to_list:
        print("MAIL_TO 中没有有效邮箱，跳过邮件发送。")
        return

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
