from pathlib import Path
import shutil

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(r"C:\Users\robin\Desktop\alpha momentum")
OUT = ROOT / "output" / "pdf" / "Robin_Taki_Resume_Universal.pdf"


def draw_para(c, text, style, x, y, width):
    p = Paragraph(text, style)
    w, h = p.wrap(width, 1000)
    p.drawOn(c, x, y - h)
    return y - h


def draw_heading(c, text, x, y, width, color):
    c.setStrokeColor(color)
    c.setLineWidth(0.8)
    c.line(x, y, x + width, y)
    y -= 8
    return y


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    page_w, page_h = A4
    margin_x = 15 * mm
    right = page_w - margin_x
    width = right - margin_x
    y = page_h - 16 * mm

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "Name",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=25.5,
        leading=27.5,
        textColor=HexColor("#0f1720"),
        spaceAfter=0,
    )
    title_style = ParagraphStyle(
        "Title",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11.7,
        leading=13.8,
        textColor=HexColor("#123f44"),
    )
    contact_style = ParagraphStyle(
        "Contact",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=11.4,
        textColor=HexColor("#5f6668"),
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.4,
        leading=12.9,
        textColor=HexColor("#0e6b63"),
        spaceAfter=0,
        uppercase=True,
        tracking=0.6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.0,
        leading=12.2,
        textColor=HexColor("#151515"),
        spaceAfter=0,
    )
    body_bold_style = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=9.0,
        leading=10.8,
    )

    # Header
    y = draw_para(c, "Robin Taki", name_style, margin_x, y, width)
    y -= 2
    y = draw_para(c, "M.Sc. Engineering Student in Engineering Mathematics | KTH Royal Institute of Technology", title_style, margin_x, y, width)
    y -= 1
    y = draw_para(c, "LinkedIn | Portfolio | Streakora | Stockholm, Sweden | Open to relocate", contact_style, margin_x, y, width)
    y -= 6
    c.setStrokeColor(HexColor("#c9d3cf"))
    c.setLineWidth(0.8)
    c.line(margin_x, y, right, y)
    y -= 10

    # Summary
    y = draw_para(c, "<b>SUMMARY</b>", section_style, margin_x, y, width)
    y -= 2
    summary = (
        "Engineering mathematics student at KTH building research and software projects, including a public quant portfolio and a live consumer product. "
        "Experience shipping a live product, working in customer-facing operations, and solving graph and matching problems. "
        "Interested in quantitative analysis, software engineering, and data-driven finance."
    )
    y = draw_para(c, summary, body_style, margin_x, y, width)
    y -= 8

    # Selected Projects
    y = draw_para(c, "<b>SELECTED PROJECTS</b>", section_style, margin_x, y, width)
    y -= 3
    projects = [
        "<b>Systematic Nordic Equity Research &amp; Validation</b> - Public research portfolio on a Nordic momentum strategy with walk forward validation, untouched holdout testing, and forward monitoring.",
        "<b>Streakora</b> - Founder and builder of a live productivity product for routines, habits, and focus.",
    ]
    for proj in projects:
        y = draw_para(c, proj, body_style, margin_x + 4 * mm, y, width - 4 * mm)
        y -= 4
    y -= 2

    # Experience
    y = draw_para(c, "<b>EXPERIENCE</b>", section_style, margin_x, y, width)
    y -= 3
    experiences = [
        (
            "<b>Founder &amp; Builder, Streakora</b> | Jan 2026 - Present",
            [
                "Shipped a live web product end to end.",
                "Owned product direction, implementation, and public release.",
            ],
        ),
        (
            "<b>Idrottsplatsarbetare, Stockholms stad</b> | Aug 2024 - Present",
            [
                "Support and supervise sports facilities, customer service, and seasonal maintenance.",
                "Work across football pitches, sports halls, and ice rinks.",
            ],
        ),
        (
            "<b>Idrottsplatsarbetare, Stockholms stad</b> | Oct 2023 - Aug 2024",
            [
                "Maintained football pitches, sports halls, and ice rinks in Stockholm West.",
            ],
        ),
        (
            "<b>Football Coach, Skattkarrs IF</b> | May 2022 - Oct 2022",
            [
                "Led a summer football school for ages 9-15.",
            ],
        ),
        (
            "<b>Mathematics Teaching Intern, Tingvalla gymnasiet</b> | Oct 2018 - 1 month",
            [
                "Completed classroom teaching practice in mathematics.",
            ],
        ),
    ]
    for title, bullets in experiences:
        y = draw_para(c, title, body_bold_style, margin_x, y, width)
        y -= 1
        for bullet in bullets:
            y = draw_para(c, f"- {bullet}", body_style, margin_x + 4 * mm, y, width - 4 * mm)
            y -= 1
        y -= 3

    # Education
    y = draw_para(c, "<b>EDUCATION</b>", section_style, margin_x, y, width)
    y -= 3
    y = draw_para(c, "<b>KTH Royal Institute of Technology</b> | M.Sc. Engineering, Engineering Mathematics | Aug 2024 - Jun 2029", body_style, margin_x + 4 * mm, y, width - 4 * mm)
    y -= 8

    # Skills
    y = draw_para(c, "<b>TECHNICAL SKILLS</b>", section_style, margin_x, y, width)
    y -= 3
    skill_lines = [
        "<b>Technical</b> - Python, MATLAB, quantitative analysis, statistics, backtesting",
        "<b>Have worked with</b> - SQL, HTML, CSS",
        "<b>Tools</b> - Git, Vercel, Supabase, Resend, PostHog",
        "<b>Focus</b> - Solo project delivery, product shipping, and analytical problem solving",
    ]
    for line in skill_lines:
        y = draw_para(c, line, body_style, margin_x, y, width)
        y -= 1
    y -= 6

    # Algorithms / Competitive Programming
    y = draw_para(c, "<b>ALGORITHMS / COMPETITIVE PROGRAMMING</b>", section_style, margin_x, y, width)
    y -= 3
    algo = (
        "Solved MAPS 2020 Problem L (\"The Wrath of Kahn\") using Kahn's algorithm, "
        "reachability analysis, and a maximum bipartite matching reduction via Dilworth's theorem."
    )
    y = draw_para(c, f"- {algo}", body_style, margin_x + 4 * mm, y, width - 4 * mm)

    c.showPage()
    c.save()

    # Mirror into the existing filenames used in the conversation.
    for target_name in [
        "Robin_Taki_Resume.pdf",
        "Robin_Taki_Resume_Wintermute.pdf",
        "Robin_Taki_Resume_Wintermute_Industry.pdf",
    ]:
        shutil.copyfile(OUT, OUT.with_name(target_name))


if __name__ == "__main__":
    main()
