#!/usr/bin/env python3
"""Build the print-ready participant materials pack (A4, 4 pages)."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    HRFlowable, KeepTogether,
)

OUT = "/mnt/user-data/outputs/participant_materials.pdf"

# --- Study constants: edit these two lines, rebuild, done -------------------
INSTITUTION = "Universität des Saarlandes"
RETENTION = "31 December 2026"
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=14.5, leading=18, spaceAfter=2, spaceBefore=0,
    textColor=colors.HexColor("#111111"),
)
SUB = ParagraphStyle(
    "SUB", parent=styles["Normal"], fontName="Helvetica-Oblique",
    fontSize=9, leading=12, textColor=colors.HexColor("#555555"),
    spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=10, leading=13, spaceBefore=8.5, spaceAfter=3,
    textColor=colors.HexColor("#1a1a1a"),
)
BODY = ParagraphStyle(
    "BODY", parent=styles["Normal"], fontName="Helvetica",
    fontSize=9, leading=12.4, spaceAfter=4.5, alignment=TA_LEFT,
)
SMALL = ParagraphStyle(
    "SMALL", parent=BODY, fontSize=8.2, leading=11,
    textColor=colors.HexColor("#444444"),
)
ITEM = ParagraphStyle(
    "ITEM", parent=BODY, fontSize=9, leading=12, spaceAfter=0,
)
NOTE = ParagraphStyle(
    "NOTE", parent=BODY, fontSize=8.5, leading=12,
    textColor=colors.HexColor("#7a3b00"),
    backColor=colors.HexColor("#fdf6ec"),
    borderPadding=6, spaceBefore=6, spaceAfter=8,
)


def rule():
    return HRFlowable(width="100%", thickness=0.7,
                      color=colors.HexColor("#cccccc"),
                      spaceBefore=3, spaceAfter=6)


def bullets(items, style=BODY):
    return [Paragraph("&bull;&nbsp;&nbsp;" + t, style) for t in items]


# ============================ PAGE 1: CONSENT ==============================
def page_consent():
    s = []
    s.append(Paragraph("Participant Information &amp; Consent", H1))
    s.append(Paragraph(
        f"Seminar research project &middot; {INSTITUTION}", SUB))
    s.append(rule())

    s.append(Paragraph("What is this study about?", H2))
    s.append(Paragraph(
        "This project studies how people work with an AI assistant on a creative "
        "writing task. You will develop a short story with an AI system and then "
        "answer some questions about how the session felt. There "
        "are no right or wrong answers, and nothing about your performance is being "
        "assessed &mdash; the system is what is under study, not you.", BODY))
    s.append(Paragraph(
        "So that your behaviour stays natural, one aspect of what the system does is "
        "not described in advance. It will be explained to you in full at the end of "
        "the session, and you may withdraw your data at that point if you wish.", BODY))

    s.append(Paragraph("What will you be asked to do?", H2))
    for p in bullets([
        "A story-building conversation with the AI, in English (12&ndash;15 min).",
        "A short written questionnaire, then a brief informal conversation "
        "about your experience (about 10 min).",
    ]):
        s.append(p)

    s.append(Paragraph("What data is recorded?", H2))
    for p in bullets([
        "The text of your conversation with the AI, and the timing of your "
        "interactions with the interface.",
        "Your questionnaire answers and my written notes from our conversation.",
        "<b>No audio or video recording.</b> No screen recording.",
        "<b>No name, e-mail, student number, or any other identifying "
        "information.</b> You are recorded only as a participant code such as P01.",
    ]):
        s.append(p)
    s.append(Paragraph(
        "Please avoid typing personal information about yourself or other people "
        "into the conversation. If you do, tell me and I will delete it.", BODY))

    s.append(Paragraph("How is the data used and stored?", H2))
    for p in bullets([
        "Used only for a seminar report and presentation at " + INSTITUTION + ".",
        "It is held in password-protected, university-provided storage.",
        "AI replies come from a third-party language-model API (Google Gemini), "
        "so your conversation text is sent to that service to generate them.",
        "Anonymised excerpts of the conversation may be quoted in the report.",
        f"All raw data is deleted by {RETENTION}.",
    ]):
        s.append(p)

    s.append(Paragraph("Your rights", H2))
    s.append(Paragraph(
        "Participation is entirely voluntary. You may stop at any moment, without "
        "giving a reason and without any disadvantage. You may ask me to delete your "
        "session at any time up to submission &mdash; just tell me your participant "
        "code.", BODY))
    s.append(Paragraph(
        "If you have a question at any point &mdash; before, during, or after "
        "&mdash; just ask the researcher sitting with you.", BODY))

    s.append(rule())
    s.append(Paragraph("Declaration of consent", H2))
    s.append(Paragraph(
        "I have read and understood the information above. I have had the chance to "
        "ask questions. I take part voluntarily and I know I can stop at any time.",
        BODY))
    s.append(Spacer(1, 6))

    sig = Table(
        [["Participant code", "P _______"],
         ["Signature", ""],
         ["Date", ""]],
        colWidths=[38 * mm, 118 * mm], rowHeights=[8 * mm, 11 * mm, 8 * mm],
    )
    sig.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("FONT", (1, 0), (1, -1), "Helvetica", 9),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LINEBELOW", (1, 0), (1, -1), 0.6, colors.HexColor("#666666")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    s.append(sig)
    return s


# ============================ PAGE 2: TASK =================================
def page_task():
    s = []
    s.append(Paragraph("Your Task", H1))
    s.append(Paragraph("Hand this to the participant at the start of the "
                       "conversation.", SUB))
    s.append(rule())

    s.append(Paragraph("The brief", H2))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "<b>Build a short story around this opening: a person comes home and finds "
        "a door in their flat that was not there yesterday.</b>",
        ParagraphStyle(
            "brief", parent=BODY, fontSize=11.5, leading=16,
            backColor=colors.HexColor("#f2f5fa"), borderPadding=9,
            spaceBefore=4, spaceAfter=14)))
    s.append(Paragraph(
        "Work with the AI to develop it. You do not have to write finished prose "
        "&mdash; talking through who the person is, what is behind the door, and "
        "what happens next is exactly the point. By the end, aim to have a story "
        "you could summarise to someone in a minute.", BODY))

    s.append(Paragraph("How to work", H2))
    for p in bullets([
        "Treat the AI as a writing partner, not a text generator. Push back, ask "
        "it to develop things, reject what does not fit your story.",
        "Follow whatever direction genuinely interests you. There is no intended "
        "story and nothing you can get wrong.",
        "Write however is comfortable &mdash; full sentences or fragments. "
        "Spelling and grammar do not matter.",
        "You have about 12&ndash;15 minutes. I will tell you when time is nearly up.",
        "If the interface offers you anything during the session, respond however "
        "feels natural. There is no expected response.",
    ]):
        s.append(p)

    s.append(Paragraph("A few practical things", H2))
    for p in bullets([
        "Responses take a few seconds to arrive &mdash; that is normal.",
        "Please do not put personal details about yourself or anyone you know "
        "into the story.",
        "If anything breaks or confuses you, just say so out loud.",
        "You can stop at any point for any reason.",
    ]):
        s.append(p)

    s.append(Spacer(1, 14))
    s.append(Paragraph("Notes / scratch space", H2))
    box = Table([[""]], colWidths=[162 * mm], rowHeights=[60 * mm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#bbbbbb")),
    ]))
    s.append(box)
    return s


# ======================= PAGE 3: QUESTIONNAIRE =============================
CONSTRUCTS = [
    ("A. Creativity support", [
        ("A1", "The system helped me come up with ideas I would not have "
               "thought of on my own."),
        ("A2", "I was able to explore many different ideas during the session."),
        ("A3", "The system supported my thinking rather than getting in the way."),
        ("A4", "What I ended up with was worth the effort I put in."),
    ]),
    ("B. Getting stuck", [
        ("B1", "At some point I felt stuck on one narrow idea."),
        ("B2", "I noticed myself repeating the same idea in different words."),
        ("B3", "When I felt stuck, something in the session helped me move on."),
    ]),
    ("C. Exploration", [
        ("C1", "I seriously considered directions quite different from my "
               "first idea."),
        ("C2", "The session opened up directions I had not been thinking about."),
        ("C3", "I stayed close to my very first idea for most of the session."),
    ]),
    ("D. Timing and satisfaction", [
        ("D1", "Anything the system offered me arrived at a good moment."),
        ("D2", "The system interrupted me when I did not want to be "
               "interrupted."),
        ("D3", "I would use a tool like this for my own creative work."),
        ("D4", "Overall, I was satisfied with the session."),
    ]),
]


def page_questionnaire():
    s = []
    s.append(Paragraph("Post-Session Questionnaire", H1))
    s.append(Paragraph("Participant code: P _______ &nbsp;&nbsp;&nbsp; "
                       "Date: ____________", SUB))
    s.append(Paragraph(
        "For each statement, circle the number that best matches your experience. "
        "<b>1 = strongly disagree, 7 = strongly agree.</b> Answer for how the "
        "session actually felt, not how you think it should have felt. If a "
        "statement does not apply, leave it blank.", BODY))
    s.append(rule())

    scale_hdr = ["", "1", "2", "3", "4", "5", "6", "7"]

    for title, items in CONSTRUCTS:
        rows = [[Paragraph(f"<b>{title}</b>", ITEM)] + scale_hdr[1:]]
        for code, text in items:
            rows.append([Paragraph(f"<b>{code}</b>&nbsp; {text}", ITEM),
                         "1", "2", "3", "4", "5", "6", "7"])
        t = Table(rows, colWidths=[102 * mm] + [8.5 * mm] * 7)
        t.setStyle(TableStyle([
            ("FONT", (1, 0), (-1, -1), "Helvetica", 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (1, 0), (-1, 0), colors.HexColor("#888888")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dddddd")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ("LEFTPADDING", (0, 0), (0, -1), 6),
        ]))
        s.append(t)
        s.append(Spacer(1, 5))

    s.append(Paragraph("In your own words", H2))
    s.append(Paragraph(
        "Was there a moment in the session where your thinking changed direction? "
        "What caused it?", ITEM))
    box = Table([[""]], colWidths=[162 * mm], rowHeights=[42 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6,
                             colors.HexColor("#bbbbbb"))]))
    s.append(Spacer(1, 4))
    s.append(box)
    return s


# ==================== PAGE 4: RESEARCHER SHEET =============================
def page_researcher():
    s = []
    s.append(Paragraph("Researcher Sheet &mdash; Do Not Show Participant", H1))
    s.append(Paragraph("Observation log, interview guide, scoring key, and "
                       "debrief script.", SUB))
    s.append(rule())

    s.append(Paragraph("Live observation log", H2))
    s.append(Paragraph(
        "Fill this in <i>during</i> the session. The turn numbers are what let you "
        "line these notes up against the fixation trajectory afterwards.", SMALL))
    hdr = ["Turn #", "Clock", "Intervention fired?",
           "Branch taken / ignored", "Visible reaction"]
    rows = [hdr] + [[""] * 5 for _ in range(8)]
    t = Table(rows, colWidths=[15 * mm, 15 * mm, 26 * mm, 38 * mm, 68 * mm],
              rowHeights=[6.5 * mm] + [8.5 * mm] * 8)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    s.append(t)
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "Session ID: ____________________ &nbsp; Strategy: ____________ &nbsp; "
        "Start: ______ &nbsp; End: ______",
        SMALL))

    s.append(Paragraph("Semi-structured interview (~5 min, after questionnaire)",
                       H2))
    for p in bullets([
        "Talk me through what you were thinking in the first couple of minutes.",
        "Was there a point where you felt you were going round in circles? What "
        "did that feel like?",
        "The system offered you some alternative directions at one point. What "
        "went through your head when that appeared?",
        "Did any of those suggestions change what you did next &mdash; or did you "
        "carry on with your own idea?",
        "Was the timing right? Too early, too late, too often?",
        "If this were a real tool you used for coursework, what would you change?",
    ], SMALL):
        s.append(p)
    s.append(Paragraph(
        "Write answers verbatim where you can. One vivid quote is worth more in "
        "an N=1&ndash;2 write-up than any average.", SMALL))

    s.append(Paragraph("Scoring key", H2))
    s.append(Paragraph(
        "<b>Reverse-code before analysis: C3, D2.</b> (score = 8 &minus; raw). "
        "B1 and B2 are <i>not</i> reverse-coded &mdash; they measure whether "
        "fixation occurred at all, which is a manipulation check, not an outcome. "
        "High B1/B2 with high B3 is the pattern your hypothesis predicts: the "
        "participant got stuck <i>and</i> then got unstuck. High B1/B2 with low B3 "
        "is a negative result and is worth reporting honestly.", SMALL))
    s.append(Paragraph(
        "Constructs map to the proposal as: A = perceived creativity support, "
        "B = fixation reduction, C = exploration, D = satisfaction. Item wording "
        "for A and C is adapted from the Creativity Support Index; cite it as "
        "adapted, not as administered.", SMALL))

    s.append(Paragraph("Debrief script (read after the interview)", H2))
    s.append(Paragraph(
        "&ldquo;Now I can tell you the part I left out. The system was continuously "
        "measuring how much your conversation was circling around the same idea "
        "&mdash; it compares the meaning of each message to the ones before it. When "
        "that measure crossed a threshold, it decided you might be stuck and offered "
        "those alternative directions. I did not tell you in advance because knowing "
        "would have changed how you responded to them. Everything you did was "
        "completely normal and there was no correct behaviour. Are you still happy "
        "for me to use your session? You can say no and I will delete it right now.&rdquo;",
        SMALL))
    s.append(Paragraph(
        "Record the answer here: &nbsp; consent confirmed after debrief &nbsp; "
        "[ &nbsp; ] yes &nbsp;&nbsp;&nbsp; [ &nbsp; ] no &mdash; delete now", SMALL))
    return s


def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm,
        topMargin=17 * mm, bottomMargin=15 * mm,
        title="Participant Materials — Conversational Fixation Study",
    )
    story = []
    for i, fn in enumerate([page_consent, page_task, page_questionnaire,
                            page_researcher]):
        story.extend(fn())
        if i < 3:
            story.append(PageBreak())
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
