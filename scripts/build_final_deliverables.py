"""Export the final report and native editable slides from bundled evidence.

No fitting, statistical recomputation, raw lyrics, model downloads or network.
The PPTX and PDF share vector geometry and explicit text wrapping. The deck PDF
is rendered directly from that geometry, not by a PowerPoint/Keynote engine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import fitz
import markdown
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/final"
W, H = 960, 540
NAVY, TEAL, RUST = "14273E", "087F8C", "C7633E"
GRAY, PALE, BG, WHITE = "586C80", "E5EDF4", "F7F9FC", "FFFFFF"


def read_facts(root=ROOT):
    """Read only bundled derived artifacts; return display values and provenance."""
    paths = ["docs/results/paired_inference.json", "docs/results/closeout_verification.json",
             "docs/figures/music_recession/results_summary.json",
             "docs/results/exploratory_primary_cov70/monthly_coverage.csv"]
    paired, verified, summary = [json.loads((root / p).read_text()) for p in paths[:3]]
    with (root / paths[3]).open(newline="", encoding="utf-8") as handle:
        coverage = list(csv.DictReader(handle))
    decades = sorted({int(r["predictor_month"][:4]) // 10 * 10 for r in coverage})
    medians = [statistics.median(float(r["scored_weight_coverage"]) for r in coverage
                                if int(r["predictor_month"][:4]) // 10 * 10 == d) for d in decades]
    for a, b in zip(paired["runs"], summary["runs"]):
        assert a["forecasts"] == b["paired_forecasts"]
        for model in ("a", "b"):
            assert abs(a["model_" + model]["brier_score"] - b["model_" + model + "_brier"]) < 1e-14
    return dict(paired=paired, verified=verified, summary=summary, decades=decades,
                median_coverage=medians,
                source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths})


def register_fonts(regular=None, bold=None):
    candidates = [("/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                  ("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                   "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf")]
    if regular and bold:
        candidates.insert(0, (str(regular), str(bold)))
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("Deck", regular))
            pdfmetrics.registerFont(TTFont("Deck-Bold", bold))
            pdfmetrics.registerFontFamily("Deck", normal="Deck", bold="Deck-Bold",
                                          italic="Deck", boldItalic="Deck-Bold")
            return dict(regular=Path(regular).name, bold=Path(bold).name,
                        regular_sha256=hashlib.sha256(Path(regular).read_bytes()).hexdigest(),
                        bold_sha256=hashlib.sha256(Path(bold).read_bytes()).hexdigest())
    raise RuntimeError("Arial TTF fonts required; supply --font-regular and --font-bold")


def wrap(text, width, size, bold=False):
    font = "Deck-Bold" if bold else "Deck"
    result = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if pdfmetrics.stringWidth(candidate, font, size) <= width:
                line = candidate
            else:
                if line:
                    result.append(line)
                if pdfmetrics.stringWidth(word, font, size) > width:
                    raise ValueError("Unbreakable text wider than box: " + word)
                line = word
        result.append(line)
    return result


class Surface:
    """Draw the same text, rules and native shapes into a PPTX and a PDF."""

    def __init__(self, presentation, canvas):
        self.slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        self.canvas = canvas
        self.boxes = []
        self.rect(0, 0, W, H, BG)

    def rect(self, x, y, w, h, fill=WHITE, stroke=None):
        shape = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x / 72), Inches(y / 72),
                                            Inches(w / 72), Inches(h / 72))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        if stroke:
            shape.line.color.rgb = RGBColor.from_string(stroke)
        else:
            shape.line.fill.background()
        self.canvas.setFillColor(colors.HexColor("#" + fill))
        self.canvas.setStrokeColor(colors.HexColor("#" + (stroke or fill)))
        self.canvas.rect(x, H - y - h, w, h, fill=1, stroke=int(bool(stroke)))

    def line(self, x1, y1, x2, y2, color=GRAY, width=1):
        shape = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                                Inches(x1 / 72), Inches(y1 / 72),
                                                Inches(x2 / 72), Inches(y2 / 72))
        shape.line.color.rgb = RGBColor.from_string(color)
        shape.line.width = Pt(width)
        self.canvas.setStrokeColor(colors.HexColor("#" + color))
        self.canvas.setLineWidth(width)
        self.canvas.line(x1, H - y1, x2, H - y2)

    def text(self, x, y, w, h, text, size=22, color=NAVY, bold=False):
        font = pdfmetrics.getFont("Deck-Bold" if bold else "Deck")
        missing = {c for c in str(text) if not c.isspace() and ord(c) not in font.face.charToGlyph}
        if missing:
            raise ValueError("Unsupported font glyphs: " + repr(missing))
        lines = wrap(str(text), w, size, bold)
        leading = size * 1.2
        if (len(lines) - 1) * leading + size * 1.1 > h + .1:
            raise ValueError("Text overflows %.0fx%.0f box (%d lines at %.1f): %s" %
                             (w, h, len(lines), size, text))
        if not (0 <= x <= W and 0 <= y <= H and x + w <= W + .1 and y + h <= H + .1):
            raise ValueError("Out-of-page text box: " + str(text))
        box = self.slide.shapes.add_textbox(Inches(x / 72), Inches(y / 72), Inches(w / 72), Inches(h / 72))
        tf = box.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name, p.font.size, p.font.bold = "Arial", Pt(size), bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.line_spacing = Pt(leading)
            p.space_before = p.space_after = Pt(0)
        self.canvas.setFillColor(colors.HexColor("#" + color))
        self.canvas.setFont("Deck-Bold" if bold else "Deck", size)
        for i, line in enumerate(lines):
            self.canvas.drawString(x, H - y - size * .91 - i * leading, line)
        self.boxes.append(dict(text=str(text),x=x,y=y,w=w,h=h,font=size,lines=len(lines)))

    def card(self, x, y, w, h, title, body, size=20):
        self.rect(x, y, w, h)
        self.rect(x, y, 4, h, TEAL)
        self.text(x + 16, y + 12, w - 32, 34, title, size=22, bold=True)
        self.text(x + 16, y + 52, w - 32, h - 58, body, size=size, color=GRAY)


def table(s, rows, widths, x=40, y=190, row_h=48, size=20):
    for i, row in enumerate(rows):
        s.rect(x, y + i * row_h, sum(widths), row_h, NAVY if i == 0 else (WHITE if i % 2 else PALE))
        left = x
        for cell, width in zip(row, widths):
            s.text(left + 10, y + i * row_h + 10, width - 18, row_h - 12, str(cell),
                   size=16 if i == 0 else size, color=WHITE if i == 0 else NAVY, bold=i == 0)
            left += width


def render_slide(s, spec, facts, index):
    body, layout = spec["body"], spec["layout"]
    runs = facts["paired"]["runs"]
    primary = runs[0]
    if layout != "opening":
        s.text(40, 19, 880, 18, "NYU PREDICTIVE ANALYTICS  /  TEAM 5", size=11, color=TEAL, bold=True)
        s.text(40, 54, 880, 78, spec["title"], size=32, bold=True)
        s.text(40, 135, 880, 44, spec["subtitle"], size=18, color=GRAY)
    s.line(40, 506, 920, 506, PALE)
    footer = "MAIN %s / 11" % spec["id"] if spec["seconds"] else "TECHNICAL APPENDIX  /  " + spec["id"]
    s.text(40, 517, 660, 15, "Exploratory retrospective study  ·  December 3, 2026 (planning date)", size=10, color=GRAY)
    s.text(735, 517, 185, 15, footer, size=10, color=GRAY)
    if layout == "opening":
        s.rect(0, 0, 15, H, TEAL)
        s.text(40, 31, 880, 20, "NYU PREDICTIVE ANALYTICS  /  FINAL PROJECT", size=12, color=TEAL, bold=True)
        s.text(40, 82, 880, 125, spec["title"], size=42, bold=True)
        s.text(40, 217, 880, 52, body[0], size=24)
        for x, value, label in [(40,"183","paired forecasts"),(340,"+0.0299","mean Brier B−A"),(640,"3","recession episodes")]:
            s.rect(x, 300, 280, 100)
            s.text(x+16, 312, 248, 48, value, size=36, color=RUST if x==340 else TEAL, bold=True)
            s.text(x+16, 365, 248, 28, label, size=18, color=GRAY)
        s.text(40, 423, 880, 55, body[1], size=21, bold=True)
        # Team names appear in the notes and title metadata as well as on slide.
        s.text(40, 273, 880, 20, "Keefer Wu · Nathan Starkman · Kiran Reddy Bhumireddy", size=16, color=GRAY)
    elif layout == "data":
        for x,value,label in [(40,"21,542","Billboard identities"),(340,"19,096","OSF candidates"),(640,"16,290","screened / scored songs")]:
            s.rect(x, 190, 280, 90)
            s.text(x+16, 202, 248, 44, value, size=34, color=TEAL, bold=True)
            s.text(x+16, 250, 248, 26, label, size=19)
        for i, entry in enumerate(body[:3]):
            title, detail = entry.split("|",1)
            s.text(40, 305+i*46, 255, 30, title, size=20, bold=True)
            s.text(305, 305+i*46, 615, 44, detail, size=19, color=GRAY)
        s.text(40, 455, 880, 36, body[3], size=18, color=RUST)
    elif layout == "coverage":
        x0,y0,bw,gap,chart_h=65,428,66,30,220
        for y,val in [(y0,0),(y0-chart_h*.7,.7),(y0-chart_h,1)]:
            s.line(55,y,650,y,PALE if val != .7 else RUST)
            s.text(18,y-9,35,25,"%g" % val,size=15,color=GRAY)
        for i,(d,v) in enumerate(zip(facts["decades"],facts["median_coverage"])):
            x=x0+i*(bw+gap)
            s.rect(x,y0-v*chart_h,bw,v*chart_h,TEAL)
            s.text(x-2,y0-v*chart_h-30,bw+16,27,"%.3f"%v,size=20,bold=True)
            label=("2020–23" if d==2020 else str(d)+"s")
            s.text(x-2,y0+10,bw+24,27,label,size=17,color=GRAY)
        s.card(700,195,220,252,"Coverage ≠ truth","Unverified identities and shifting songs remain even when the coverage gate passes.",size=20)
        s.text(40,466,880,31,"Eligible feature months at 70% / 75% / 80%: 429 / 348 / 291",size=20,bold=True)
    elif layout == "pipeline":
        coords=[(40,190),(340,190),(640,190),(640,350),(340,350),(40,350)]
        for (x,y),entry in zip(coords,body):
            title,detail=entry.split("|",1)
            s.card(x,y,280,132,title,detail,size=19)
        for x in (322,622):s.text(x,235,18,40,"→",size=14,color=TEAL)
        s.text(770,324,35,26,"↓",size=20,color=TEAL)
        for x in (322,622):s.text(x,393,18,40,"←",size=14,color=TEAL)
    elif layout == "models":
        s.rect(40,190,880,125)
        s.text(57,204,846,44,body[0],size=28,bold=True)
        s.text(57,260,846,45,body[1],size=28,bold=True)
        s.text(40,339,880,36,body[2],size=25,color=TEAL,bold=True)
        s.text(40,390,880,34,body[3],size=23)
        s.text(40,442,880,53,body[4],size=20,color=GRAY)
    elif layout == "timeline":
        s.text(40,188,880,52,body[0],size=22)
        s.text(40,251,880,37,body[1],size=23,color=TEAL,bold=True)
        labels=[("t−1","Latest training\ntarget month"),("t","Predictor\nobservation month"),("t+1 · day 15","Assumed\nissuance"),("t+6","Single-month\nstatus target")]
        s.line(95,322,860,322,TEAL,2)
        for i,(label,caption) in enumerate(labels):
            x=40+i*225
            s.rect(x+92,316,10,12,TEAL)
            s.text(x,287,220,31,label,size=23,bold=True)
            s.text(x,339,218,60,caption,size=20,color=GRAY)
        s.text(40,415,880,50,body[2],size=21,bold=True)
        s.text(40,470,880,27,body[3],size=18,color=GRAY)
    elif layout == "benchmarks":
        values=[facts["verified"]["runs"]["exploratory_primary_cov70"]["training_rate_brier"],
                facts["verified"]["primary_checks"]["full_history_a_same_dates"]["brier_score"],
                primary["model_a"]["brier_score"],primary["model_b"]["brier_score"]]
        labels=["Training rate (paired rows)","Full-history spread A","Paired spread A","Music B (same paired rows)"]
        for i,(value,label) in enumerate(zip(values,labels)):
            y=198+i*53
            s.text(40,y+3,300,31,label,size=20,bold=i==3)
            s.rect(355,y,460*value/.14,30,RUST if i==3 else TEAL)
            s.text(835,y+1,85,31,"%.4f"%value,size=22,bold=True)
        s.text(40,410,300,28,"Brier loss (lower is better)",size=17,color=GRAY)
        for v in (0,.05,.10):
            x=355+460*v/.14
            s.text(x,410,80,28,"%.2f"%v,size=17,color=GRAY)
        s.text(40,450,880,50,body[2],size=22,bold=True)
    elif layout == "samples":
        rows=[["Coverage","n","Pos.","Episodes","B−A Brier","A AP","B AP"]]
        for i,r in enumerate(runs):
            rows.append(["70% primary" if i==0 else ("75%" if i==1 else "80%"),r["forecasts"],r["recession_target_months"],len(r["recession_episodes_reached"]),"%+.4f"%r["loss_differential"]["mean"],"%.4f"%r["model_a"]["average_precision"],"%.4f"%r["model_b"]["average_precision"]])
        table(s,rows,[175,65,70,95,145,165,165],row_h=49,size=22)
        s.text(40,408,880,53,body[0],size=21,bold=True)
        s.text(40,467,880,28,body[1],size=18,color=GRAY)
    elif layout == "uncertainty":
        xmin,xmax=-.01,.07
        def axis(v):return 285+(v-xmin)/(xmax-xmin)*605
        s.text(40,187,880,32,body[0],size=20,bold=True)
        s.text(660,214,260,22,"Mean B−A: %+.4f" % primary["loss_differential"]["mean"],size=16,color=RUST,bold=True)
        s.line(axis(0),225,axis(0),390,GRAY,1)
        for i,b in enumerate(primary["loss_differential"]["block_bootstrap"]):
            y=245+i*57
            s.text(40,y-13,230,32,"%d-month blocks"%b["block_length_months"],size=22)
            l,u=axis(b["ci_lower"]),axis(b["ci_upper"])
            s.line(l,y,u,y,TEAL,3)
            s.line(l,y-7,l,y+7,TEAL,2);s.line(u,y-7,u,y+7,TEAL,2)
            s.rect(axis(primary["loss_differential"]["mean"])-5,y-5,10,10,RUST)
        s.line(285,392,890,392,PALE)
        for v in (0,.02,.04,.06):s.text(axis(v)-15,401,65,27,"%.2f"%v,size=17,color=GRAY)
        s.text(40,443,880,48,body[1],size=20,bold=True)
    elif layout == "limits":
        for i,entry in enumerate(body):
            title,detail=entry.split("|",1)
            s.card(40+(i%2)*450,190+(i//2)*150,430,137,title,detail,size=19)
    elif layout == "conclusion":
        s.text(40,195,880,80,body[0],size=29,bold=True)
        s.rect(40,290,880,65,NAVY)
        s.text(56,306,848,42,body[1],size=27,color=WHITE,bold=True)
        s.text(40,383,880,55,body[2],size=22)
        s.text(40,448,880,49,body[3],size=20,color=TEAL)
    elif layout == "sensitivity":
        rows=[["Coverage","6-month CI","12-month CI","24-month CI"]]
        for label,r in zip(("70%","75%","80%"),runs):
            rows.append([label]+["[%+.4f, %+.4f]"%(b["ci_lower"],b["ci_upper"]) for b in r["loss_differential"]["block_bootstrap"]])
        table(s,rows,[110,257,257,256],y=185,row_h=38,size=17)
        rows=[["Coverage","HAC p, lag 5","HAC p, lag 12","HAC p, lag 24"]]
        for label,r in zip(("70%","75%","80%"),runs):
            rows.append([label]+["%.6f"%v["p_value_two_sided"] for v in r["loss_differential"]["diebold_mariano"]])
        table(s,rows,[110,257,257,256],y=352,row_h=32,size=17)
        s.text(40,485,880,20,body[1],size=14,color=GRAY)
    elif layout == "diagnostics":
        rows=[["Coverage","A Brier","B Brier","A ROC-AUC","B ROC-AUC"]]
        for label,r in zip(("70%","75%","80%"),runs):
            rows.append([label]+["%.4f"%r[m][metric] for metric in ("brier_score","roc_auc") for m in ("model_a","model_b")])
        table(s,rows,[150,170,170,195,195],y=185,row_h=38,size=20)
        for i,line in enumerate(body):s.text(40,354+i*47,880,44,line,size=18,bold=i==0)
    elif layout in ("reproduction","crisp"):
        entries=body[:3] if layout=="reproduction" else body[:4]
        for i,entry in enumerate(entries):
            title,detail=entry.split("|",1)
            y=190+i*(85 if layout=="reproduction" else 63)
            s.text(40,y,245,33,title,size=21,bold=True,color=TEAL)
            s.text(295,y,625,73 if layout=="reproduction" else 59,detail,size=20)
        s.text(40,459,880,44,body[-1],size=18,color=GRAY)
    elif layout == "references":
        for i,line in enumerate(body[:3]):s.text(40,190+i*63,880,58,line,size=21)
        s.rect(40,391,880,105,NAVY)
        s.text(56,407,848,82,body[3],size=23,color=WHITE,bold=True)
    else:
        raise ValueError("Unknown layout: " + layout)
    notes = ("Planned narration: %d seconds.\n\n" % spec["seconds"] if spec["seconds"] else "Appendix: no prepared narration.\n\n") + spec["notes"]
    notes += "\n\nEvidence:\n" + "\n".join("https://github.com/kirankigi5/music_as_recession_alternate_data/blob/docs/final-project-deliverables/"+p for p in spec["evidence"])
    s.slide.notes_slide.notes_text_frame.text = notes


def report_pdf(source, out):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Report",fontName="Deck",fontSize=10.4,leading=14.5,spaceAfter=8,textColor=colors.HexColor("#"+NAVY)))
    styles.add(ParagraphStyle(name="Cell",parent=styles["Report"],fontSize=8.4,leading=11,spaceAfter=0))
    for name,size,leading in (("Title",24,29),("Heading1",19,24),("Heading2",16,21),("Heading3",12,16)):
        styles[name].fontName="Deck-Bold";styles[name].fontSize=size;styles[name].leading=leading
        styles[name].textColor=colors.HexColor("#"+NAVY)
        styles[name].spaceAfter=10
    parsed=ET.fromstring("<root>"+markdown.markdown(source,extensions=["tables","fenced_code"])+"</root>")
    for font in ("Deck", "Deck-Bold"):
        mapping = pdfmetrics.getFont(font).face.charToGlyph
        missing = {c for c in "".join(parsed.itertext()) if not c.isspace() and ord(c) not in mapping}
        if missing:
            raise ValueError("Unsupported report glyphs: " + repr(missing))
    flow=[]
    base="https://github.com/kirankigi5/music_as_recession_alternate_data/blob/docs/final-project-deliverables/"
    def inline(el):
        el=ET.fromstring(ET.tostring(el,encoding="unicode"))
        for a in el.iter("a"):
            href=a.get("href","")
            if not href.startswith(("https:","http:","mailto:")):
                part,sep,anchor=href.partition("#")
                absolute=(SOURCE/part).resolve().relative_to(ROOT)
                a.set("href",base+absolute.as_posix()+(sep+anchor if sep else ""))
        for code in el.iter("code"):
            code.tag="font";code.set("face","Deck")
        text=ET.tostring(el,encoding="unicode")
        start=text.find(">")+1
        return text[start:text.rfind("</")]
    for el in parsed:
        if el.tag=="div" and el.get("class")=="page-break":flow.append(PageBreak())
        elif el.tag in ("h1","h2","h3"):
            flow.append(Paragraph(inline(el),styles[{"h1":"Title","h2":"Heading2","h3":"Heading3"}[el.tag]]))
        elif el.tag=="p":
            img=el.find("img")
            if img is not None:
                path=out/img.get("src")
                with fitz.open(path) as image_doc:
                    ratio=image_doc[0].rect.height/image_doc[0].rect.width
                flow.append(Image(str(path),width=510,height=510*ratio))
                flow.append(Paragraph(html.escape(img.get("alt","")),styles["Cell"]))
                flow.append(Spacer(1,7))
            else:flow.append(Paragraph(inline(el),styles["Report"]))
        elif el.tag in ("ul","ol"):
            items=[ListItem(Paragraph(inline(li),styles["Report"])) for li in el.findall("li")]
            flow.append(ListFlowable(items,bulletType="bullet" if el.tag=="ul" else "1",leftIndent=15,bulletFontName="Deck",bulletFontSize=8))
        elif el.tag=="pre":
            flow.append(Preformatted("".join(el.itertext()),ParagraphStyle(name="Code",fontName="Courier",fontSize=7.4,leading=10,spaceAfter=10)))
        elif el.tag=="table":
            rows=[]
            for row in el.iter("tr"):
                rows.append([Paragraph(inline(cell),styles["Cell"]) for cell in row])
            n=len(rows[0])
            if n==2:widths=[150,360]
            elif n==3:widths=[120,195,195]
            elif n==4:widths=[80,143.33,143.33,143.34]
            elif n==7:widths=[75,87,64,64,72,74,74]
            else:widths=[510/n]*n
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#"+PALE)),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F7F9")]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            flow.extend([t,Spacer(1,10)])
        else:raise ValueError("Unsupported report element: "+el.tag)
    def footer(canvas,doc):
        canvas.setFont("Deck",8);canvas.setFillColor(colors.HexColor("#"+GRAY))
        canvas.drawString(42,23,"NYU Team 5 · Final project · Exploratory retrospective research")
        canvas.drawRightString(552,23,str(doc.page))
    SimpleDocTemplate(str(out/"report.pdf"),pagesize=(594,842),leftMargin=42,rightMargin=42,
                      topMargin=36,bottomMargin=42,title="Can Popular Music Predict U.S. Recessions?",
                      author="Keefer Wu; Nathan Starkman; Kiran Reddy Bhumireddy").build(flow,onFirstPage=footer,onLaterPages=footer)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True,help="new output directory; existing paths are refused")
    parser.add_argument("--font-regular",type=Path)
    parser.add_argument("--font-bold",type=Path)
    args=parser.parse_args(argv)
    if args.out.exists():parser.error("--out must be a fresh path")
    fonts=register_fonts(args.font_regular,args.font_bold)
    specs=json.loads((SOURCE/"slides.json").read_text())
    facts=read_facts()
    assert sum(s["seconds"] for s in specs["slides"])==specs["planned_seconds"]==660
    args.out.mkdir(parents=True)
    prs=Presentation();prs.slide_width=Inches(W/72);prs.slide_height=Inches(H/72)
    prs.core_properties.title=specs["title"]
    prs.core_properties.author="Keefer Wu; Nathan Starkman; Kiran Reddy Bhumireddy"
    canvas=Canvas(str(args.out/"presentation.pdf"),pagesize=(W,H),invariant=1)
    canvas.setTitle(specs["title"]);canvas.setAuthor(prs.core_properties.author)
    layouts=[];notes=["# Speaker notes and timing\n", "Planned narration: **11:00** across 11 main slides. Supplied slot: **12:30**; Q&A inclusion remains unconfirmed. These are targets, not measured rehearsal times. Appendix slides have no scheduled narration.\n"]
    elapsed=0
    for index,spec in enumerate(specs["slides"]):
        surface=Surface(prs,canvas)
        render_slide(surface,spec,facts,index)
        layouts.append(dict(id=spec["id"],boxes=surface.boxes))
        canvas.showPage()
        notes.append("## %s · %s\n"%(spec["id"],spec["title"].replace("\n"," ")))
        if spec["seconds"]:
            end=elapsed+spec["seconds"]
            notes.append("Target **%d:%02d–%d:%02d** (%d seconds).\n"%(elapsed//60,elapsed%60,end//60,end%60,spec["seconds"]))
            elapsed=end
        else:notes.append("Appendix — use only if asked; no scheduled narration.\n")
        notes.append(spec["notes"]+"\n")
        notes.append("Evidence: "+", ".join("["+p+"]("+"../../"+p+")" for p in spec["evidence"])+".\n")
    canvas.save();prs.save(args.out/"presentation.pptx")
    (args.out/"speaker_notes.md").write_text("\n".join(notes))
    (args.out/"figures").mkdir()
    with fitz.open(args.out/"presentation.pdf") as doc:
        for page,rect,name in [(6,(25,185,930,440),"primary_benchmarks"),(8,(25,182,925,432),"primary_uncertainty")]:
            doc[page].get_pixmap(matrix=fitz.Matrix(2,2),clip=fitz.Rect(rect)).save(args.out/"figures"/(name+".png"))
    report_pdf((SOURCE/"report.md").read_text(),args.out)
    manifest=dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/"slides.json",SOURCE/"report.md",Path(__file__))},
                  evidence_sha256=facts["source_sha256"],fonts=fonts,
                  packages={p:version(p) for p in ("python-pptx","reportlab","pymupdf","markdown")},
                  deck_pdf_renderer="ReportLab shared vector geometry, not a PowerPoint roundtrip",
                  main_slides=11,appendix_slides=5,planned_narration_seconds=660,layouts=layouts)
    (args.out/"build_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print("Exported report PDF, native PPTX, deck PDF, notes and two report figures to",args.out)


if __name__=="__main__":
    main()
