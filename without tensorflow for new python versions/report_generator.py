from fpdf import FPDF
from datetime import datetime
import webbrowser

def clean_text(text):
    if not text:
        return ""
    replacements = {
        "—": "-", "–": "-", "’": "'", "“": '"', "”": '"', "…": "..."
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def generate_report(p_eeg, p_nlp, risk_score, category, text_input):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    text_clean = clean_text(text_input)
    category_clean = clean_text(category)

    # HTML Report (Most Reliable)
    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Risk Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    h1 {{ color: #2c3e50; }}
    .section {{ padding: 15px; margin: 15px 0; border-left: 6px solid #3498db; background: #f8f9fa; }}
    .high {{ color: #e74c3c; font-weight: bold; }}
</style>
</head>
<body>
    <h1>🧠 Mental Health Risk Assessment Report</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <div class="section"><h2>EEG Module</h2><p>Depression Probability: <strong>{p_eeg:.4f}</strong></p></div>
    <div class="section"><h2>NLP Module</h2><p>Suicide Risk Probability: <strong>{p_nlp:.4f}</strong></p></div>
    <div class="section"><h2>Fusion Result</h2>
        <p><strong>Final Risk Score:</strong> {risk_score:.4f}</p>
        <p><strong>Category:</strong> <span class="high">{category}</span></p>
    </div>
    <div class="section"><h2>Text Analyzed</h2><p>{text_clean}</p></div>
    
    <h3>Recommendation</h3>
    <p><strong>{category}</strong> — Please consult a mental health professional immediately if this is high risk.</p>
    <p><small>This is a research prototype. Not a substitute for professional diagnosis.</small></p>
</body>
</html>"""

    html_path = f"risk_report_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"📄 HTML Report saved: {html_path}")

    # Simplified PDF Report
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, 'Mental Health Risk Assessment Report', ln=True, align='C')
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    content = [
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "EEG Module",
        f"Depression Probability: {p_eeg:.4f}",
        "",
        "NLP Module",
        f"Suicide Risk Probability: {p_nlp:.4f}",
        "",
        "Fusion Result",
        f"Final Risk Score: {risk_score:.4f}",
        f"Clinical Category: {category_clean}",
        "",
        "Text Analyzed:",
        text_clean[:800],  # Limit length to avoid overflow
        "",
        "Recommendation:",
        f"{category_clean} - Seek professional help if high risk detected.",
        "This is an AI prototype only."
    ]

    for line in content:
        pdf.multi_cell(0, 7, line)
        if line.startswith("Text Analyzed") or line.startswith("Recommendation"):
            pdf.ln(3)

    pdf_path = f"risk_report_{timestamp}.pdf"
    pdf.output(pdf_path)
    print(f"📕 PDF Report saved: {pdf_path}")

    # Auto open HTML report
    try:
        webbrowser.open(html_path)
    except:
        pass

    return f"✅ Reports generated!\n• HTML: {html_path}\n• PDF: {pdf_path}"