"""Create realistic sample policy PDFs for testing"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from pathlib import Path

# Create sample_policies directory
Path('sample_policies').mkdir(exist_ok=True)

print("Creating sample policy PDFs...\n")

# ============================================================================
# Policy 2025 — Earlier Version
# ============================================================================
doc = SimpleDocTemplate('sample_policies/Academic_Regulations_2025.pdf', pagesize=letter)
story = []

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=16,
    textColor='black',
    spaceAfter=6,
)
heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=12,
    textColor='black',
    spaceAfter=6,
)
body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    spaceAfter=12,
)
meta_style = ParagraphStyle(
    'Meta',
    parent=styles['Normal'],
    fontSize=9,
    textColor='#666666',
    spaceAfter=20,
)

story.append(Paragraph('ACADEMIC REGULATIONS 2025', title_style))
story.append(Paragraph('Institution of Excellence • Year 2025', meta_style))

sections_2025 = [
    ('1. ATTENDANCE POLICY',
     'Students must maintain a minimum attendance of 75% in all theory classes and 85% in practical/laboratory sessions to be eligible to appear in the end-semester examination.'),
    ('2. MINIMUM PASSING SCORE',
     'A student must secure a minimum of 40% marks in the end-semester examination to pass the subject.'),
    ('3. LIBRARY BORROWING LIMIT',
     'Students are permitted to borrow a maximum of 3 books at a time from the library for a loan period of 14 days.'),
    ('4. BACK-PAPER EXAMINATION CONDUCT',
     'Students appearing for back-paper examinations are permitted to attend regular classes during the examination period.'),
    ('5. PLACEMENT ELIGIBILITY & TRAINING',
     'The placement cell will conduct a minimum of 2 pre-placement training sessions per semester for final-year students.'),
]

for title, content in sections_2025:
    story.append(Paragraph(title, heading_style))
    story.append(Paragraph(content, body_style))

doc.build(story)
print("✓ Created Academic_Regulations_2025.pdf")

# ============================================================================
# Policy 2026 — Newer Version (with substantive changes)
# ============================================================================
doc = SimpleDocTemplate('sample_policies/Academic_Regulations_2026.pdf', pagesize=letter)
story = []

story.append(Paragraph('ACADEMIC REGULATIONS 2026', title_style))
story.append(Paragraph('Institution of Excellence • Year 2026', meta_style))

sections_2026 = [
    ('1. ATTENDANCE POLICY',
     'Students must maintain a minimum attendance of 80% in all theory classes and 85% in practical/laboratory sessions to be eligible to appear in the end-semester examination.'),
    ('2. MINIMUM PASSING SCORE',
     'A student must secure a minimum of 50% marks in the end-semester examination to pass the subject.'),
    ('3. LIBRARY BORROWING LIMIT',
     'Students are permitted to borrow a maximum of 5 books at a time from the library for a loan period of 14 days.'),
    ('4. BACK-PAPER EXAMINATION CONDUCT',
     'Students appearing for back-paper examinations are not permitted to attend regular classes during the examination period.'),
    ('5. PLACEMENT ELIGIBILITY & TRAINING',
     'The placement cell will conduct a minimum of 4 pre-placement training sessions per semester for final-year students.'),
    ('6. ACADEMIC INTEGRITY & UNFAIR MEANS',
     'Students found using unfair means during any examination, including possession of unauthorised materials or electronic devices, will be awarded zero marks for the entire examination and referred to the Disciplinary Committee.'),
]

for title, content in sections_2026:
    story.append(Paragraph(title, heading_style))
    story.append(Paragraph(content, body_style))

doc.build(story)
print("✓ Created Academic_Regulations_2026.pdf")

print("\n✅ Sample PDFs created successfully!")
print("   - Academic_Regulations_2025.pdf")
print("   - Academic_Regulations_2026.pdf")
print("\nExpected changes:")
print("   1. Attendance: 75% → 80% (SUBSTANTIVE)")
print("   2. Pass score: 40% → 50% (SUBSTANTIVE)")
print("   3. Book limit: 3 → 5 (SUBSTANTIVE)")
print("   4. Back-paper permission: allowed → not allowed (SUBSTANTIVE)")
print("   5. Training sessions: 2 → 4 (SUBSTANTIVE)")
print("   6. Academic Integrity: NEW clause (ADDED)")
