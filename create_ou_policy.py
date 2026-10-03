"""
Create an official Osmania University B.Tech Academic Regulations 2026 PDF using PyMuPDF.
"""
from pathlib import Path
import fitz

out_dir = Path('sample_policies')
out_dir.mkdir(exist_ok=True)
pdf_path = out_dir / 'Osmania_University_Academic_Regulations_2026.pdf'

doc = fitz.open()
page = doc.new_page(width=612, height=792)  # Letter size

text = """OSMANIA UNIVERSITY
HYDERABAD - 500 007, TELANGANA, INDIA
ACADEMIC REGULATIONS FOR B.TECH DEGREE PROGRAMME (CBCS)
Applicable for the Academic Year 2025-2026 onwards

1. ATTENDANCE REQUIREMENTS
A student shall be eligible to appear for the Semester End Examinations if he/she acquires a minimum of 75% of attendance in aggregate of all subjects for that semester. A relaxation of up to 10% in attendance may be granted by the Vice-Chancellor on medical grounds.

2. MINIMUM PASSING CRITERIA AND GRADING
A student shall be declared to have passed a theory course if he/she secures a minimum of 40% marks in the Semester End Examination and a minimum of 50% marks in aggregate (Internal Evaluation + Semester End Examination combined).

3. CREDIT CONDITIONS FOR PROMOTION
A student shall be promoted from II Year to III Year B.Tech if he/she secures at least 50% of total credits up to II Year I Semester. For promotion from III Year to IV Year, a student must secure at least 60% of total credits up to III Year I Semester.

4. BACK-PAPER AND RE-EXAMINATION CONDUCT
Students appearing for back-paper supplementary examinations shall not be permitted to attend regular laboratory or practical sessions during the examination period without prior written approval from the Dean of Academic Affairs.

5. PLACEMENT ELIGIBILITY AND INTERNSHIP MANDATE
All B.Tech students must complete a minimum of 2 compulsory industrial internships (minimum 4 weeks each) prior to the VII Semester to be eligible for Campus Placement Drives.

6. ACADEMIC INTEGRITY AND PREVENTION OF UNFAIR MEANS
Any student found indulging in malpractice or unfair means during examinations shall be subject to disciplinary action by the Malpractice Committee, including cancellation of performance in all subjects of that semester and debarment for up to 2 semesters.
"""

# Insert text onto page
rect = fitz.Rect(50, 50, 562, 742)
page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))

doc.save(str(pdf_path))
doc.close()
print(f"Successfully generated {pdf_path}")
