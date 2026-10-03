"""
create_samples.py — Generate two realistic institutional policy PDFs for demo.
Run once: python create_samples.py
"""
import fitz
from pathlib import Path

OUT = Path("d:/jig/sample_policies")
OUT.mkdir(exist_ok=True)


def make_pdf(path: str, title: str, sections: list[tuple[str, str]]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60

    # Title
    page.insert_text((50, y), title, fontsize=16, fontname="helv", color=(0.1, 0.1, 0.5))
    y += 30
    page.insert_text((50, y), "Greenfield Institute of Technology & Management", fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
    y += 20
    page.insert_text((50, y), "_" * 80, fontsize=8, fontname="helv", color=(0.7, 0.7, 0.7))
    y += 20

    for heading, body in sections:
        if y > 760:
            page = doc.new_page(width=595, height=842)
            y = 60
        # Heading
        page.insert_text((50, y), heading, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.45))
        y += 16
        # Body — wrap long lines
        words = body.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 80:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            if y > 760:
                page = doc.new_page(width=595, height=842)
                y = 60
            page.insert_text((58, y), ln, fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
            y += 13
        y += 10

    doc.save(path)
    doc.close()
    print(f"Created: {path}")


# ── Policy v1 ──────────────────────────────────────────────────
V1_SECTIONS = [
    ("1. Attendance Policy",
     "Students must maintain a minimum attendance of 75% in all theory classes and 85% in practical sessions to be eligible to appear in the end-semester examination."),
    ("1.1 Condonation",
     "Students with attendance between 65% and 74% may apply for medical condonation by submitting a valid medical certificate from a registered practitioner within 7 days."),
    ("1.2 Detained Status",
     "Students who fail to meet the minimum attendance requirement shall not be permitted to sit the end-semester examination and will be awarded a detained status for that subject."),
    ("2. Academic Examination Policy",
     "A student must secure a minimum of 40% marks in the end-semester examination to pass the subject. The examination timetable shall be published at least 15 days before commencement."),
    ("2.1 Back Paper",
     "Students appearing for back-paper examinations are permitted to attend regular classes during the examination period."),
    ("2.2 Unfair Means",
     "Students found in possession of unauthorised materials during examinations will be awarded zero in that paper and the matter will be reported to the academic council."),
    ("3. Lecturer Policy",
     "Faculty members are required to submit lesson plans for the entire semester to the Head of Department before the commencement of classes. Faculty should attend at least one national or international conference per academic year."),
    ("3.1 Appraisal",
     "Faculty must submit their annual self-appraisal report to the administration by 31st March each year."),
    ("4. Disciplinary Policy",
     "The Disciplinary Committee must issue a show-cause notice to the accused student before initiating formal disciplinary proceedings. Students should report any ragging incident within 24 hours."),
    ("4.1 Penalties",
     "A student accumulating more than 3 disciplinary warnings in a semester will be placed on academic probation."),
    ("5. Placement Policy",
     "Students wishing to apply for placement activities must register with the placement cell by 31st August of their final year. The placement cell will conduct a minimum of 2 pre-placement training sessions per semester."),
    ("5.1 Eligibility",
     "Students with active backlogs are permitted to participate in campus placement activities subject to company eligibility criteria."),
    ("6. Financial and Administrative Policy",
     "Annual fee payment must be completed by 30th June of each academic year to avoid late payment penalties. A late fee of Rs. 100 per day will be charged for examination fee payments made after the prescribed last date."),
    ("6.1 Scholarships",
     "The institution will provide a merit scholarship of Rs. 5000 per semester to students who rank in the top 5% of their batch in the end-semester examinations."),
    ("7. Hostel Policy",
     "Hostel curfew timings are 10:00 PM for all residents. Residents returning after curfew must sign the late entry register at the security gate. Hostel residents may entertain guests in common areas between 9:00 AM and 6:00 PM."),
    ("7.1 Vacating",
     "Hostel residents must vacate their rooms within 48 hours of the completion of their final examination of the academic year."),
    ("8. Library Policy",
     "Students are permitted to borrow a maximum of 3 books at a time from the library for a loan period of 14 days. Mobile phones are allowed inside the library premises provided they are kept on silent mode."),
    ("8.1 Fines",
     "All library books must be returned before the commencement of end-semester examinations. Failure to return books will result in a fine of Rs. 2 per day per book."),
]

# ── Policy v2 (with deliberate changes) ───────────────────────
V2_SECTIONS = [
    ("1. Attendance Policy",
     "Students must maintain a minimum attendance of 75% in all theory classes and 85% in practical sessions to be eligible to appear in the end-semester examination."),
    ("1.1 Condonation",
     "Medical condonation provisions for attendance shortfall have been discontinued with effect from this academic year."),
    ("1.2 Detained Status",
     "Students failing to achieve the required minimum attendance will be detained and are not eligible to appear in the end-semester examination."),
    ("2. Academic Examination Policy",
     "A student must secure a minimum of 50% marks in the end-semester examination to pass the subject. The examination timetable shall be published at least 15 days before commencement."),
    ("2.1 Back Paper",
     "Students appearing for back-paper examinations are not permitted to attend regular classes during the examination period."),
    ("2.2 Unfair Means",
     "Students found using unfair means during any examination, including possession of unauthorised materials or electronic devices, will be awarded zero marks for the entire examination and referred to the Disciplinary Committee."),
    ("3. Lecturer Policy",
     "All teaching staff must submit their semester lesson plans to the Head of Department prior to the start of classes. Faculty members must attend at least one national or international conference per academic year."),
    ("3.1 Appraisal",
     "Faculty must submit their annual self-appraisal report to the administration by 28th February each year."),
    ("4. Disciplinary Policy",
     "The Disciplinary Committee should issue a show-cause notice to the accused student before initiating formal disciplinary proceedings. Students must report any ragging incident to the Anti-Ragging Committee within 24 hours."),
    ("4.1 Penalties",
     "A student accumulating more than 2 disciplinary warnings in a semester will be placed on academic probation."),
    ("5. Placement Policy",
     "Students wishing to apply for placement activities must register with the placement cell by 15th July of their final year. The placement cell will conduct a minimum of 4 pre-placement training sessions per semester."),
    ("5.1 Eligibility",
     "Students with active backlogs are not permitted to participate in campus placement activities."),
    ("5.2 Offer Notification",
     "Students who secure placement offers must inform the placement cell in writing within 48 hours of receiving the offer letter."),
    ("6. Financial and Administrative Policy",
     "Annual fee payment must be completed by 31st May of each academic year to avoid late payment penalties. A late fee of Rs. 200 per day will be charged for examination fee payments made after the prescribed last date."),
    ("7. Hostel Policy",
     "Hostel curfew timings are 10:00 PM for all residents. Residents returning after curfew must sign the late entry register at the security gate. Hostel residents may not entertain outside guests in any area of the hostel without written permission from the Chief Warden."),
    ("7.1 Vacating",
     "Students residing in the hostel are required to vacate their accommodation within 48 hours after completing their final examination for the academic year."),
    ("8. Library Policy",
     "Students are permitted to borrow a maximum of 5 books at a time from the library for a loan period of 14 days. Mobile phones are not allowed inside the library premises under any circumstances."),
    ("8.1 Fines",
     "All library books must be returned before the commencement of end-semester examinations. Failure to return books will result in a fine of Rs. 2 per day per book."),
]

make_pdf(str(OUT / "Greenfield_Policy_2023.pdf"), "INSTITUTIONAL POLICY MANUAL 2023", V1_SECTIONS)
make_pdf(str(OUT / "Greenfield_Policy_2024.pdf"), "INSTITUTIONAL POLICY MANUAL 2024", V2_SECTIONS)
print("\nSample PDFs ready in sample_policies/")
