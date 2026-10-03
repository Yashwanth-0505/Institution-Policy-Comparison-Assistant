from ml_pipeline.detector import _extract_signals

# eval_025
a = "3.1 Hostel curfew timings are 10:00 PM for all residents. Residents returning after curfew must sign the late entry register maintained at the security gate."
b = "4.1 Hostel curfew timings are 10:00 PM for all residents. Residents returning after curfew must sign the late entry register maintained at the security gate."

signals, ratio = _extract_signals(a, b)
print(f"eval_025 ratio: {ratio:.4f}")
for s in signals:
    print(f"  {s.signal_type}: {s.old_value} -> {s.new_value}")

print()

# eval_026
a = "2. The Disciplinary Committee shall comprise the Principal, two senior faculty members nominated by the Academic Council, and one student representative."
b = "3. The Disciplinary Committee shall comprise the Principal, two senior faculty members nominated by the Academic Council, and one student representative."

signals, ratio = _extract_signals(a, b)
print(f"eval_026 ratio: {ratio:.4f}")
for s in signals:
    print(f"  {s.signal_type}: {s.old_value} -> {s.new_value}")

print()

# eval_004 (should pass)
a = "Students who fail to meet the minimum attendance requirement will not be permitted to appear in the end-semester examination and will be awarded a detained status."
b = "Students who fail to achieve the minimum attendance requirement will not be eligible to appear in the end-semester examination and will be given a detained status."

signals, ratio = _extract_signals(a, b)
print(f"eval_004 ratio: {ratio:.4f}")
for s in signals:
    print(f"  {s.signal_type}: {s.old_value} -> {s.new_value}")
