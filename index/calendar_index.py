import sys
from pathlib import Path

# So "mappings" is found when run as script or in Jupyter (start kernel from project root)
_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from elasticsearch import helpers

from common.embedding_model import model_large
from common.es_client import es
from mappings import calendar_mapping
import pandas as pd

def create_index(index_name):
    try:
        if es.indices.exists(index = index_name):
            es.indices.delete(index = index_name)

        mapping = calendar_mapping.mapping["mappings"]
        es.indices.create(index = index_name, mappings = mapping)
        print(f"Index {index_name} created successfully.")
    except Exception as e:
        raise Exception(f"Failed to create index {index_name}: {str(e)}")


# Indexing
index_name = "iit_calendar"

def build_semantic_text(row):
    term = row["term"]
    event_name = row["event_name"]
    start_date = row["start_date"]
    end_date = row["end_date"]

    # Lookup keyed by (event_name, term, start_date) to handle repeated event names
    key = (event_name, term, start_date)

    lookup = {
        # ── 2026-27 Calendar Year holidays ────────────────────────────────────────
        ("New Year's Day", "2026-27 Calendar Year", "2026-01-01"):
            "New Year's Day is a paid university holiday on January 1, 2026. IIT offices are closed and no classes are held. This is the first official university holiday of the 2026-27 academic year.",

        ("Floating Holiday", "2026-27 Calendar Year", "2026-01-02"):
            "January 2, 2026 is a floating holiday for IIT employees in the 2026-27 calendar year. University offices are closed on this day. This floating holiday falls immediately after New Year's Day.",

        ("Early Departure", "2026-27 Calendar Year", "2026-01-16"):
            "January 16, 2026 is an early departure day for IIT employees. Staff are permitted to leave early on this date as part of the 2026-27 university holiday schedule.",

        ("Martin Luther King, Jr. Day", "2026-27 Calendar Year", "2026-01-19"):
            "Martin Luther King Jr. Day is a paid university holiday on January 19, 2026. IIT offices are closed. This is a federal and university holiday honoring Dr. King.",

        ("Early Departure", "2026-27 Calendar Year", "2026-05-22"):
            "May 22, 2026 is an early departure day for IIT employees. Staff leave early on this date as part of the 2026-27 university holiday schedule, preceding Memorial Day weekend.",

        ("Memorial Day", "2026-27 Calendar Year", "2026-05-25"):
            "Memorial Day is a paid university holiday on May 25, 2026. IIT offices are closed and no classes are held. This federal holiday falls on the last Monday of May.",

        ("Early Departure", "2026-27 Calendar Year", "2026-06-18"):
            "June 18, 2026 is an early departure day for IIT employees. Staff leave early on this date, preceding the Juneteenth holiday on June 19.",

        ("Juneteenth Day", "2026-27 Calendar Year", "2026-06-19"):
            "Juneteenth Day is a paid university holiday on June 19, 2026. IIT offices are closed and no classes are held. Juneteenth is a federal holiday commemorating the emancipation of enslaved African Americans.",

        ("Early Departure", "2026-27 Calendar Year", "2026-07-02"):
            "July 2, 2026 is an early departure day for IIT employees. Staff leave early on this date, preceding the Independence Day holiday on July 3.",

        ("Independence Day", "2026-27 Calendar Year", "2026-07-03"):
            "Independence Day is observed on July 3, 2026 as a paid university holiday. IIT offices are closed. The federal Fourth of July holiday is observed on July 3 this year.",

        ("Early Departure", "2026-27 Calendar Year", "2026-09-04"):
            "September 4, 2026 is an early departure day for IIT employees. Staff leave early on this date, preceding Labor Day on September 7.",

        ("Labor Day", "2026-27 Calendar Year", "2026-09-07"):
            "Labor Day is a paid university holiday on September 7, 2026. IIT offices are closed. This federal holiday falls on the first Monday of September.",

        ("Early Departure", "2026-27 Calendar Year", "2026-11-25"):
            "November 25, 2026 is an early departure day for IIT employees. Staff leave early on this date, the Wednesday before Thanksgiving break.",

        ("Thanksgiving Day", "2026-27 Calendar Year", "2026-11-26"):
            "Thanksgiving Day is a paid university holiday on November 26, 2026. IIT offices are closed. This is the fourth Thursday of November.",

        ("Floating Holiday", "2026-27 Calendar Year", "2026-11-27"):
            "November 27, 2026 is a floating holiday for IIT employees. University offices are closed on the Friday after Thanksgiving, extending the Thanksgiving break to a four-day weekend.",

        ("Early Departure", "2026-27 Calendar Year", "2026-12-23"):
            "December 23, 2026 is an early departure day for IIT employees. Staff leave early on this date, preceding the Christmas and winter university holidays.",

        ("Floating Holiday", "2026-27 Calendar Year", "2026-12-24"):
            "December 24, 2026 is a floating holiday for IIT employees (Christmas Eve). University offices are closed. This holiday is part of the end-of-year winter break period.",

        ("Christmas Day", "2026-27 Calendar Year", "2026-12-25"):
            "Christmas Day is a paid university holiday on December 25, 2026. IIT offices are closed. This is a federal holiday and part of the university's winter break.",

        ("University Holidays", "2026-27 Calendar Year", "2026-12-28"):
            "IIT University Holidays run from December 28 through December 31, 2026. University offices are closed during this entire period. This is the year-end winter break for all IIT staff and employees before New Year's Day.",

        ("New Year's Day", "2026-27 Calendar Year", "2027-01-01"):
            "New Year's Day is a paid university holiday on January 1, 2027. IIT offices are closed. This marks the start of the calendar year 2027 and is the final holiday in the 2026-27 academic year holiday schedule.",

        # ── Coursera Spring 2026 Term A ───────────────────────────────────────────
        ("Courses Begin-Coursera A", "Coursera Spring 2026 (Term A)", "2026-01-12"):
            "Coursera Spring 2026 Term A courses begin on January 12, 2026. This is the first day of instruction for Coursera online courses in Spring Term A. Coursera Term A runs from January 12 through March 7, 2026.",

        ("Martin Luther King, Jr. Day—No Classes", "Coursera Spring 2026 (Term A)", "2026-01-19"):
            "Martin Luther King Jr. Day is observed on January 19, 2026 — no Coursera classes are held on this date. Coursera Spring Term A courses resume on January 20.",

        ("Last Day to add a course or drop with no tuition charges", "Coursera Spring 2026 (Term A)", "2026-01-20"):
            "January 20, 2026 is the last day to add a Coursera Spring Term A course or drop without incurring tuition charges. After this date, students cannot add Coursera A courses and dropping will result in tuition charges.",

        ("Spring Degree Conferral Applications Due", "Coursera Spring 2026 (Term A)", "2026-02-02"):
            "February 2, 2026 is the deadline for Spring 2026 degree conferral applications for Coursera students. Students planning to graduate in Spring 2026 must submit their degree conferral application by this date.",

        ("Last day to enroll for PBA courses for credit—Coursera A", "Coursera Spring 2026 (Term A)", "2026-02-09"):
            "February 9, 2026 is the last day to enroll in PBA (Partnership-Based Assessment) Coursera courses for academic credit in Spring Term A. After this date, Coursera A PBA enrollment for credit is closed.",

        ("Last Day to Withdraw—Coursera A", "Coursera Spring 2026 (Term A)", "2026-02-20"):
            "February 20, 2026 is the last day to withdraw from a Coursera Spring Term A course. After this date, students cannot withdraw from Coursera A courses. Withdrawal after this date results in a failing grade.",

        ("Last Day of Courses—Coursera A", "Coursera Spring 2026 (Term A)", "2026-03-07"):
            "March 7, 2026 is the last day of Coursera Spring Term A instruction. All Coursera Term A courses end on this date. Term A final grades are due by March 11, 2026.",

        ("Final Grades Due—Coursera A", "Coursera Spring 2026 (Term A)", "2026-03-11"):
            "March 11, 2026 is the deadline for Coursera Spring Term A final grades submission. Instructors must submit all final grades for Coursera A courses by this date.",

        ("Spring Registration Begins-Coursera A", "Coursera Spring 2026 (Term A)", "2026-11-03"):
            "November 3, 2026 is when registration opens for Coursera Spring 2027 Term A courses. Students can begin enrolling in upcoming Coursera A courses starting this date. This registration opening occurs during Fall 2026.",

        # ── Spring 2026 ───────────────────────────────────────────────────────────
        ("Courses Begin for Spring and Online A Session", "Spring 2026", "2026-01-12"):
            "Spring 2026 semester begins on January 12, 2026. This is the first day of classes for the Spring 2026 term at IIT. All on-campus Spring 2026 courses and Online A session courses start on January 12. When does the Spring 2026 semester start? The Spring semester starts January 12, 2026. When does Spring semester begin? Instruction begins January 12, 2026.",

        ("Martin Luther King, Jr. Day—No Classes", "Spring 2026", "2026-01-19"):
            "Martin Luther King Jr. Day is observed on January 19, 2026. No Spring 2026 classes are held. IIT is closed for this university holiday. Classes resume on January 20, 2026 with a Monday schedule makeup.",

        ("Courses Begin for ID Full Semester and ID A Session", "Spring 2026", "2026-01-20"):
            "Institute of Design (ID) Spring 2026 Full Semester and ID A Session courses begin on January 20, 2026. This is the first day of instruction for Institute of Design students in Spring 2026.",

        ("Last Day to Add/Drop for Full Semester and Online A Session Courses with No Tuition Charges", "Spring 2026", "2026-01-20"):
            "January 20, 2026 is the last day to add or drop Spring 2026 Full Semester or Online A Session courses without any tuition charges. Students who drop on or before January 20 receive a full tuition refund for those courses.",

        ("Monday Classes Meet (Dr. King's Birthday Makeup)", "Spring 2026", "2026-01-23"):
            "January 23, 2026 follows a Monday class schedule as a makeup for the Martin Luther King Jr. Day holiday. Even though January 23 is a Friday, Monday classes are held to compensate for the January 19 university holiday.",

        ("Last Day to Add/Drop for ID Full Semester and ID A Session Courses with No Tuition Charges", "Spring 2026", "2026-01-27"):
            "January 27, 2026 is the last day to add or drop Institute of Design Full Semester or ID A Session Spring 2026 courses without tuition charges. ID students must add or drop by this date to avoid tuition penalties.",

        ("Last Day to Request Late Registration", "Spring 2026", "2026-01-27"):
            "January 27, 2026 is the last day to request late registration for Spring 2026 courses. After this date, late registration requests will not be accepted for the Spring 2026 semester.",

        ("Spring Degree Conferral Applications Due", "Spring 2026", "2026-02-02"):
            "February 2, 2026 is the deadline to submit a Spring 2026 degree conferral application. Students who plan to graduate in Spring 2026 — at the May 16 commencement — must apply by February 2. Late applications may not be processed for Spring commencement.",

        ("Last Day to Withdraw for ID and Online A Session Courses", "Spring 2026", "2026-02-20"):
            "February 20, 2026 is the last day to withdraw from Spring 2026 Institute of Design (ID) and Online A Session courses. After this date, withdrawal from ID or Online A courses is not permitted and may result in a failing grade.",

        ("Fall Incomplete Grades Due", "Spring 2026", "2026-02-23"):
            "February 23, 2026 is the deadline for Fall 2025 incomplete grades to be resolved and submitted. Students who received an Incomplete (I) grade in Fall 2025 must complete their work and faculty must submit grades by this date.",

        ("Midterm Grading Begins", "Spring 2026", "2026-03-04"):
            "Spring 2026 midterm grading begins on March 4, 2026. Faculty start submitting midterm grades for Spring 2026 students. Midterm grades are due by March 13, 2026.",

        ("Last Day of ID and Online A Session Courses", "Spring 2026", "2026-03-07"):
            "March 7, 2026 is the last day of instruction for Spring 2026 Institute of Design (ID) and Online A Session courses. These courses end on March 7. Final grades for Online A session are due March 11.",

        ("Online B Session Courses Begin", "Spring 2026", "2026-03-09"):
            "Spring 2026 Online B Session courses begin on March 9, 2026. This is the first day of instruction for Online B session courses, which run from March 9 through May 9, 2026.",

        ("Final Online A Session Courses Grades Due", "Spring 2026", "2026-03-11"):
            "March 11, 2026 is the deadline for submitting final grades for Spring 2026 Online A Session courses. Instructors must have all Online A final grades entered by this date.",

        ("Spring Final Exam Schedule Published Online", "Spring 2026", "2026-03-11"):
            "The Spring 2026 final exam schedule is published online on March 11, 2026. Students can view their final exam times and locations starting March 11. Final exams themselves take place May 4-9, 2026.",

        ("Midterm Grades Due", "Spring 2026", "2026-03-13"):
            "Spring 2026 midterm grades are due on March 13, 2026. All faculty must submit midterm grades for Spring 2026 students by this deadline. Midterm grades help students assess their academic progress.",

        ("Spring Break Week—No Classes", "Spring 2026", "2026-03-16"):
            "Spring Break for Spring 2026 runs from March 16 through March 21, 2026. No classes are held during this entire week. Spring Break is the mid-semester holiday for the Spring 2026 term. Classes resume on March 23.",

        ("Last Day to Add/Drop for Online B Session Courses with No Tuition Charges", "Spring 2026", "2026-03-17"):
            "March 17, 2026 is the last day to add or drop Spring 2026 Online B Session courses without incurring tuition charges. Students must add or drop Online B courses by this date to receive a full tuition refund.",

        ("Fall Course Schedule Published Online", "Spring 2026", "2026-03-23"):
            "March 23, 2026 is when the Fall 2026 course schedule is published online. Students can begin browsing Fall 2026 course offerings starting this date. Fall 2026 registration itself begins April 6, 2026.",

        ("ID B Session Courses Begin", "Spring 2026", "2026-03-23"):
            "Spring 2026 Institute of Design (ID) B Session courses begin on March 23, 2026. This is the start of instruction for ID B session courses in the Spring 2026 term.",

        ("Last Day to Add/Drop for ID B Session Courses with No Tuition Charges", "Spring 2026", "2026-03-30"):
            "March 30, 2026 is the last day to add or drop Spring 2026 Institute of Design B Session courses without tuition charges. ID B session students must act by this date to avoid tuition penalties.",

        ("Last Day to Withdraw for Full Semester Courses", "Spring 2026", "2026-03-30"):
            "March 30, 2026 is the last day to withdraw from Spring 2026 Full Semester courses. This is the primary Spring 2026 withdrawal deadline that applies to most students. Any date after March 30 is past the Spring 2026 withdrawal deadline — withdrawal is no longer permitted. Students who withdraw after March 30 will not receive a W grade and are liable for full tuition. When asking whether a date is before or after the Spring 2026 withdrawal deadline, the answer is March 30, 2026.",

        ("Fall Registration Begins", "Spring 2026", "2026-04-06"):
            "Fall 2026 registration begins on April 6, 2026. Students can start enrolling in Fall 2026 courses on this date. Registration for Fall 2026 opens during the Spring 2026 semester. When can I register for Fall 2026? Registration for Fall 2026 courses opens April 6, 2026.",

        ("Summer Reinstatement Applications Due for Undergraduate Students", "Spring 2026", "2026-04-15"):
            "April 15, 2026 is the deadline for undergraduate students to submit Summer 2026 reinstatement applications. Students who wish to be reinstated for Summer 2026 must apply by this date.",

        ("Last Day to Withdraw for ID Full Semester Courses", "Spring 2026", "2026-04-17"):
            "April 17, 2026 is the last day to withdraw from Spring 2026 Institute of Design Full Semester courses. ID full semester students must withdraw by this date.",

        ("Last Day to Withdraw for ID and Online B Session Courses", "Spring 2026", "2026-04-17"):
            "April 17, 2026 is the last day to withdraw from Spring 2026 Institute of Design B Session and Online B Session courses. Students in these sessions must submit withdrawal requests by April 17.",

        ("Last Day of Spring Courses", "Spring 2026", "2026-05-02"):
            "May 2, 2026 is the last day of Spring 2026 classes. All Spring 2026 on-campus and main session instruction ends on May 2. The Spring 2026 semester concludes with final exams starting May 4. When does the Spring 2026 semester end? The last day of Spring 2026 classes is May 2, 2026.",

        ("Last Day to Request an Incomplete Grade", "Spring 2026", "2026-05-03"):
            "May 3, 2026 is the last day for Spring 2026 students to request an Incomplete (I) grade from their instructor. Students who cannot complete coursework by the end of the semester must request an Incomplete by May 3.",

        ("Final Exam Week/Final Grading Begins on May 4", "Spring 2026", "2026-05-04"):
            "Spring 2026 Final Exam Week runs from May 4 through May 9, 2026. All Spring 2026 final examinations are held during this period. Final grading also begins May 4. When are Spring 2026 finals? Final exams are May 4-9, 2026.",

        ("Last Day of ID Full Semester and ID and Online B Session Courses", "Spring 2026", "2026-05-09"):
            "May 9, 2026 is the last day of Spring 2026 Institute of Design Full Semester, ID B Session, and Online B Session courses. These courses conclude on the same day as the Spring 2026 final exam period ends.",

        ("Final Grades Due at Noon (CST)", "Spring 2026", "2026-05-13"):
            "Spring 2026 final grades are due at noon CST on May 13, 2026. All Spring 2026 instructors must submit final grades by noon on May 13. Students can expect grades to be available shortly after this deadline.",

        ("Spring Degree Conferral and Commencement", "Spring 2026", "2026-05-16"):
            "Spring 2026 Commencement and Degree Conferral is held on May 16, 2026. This is IIT's Spring 2026 graduation ceremony where degrees are officially awarded to Spring 2026 graduates. When is IIT Spring 2026 graduation? Commencement is May 16, 2026.",

        # ── Coursera Spring 2026 Term B ───────────────────────────────────────────
        ("Courses Begin—Courses B", "Coursera Spring 2026 (Term B)", "2026-03-09"):
            "Coursera Spring 2026 Term B courses begin on March 9, 2026. This is the first day of instruction for Coursera online courses in Spring Term B. Term B runs from March 9 through May 9, 2026.",

        ("Spring Break Week—No Classes", "Coursera Spring 2026 (Term B)", "2026-03-16"):
            "Coursera Spring Term B has no classes during Spring Break Week, March 16-21, 2026. Coursera B instruction resumes March 22. The add/drop deadline for Coursera B falls within Spring Break on March 17.",

        ("Last Day to add a course or drop with no tuition charges—Coursera B", "Coursera Spring 2026 (Term B)", "2026-03-17"):
            "March 17, 2026 is the last day to add a Coursera Spring Term B course or drop without incurring tuition charges. This deadline falls during Spring Break. After March 17, dropping Coursera B courses results in tuition charges.",

        ("Last day to enroll for PBA courses for credit—Coursera B", "Coursera Spring 2026 (Term B)", "2026-04-06"):
            "April 6, 2026 is the last day to enroll in PBA (Partnership-Based Assessment) Coursera courses for academic credit in Spring Term B. After this date, Coursera B PBA enrollment for credit is closed.",

        ("Last Day to Withdraw—Coursera B", "Coursera Spring 2026 (Term B)", "2026-04-17"):
            "April 17, 2026 is the last day to withdraw from a Coursera Spring Term B course. After this date, students cannot withdraw from Coursera B courses without academic penalty.",

        ("Last Day of Classes—Coursera B", "Coursera Spring 2026 (Term B)", "2026-05-09"):
            "May 9, 2026 is the last day of Coursera Spring Term B instruction. All Coursera B courses end on this date. Final grades for Coursera B are due by May 13, 2026.",

        ("Final Grades Due—Coursera B", "Coursera Spring 2026 (Term B)", "2026-05-13"):
            "May 13, 2026 is the deadline for Coursera Spring Term B final grades submission. Instructors must submit all Coursera B final grades by this date.",

        ("Spring Degree Conferral and Commencement", "Coursera Spring 2026 (Term B)", "2026-05-16"):
            "Spring 2026 Commencement and Degree Conferral for Coursera students is held on May 16, 2026. Coursera Spring 2026 graduates receive their degrees at this ceremony.",

        ("Spring Registration Begins-Coursera", "Coursera Spring 2026 (Term B)", "2026-11-03"):
            "November 3, 2026 is when registration opens for Coursera Spring 2027 courses (Term B track). Students can begin enrolling in upcoming Coursera courses starting this date. This registration opens during Fall 2026.",

        # ── Coursera Summer 2026 ──────────────────────────────────────────────────
        ("Summer Registration Begins-Coursera", "Coursera Summer 2026", "2026-01-26"):
            "Coursera Summer 2026 registration begins on January 26, 2026. Students can start enrolling in Coursera Summer 2026 online courses on this date. Registration opens in January, well ahead of the June 15 course start date.",

        ("Summer Courses Begin", "Coursera Summer 2026", "2026-06-15"):
            "Coursera Summer 2026 courses begin on June 15, 2026. This is the first day of instruction for Coursera online courses in the Summer 2026 term. The Coursera summer session runs from June 15 through August 8, 2026.",

        ("Juneteenth Day Observance—No Classes", "Coursera Summer 2026", "2026-06-18"):
            "June 18, 2026 is the Juneteenth holiday observance — no Coursera Summer 2026 classes are held. Coursera summer instruction resumes June 19.",

        ("Last day to add a course or drop with no tuition charges—Coursera", "Coursera Summer 2026", "2026-06-23"):
            "June 23, 2026 is the last day to add a Coursera Summer 2026 course or drop without incurring tuition charges. After this date, dropping Coursera summer courses results in tuition charges.",

        ("Last day to enroll for PBA for credit-Summer Term", "Coursera Summer 2026", "2026-07-13"):
            "July 13, 2026 is the last day to enroll in PBA courses for academic credit in Coursera Summer 2026. After this date, PBA enrollment for credit in the Coursera summer term is closed.",

        ("Last Day to Withdraw—Coursera Summer Term", "Coursera Summer 2026", "2026-07-24"):
            "July 24, 2026 is the last day to withdraw from a Coursera Summer 2026 course. After this date, students cannot withdraw from Coursera summer courses without academic penalty.",

        ("Last Day of Courses—Coursera Summer Term", "Coursera Summer 2026", "2026-08-08"):
            "August 8, 2026 is the last day of Coursera Summer 2026 instruction. All Coursera summer courses end on this date. Final grades for Coursera Summer are due by August 12, 2026.",

        ("Final Grades Due—Coursera Summer Term", "Coursera Summer 2026", "2026-08-12"):
            "August 12, 2026 is the deadline for Coursera Summer 2026 final grades. Instructors must submit all Coursera summer final grades by this date.",

        ("Summer Combined Session Degree Conferral", "Coursera Summer 2026", "2026-08-15"):
            "August 15, 2026 is the Summer 2026 Combined Session Degree Conferral for Coursera students. Coursera students completing their degrees in Summer 2026 receive their degrees at this ceremony.",

        # ── Summer 2026 ───────────────────────────────────────────────────────────
        ("Courses Begin—Summer 1", "Summer 2026", "2026-05-18"):
            "Summer 2026 Session 1 courses begin on May 18, 2026. This is the first day of instruction for Summer 1 courses. Summer Session 1 runs from May 18 through July 11, 2026. When does Summer 2026 start? Summer Session 1 starts May 18, 2026.",

        ("Last Day to Add/Drop Courses with No Tuition Charges—Summer 1", "Summer 2026", "2026-05-22"):
            "May 22, 2026 is the last day to add or drop Summer 2026 Session 1 courses without tuition charges. Students must add or drop Summer 1 courses by May 22 to receive a full refund.",

        ("Memorial Day—No Classes", "Summer 2026", "2026-05-25"):
            "Memorial Day is observed on May 25, 2026 — no Summer 2026 classes are held. Summer Session 1 classes resume on May 26. Memorial Day is a university holiday.",

        ("Last Day to Request Late Registration—Summer Session 1", "Summer 2026", "2026-05-29"):
            "May 29, 2026 is the last day to request late registration for Summer 2026 Session 1. After this date, late registration for Summer 1 is not accepted.",

        ("Summer Combined Session Degree Conferral Applications Due", "Summer 2026", "2026-06-05"):
            "June 5, 2026 is the deadline for Summer 2026 degree conferral applications. Students planning to graduate in Summer 2026 — at the August 15 ceremony — must submit their application by June 5.",

        ("Courses Begin—Summer 2", "Summer 2026", "2026-06-15"):
            "Summer 2026 Session 2 courses begin on June 15, 2026. This is the first day of instruction for Summer 2 courses. Summer Session 2 runs from June 15 through August 8, 2026. When does Summer Session 2 start? It starts June 15, 2026.",

        ("Juneteenth Day—No Classes", "Summer 2026", "2026-06-19"):
            "Juneteenth is observed on June 19, 2026 — no Summer 2026 classes are held. Both Summer 1 and Summer 2 sessions have no classes on June 19. Classes resume June 22.",

        ("Last Day to Add/Drop Courses with No Tuition Charges—Summer 2", "Summer 2026", "2026-06-19"):
            "June 19, 2026 is the last day to add or drop Summer 2026 Session 2 courses without tuition charges. Students must add or drop Summer 2 courses by June 19 to receive a full refund.",

        ("Last Day to Withdraw—Summer 1", "Summer 2026", "2026-06-23"):
            "June 23, 2026 is the last day to withdraw from Summer 2026 Session 1 courses. After this date, withdrawal from Summer 1 courses is not permitted and may result in a failing grade.",

        ("Last Day to Request Late Registration—Summer Session 2", "Summer 2026", "2026-06-26"):
            "June 26, 2026 is the last day to request late registration for Summer 2026 Session 2. After this date, late registration for Summer 2 is not accepted.",

        ("Independence Day (Observed)—No Classes", "Summer 2026", "2026-07-03"):
            "Independence Day (July 4th) is observed on July 3, 2026 — no Summer 2026 classes are held. Both Summer Session 1 and Session 2 have no classes on this date. Classes resume July 6.",

        ("Last Day of Courses—Summer 1", "Summer 2026", "2026-07-11"):
            "July 11, 2026 is the last day of Summer 2026 Session 1 instruction. All Summer 1 courses end on this date. Summer 1 final grades are due by July 15. When does Summer 1 end? Summer Session 1 ends July 11, 2026.",

        ("Final Grades Due at Noon (CST)—Summer 1", "Summer 2026", "2026-07-15"):
            "Summer 2026 Session 1 final grades are due at noon CST on July 15, 2026. All Summer 1 instructors must submit final grades by noon on July 15.",

        ("Last Day to Withdraw—Summer 2", "Summer 2026", "2026-07-21"):
            "July 21, 2026 is the last day to withdraw from Summer 2026 Session 2 courses. After this date, withdrawal from Summer 2 courses is not permitted and may result in a failing grade.",

        ("PBA: Last Day for Converting MOOC into a Credit—Coursera Summer Term", "Summer 2026", "2026-07-21"):
            "July 21, 2026 is the last day for Summer 2026 students to convert a Coursera MOOC (Massive Open Online Course) into academic credit through the PBA (Partnership-Based Assessment) program.",

        ("Last Day of Courses—Summer 2", "Summer 2026", "2026-08-08"):
            "August 8, 2026 is the last day of Summer 2026 Session 2 instruction. All Summer 2 courses end on this date. Summer 2 final grades are due by August 11. When does Summer 2 end? Summer Session 2 ends August 8, 2026.",

        ("Final Grades Due at Noon (CST)—Summer 2", "Summer 2026", "2026-08-11"):
            "Summer 2026 Session 2 final grades are due at noon CST on August 11, 2026. All Summer 2 instructors must submit final grades by noon on August 11.",

        ("Summer Combined Session Degree Conferral", "Summer 2026", "2026-08-15"):
            "August 15, 2026 is the Summer 2026 Combined Session Degree Conferral ceremony. Students completing their degrees in Summer 2026 receive their diplomas at this graduation event. When is Summer 2026 graduation? The Summer degree conferral is August 15, 2026.",

        # ── Fall 2026 ─────────────────────────────────────────────────────────────
        ("Fall term starts", "Fall 2026", "2026-08-17"):
            "Fall 2026 semester begins on August 17, 2026. This is the first day of instruction for the Fall 2026 term at IIT. All Fall 2026 on-campus courses start on August 17. When does the Fall 2026 semester start? The Fall 2026 semester starts August 17, 2026. When do Fall 2026 classes begin? Classes begin August 17, 2026.",

        ("Last Day to Add/Drop for Full Semester Courses with No Tuition Charges", "Fall 2026", "2026-08-25"):
            "August 25, 2026 is the last day to add or drop Fall 2026 Full Semester courses without tuition charges. Students must add or drop by August 25 to receive a full refund for Fall 2026 courses.",

        ("Last Day to Request Late Registration", "Fall 2026", "2026-09-01"):
            "September 1, 2026 is the last day to request late registration for Fall 2026 courses. After this date, late registration requests for Fall 2026 will not be accepted.",

        ("Fall Degree Conferral Applications Due", "Fall 2026", "2026-09-04"):
            "September 4, 2026 is the deadline to submit a Fall 2026 degree conferral application. Students who plan to graduate in Fall 2026 — at the December 19 ceremony — must apply by September 4.",

        ("Labor Day—No Classes", "Fall 2026", "2026-09-07"):
            "Labor Day is observed on September 7, 2026 — no Fall 2026 classes are held. Fall 2026 instruction resumes September 8. Labor Day is a university holiday falling on the first Monday of September.",

        ("Spring and Summer Incomplete Grades Due", "Fall 2026", "2026-10-05"):
            "October 5, 2026 is the deadline for Spring 2026 and Summer 2026 incomplete grades to be resolved. Students who received an Incomplete (I) in Spring or Summer 2026 must complete their work by this date.",

        ("Fall Break Day—No Classes", "Fall 2026", "2026-10-12"):
            "Fall Break runs October 12-13, 2026. No Fall 2026 classes are held on these two days. This mid-semester break gives students a brief rest from instruction. Classes resume October 14, 2026.",

        ("Midterm Grading Begins", "Fall 2026", "2026-10-14"):
            "Fall 2026 midterm grading begins on October 14, 2026. Faculty start submitting midterm grades for Fall 2026 students. Midterm grades are due by October 23, 2026.",

        ("Fall Final Exam Schedule Published Online", "Fall 2026", "2026-10-21"):
            "The Fall 2026 final exam schedule is published online on October 21, 2026. Students can view their Fall 2026 final exam times and locations starting October 21. Final exams themselves take place December 7-12, 2026.",

        ("Midterm Grades Due", "Fall 2026", "2026-10-23"):
            "Fall 2026 midterm grades are due on October 23, 2026. All faculty must submit midterm grades for Fall 2026 students by this deadline.",

        ("Spring and Summer Course Schedules Published Online", "Fall 2026", "2026-10-26"):
            "October 26, 2026 is when the Spring 2027 and Summer 2027 course schedules are published online. Students can begin browsing upcoming course offerings starting this date. Spring/Summer 2027 registration opens November 9, 2026.",

        ("Last Day to Withdraw for Full Semester Courses", "Fall 2026", "2026-11-02"):
            "November 2, 2026 is the last day to withdraw from Fall 2026 Full Semester courses. Students who withdraw after this date will not receive a 'W' grade and are liable for full tuition. This is the final withdrawal deadline for Fall 2026.",

        ("Spring and Summer Registration Begins", "Fall 2026", "2026-11-09"):
            "Spring 2027 and Summer 2027 registration begins on November 9, 2026. Students can start enrolling in Spring 2027 and Summer 2027 courses on this date. Registration opens during the Fall 2026 semester. When can I register for Spring 2027? Registration opens November 9, 2026. Note: this is separate from Spring 2026 — Spring 2026 already started in January 2026.",

        ("Spring Reinstatement Applications Due for Undergraduate Students", "Fall 2026", "2026-11-15"):
            "November 15, 2026 is the deadline for undergraduate students to submit Spring 2027 reinstatement applications. Students who wish to be reinstated for Spring 2027 must apply by this date.",

        ("Thanksgiving Break—No Classes", "Fall 2026", "2026-11-25"):
            "Thanksgiving Break runs from November 25 through November 28, 2026. No Fall 2026 classes are held during this four-day break. Fall 2026 instruction resumes November 30, 2026.",

        ("Last Day of Fall Courses", "Fall 2026", "2026-12-05"):
            "December 5, 2026 is the last day of Fall 2026 classes. All Fall 2026 instruction ends on December 5. The Fall 2026 semester concludes with final exams starting December 7. When does Fall 2026 semester end? The last day of Fall 2026 classes is December 5, 2026.",

        ("Last Day to Request an Incomplete Grade", "Fall 2026", "2026-12-06"):
            "December 6, 2026 is the last day for Fall 2026 students to request an Incomplete (I) grade from their instructor. Students who cannot complete Fall 2026 coursework must request an Incomplete by December 6.",

        ("Final Exam Week/Final Grading Begins on December 7", "Fall 2026", "2026-12-07"):
            "Fall 2026 Final Exam Week runs from December 7 through December 12, 2026. All Fall 2026 final examinations are held during this period. Final grading also begins December 7. When are Fall 2026 finals? Final exams are December 7-12, 2026.",

        ("Final Grades Due at Noon (CST)", "Fall 2026", "2026-12-16"):
            "Fall 2026 final grades are due at noon CST on December 16, 2026. All Fall 2026 instructors must submit final grades by noon on December 16.",

        ("Fall Degree Conferral", "Fall 2026", "2026-12-19"):
            "Fall 2026 Degree Conferral (graduation) is held on December 19, 2026. Students completing their degrees in Fall 2026 receive their diplomas at this ceremony. When is Fall 2026 graduation? The Fall 2026 degree conferral is December 19, 2026.",
    }

    text = lookup.get(key)
    if text:
        return text

    # Fallback for any unmatched events (should not occur for current 118 events)
    if start_date == end_date:
        return f"{event_name} occurs during the {term} term on {start_date}."
    else:
        return f"{event_name} occurs during the {term} term from {start_date} to {end_date}."

if __name__ == "__main__":
    data = pd.read_json(_root / "data" / "calendar_chunks.json")

    actions = []
    for i, row in data.iterrows():
        semantic_text = build_semantic_text(row)
        embedding = model_large.encode(
            f"passage: {semantic_text}",
            normalize_embeddings=True
        ).tolist()

        actions.append({
            "_index": index_name,
            "_source": {
                "term": row["term"],
                "event_name": row["event_name"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "source_urls": row["source_urls"],
                "semantic_text": semantic_text,
                "semantic_vector": embedding,
            },
        })

    create_index(index_name)

    try:
        helpers.bulk(es, actions)
        n = len(actions)
        print(f"Indexed {n} documents with semantic_text and semantic_vector to {index_name}.")
    except Exception as e:
        print(f"Failed to index data: {str(e)}")