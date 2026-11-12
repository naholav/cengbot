# Benchmark run `v1.2-final-code_20261112-185030`

- Model: **meta-llama/Llama-3.2-3B + final-best-model-v1.2** (`models/final-best-model-v1.2`)
- Questions: 100 of 100 (`benchmark/questions.jsonl`, sha256 b85645185fe8)
- Scoring: **deterministic** (LLM judge off: disabled with --no-llm); deterministic rubric `det-v3`
- Git: `d0fdc307204a` (working tree dirty)
- Generated: 2026-11-12T15:50:30+00:00, graded: 2026-11-12T15:55:47+00:00

## Scores

| metric | value |
|---|---|
| overall (0-100) | 30.7 |
| EN | 30.2 |
| TR | 31.2 |
| key-fact coverage | 20.6% |
| link accuracy (correct link present) | 34.0% |
| wrong official link rate | 55.0% |
| hallucinated URL rate | 0.0% |
| hallucination rate | 0.0% |
| language match | 100.0% |
| length sanity | 100.0% |

## By category

| category | score |
|---|---|
| academic_calendar | 26.7 |
| courses_credits | 38.0 |
| department_general | 42.0 |
| double_major_minor | 18.8 |
| erasmus_exchange | 25.7 |
| exams | 29.1 |
| graduate_programs | 38.3 |
| graduation | 30.6 |
| internship | 39.5 |
| labs_research | 39.2 |
| registration | 15.8 |
| staff_contacts | 52.0 |
| student_affairs | 12.5 |

| difficulty | score |
|---|---|
| easy | 46.5 |
| hard | 20.3 |
| medium | 28.3 |

Generation: mean latency 3.2 s, blocked 0, errors 0, seed 1234.

## 10 worst items

### en-002 (courses_credits, medium) score 10.0

**Q:** In CEN317 Engineering Design, can we pick our own team members or change our project topic?

**Answer:** Yes, you can definitely choose your teammates! For changing the project topic, contact the academician who teaches the lesson first. Regarding selecting teammates, just make sure everyone is comfortable working together before making any formal assignments. It's totally your choice how you want to organize your group dynamic. Find out more here: https://ebs.cu.edu.tr/Program/DersPlan/555/2024.

**Reference:** No. Teams are formed and announced before the information meeting and students cannot choose their own teams; topics are given at the information meeting and cannot be changed. https://bmb.cu.edu.tr/storage/engineering%20design/CEN317_Engineering_desing_rubicle__turkish_final.pdf

Flags: no correct link, wrong link

### en-003 (department_general, easy) score 10.0

**Q:** Is the Computer Engineering programme accredited, and until when is the accreditation valid?

**Answer:** Yes, the Computer Engineering programme is accredited by Mühendislik ve Theoretical Foundations (20%), Software (25%) and Computer Hardware (25%) departments with a total of 75 ECTS credits. The 4-year program is currently accepted by YÖK AYŞİRLA and has an accreditation validity of 5 years from its commencement date. For detailed information, you can follow this link containing the official regulations: https://yokatlas.yok.gov.tr/lisans.php?y=102910701. This resource will give you comprehensive details about the program's academic structure, faculty credentials, and institutional oversight.

**Reference:** Yes. It is accredited by MÜDEK and also holds the EUR-ACE label; both are valid until 30 September 2028. https://bmb.cu.edu.tr/cu/kalite/mudek-ve-eur-ace-akreditasyonlarimiz-hakkinda

Flags: no correct link, wrong link

### en-006 (exams, medium) score 10.0

**Q:** For a course taught face to face, what weights do the in-term grade and the final exam have in the raw grade?

**Answer:** The lesson content of the course to be taken with substitute teaching is to be examined by the course coordinator and equivalent grades are determined. In conclusion, it is not possible to match courses with different subjects or educational approaches. For detailed information, you can contact the course coordinator of the course you want to replace. For contact information you can follow this link: https://ebs.cu.edu.tr/Program/DersPlan/555/2024 If the course is elective, the commission decision is made exclusively for that particular semester.

**Reference:** The raw grade is 40% of the in-term grade plus 60% of the final exam (for courses taught by distance learning it is 20% and 80%); the letter grade is then given by relative grading. https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf

Flags: no correct link, wrong link

### en-008 (graduation, medium) score 10.0

**Q:** What is the maximum time allowed to finish a four-year bachelor's programme such as Computer Engineering?

**Answer:** A four-year bachelor's programme must be completed within 240 ECTS credits (120 ECTS per semester x 2 semesters) with no more than 3 failed courses or 5 total failed lessons. The regulation can be followed at https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=18486&MevzuatTur=8&MevzuatTertip=5#:~:text=E%C4%9Fitim%2D%C3%B6%C4%9Fretim%20s%C3%BCresi%206%20y%C4%B1ll%C4%B1k,mezun%20olma%20hakk%C4%B1n%C4%B1%20kazanm%C4%B1%C5%9F%20say%C4%B1l%C4%B1r. For detailed technical specifications, please refer to the official regulation document available here.

**Reference:** Seven years: four-year bachelor's programmes must be completed within at most seven years, excluding a one-year language preparatory class, counted whether or not you registered each semester. https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf

Flags: no correct link, wrong link

### en-011 (registration, medium) score 10.0

**Q:** What are the three stages of semester registration renewal at Çukurova University?

**Answer:** The system sends you an email when your course enrollment is incomplete. You must reply to that email within 7 days before the next semester starts. The system then automatically enrolls you in all your previous courses. This process covers the main steps of semester registration renewal at Çukurova University - check your emails and respond promptly. For technical issues or more detailed information about specific cases, contact student affairs directly: https://oidb.cu.edu.tr/%C4%B1letisim/

**Reference:** Registration renewal has three stages: paying the contribution fee or tuition, registering online during the course registration period, and advisor approval in the first week of the semester. http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/KayitYenilemeYeniSenato.pdf

Flags: no correct link, wrong link

### en-012 (staff_contacts, easy) score 10.0

**Q:** Who is listed as the International Relations Office general coordinator on the Erasmus study contact page, and what is the e-mail address?

**Answer:** The Head of Erasmus exchange programme commission is Assoc.Prof.Dr. Mustafa ORAL, with his official email address available at https://avesis.cu.edu.tr/eefirat. For technical questions or specific inquiries about your application, you should contact him directly via email. His complete contact details including office location and telephone number are also typically included in the departmental announcements section of the website.

**Reference:** Müge Ergin is listed as the International Relations Office general coordinator; e-mail outgoing.erasmus@cu.edu.tr. https://erasmus.cu.edu.tr/cu/erasmus-study/contact

Flags: no correct link, wrong link

### en-013 (student_affairs, hard) score 10.0

**Q:** How many semesters can I freeze my registration at once and in total, and when must I apply?

**Answer:** You can freeze your course selection in 2 different semesters maximum - the semester you're currently in plus the following one. As for application timing, you need to do this before making any new course registrations during that semester. After submitting your application, make sure you choose courses from both frozen and unfrozen semesters.

**Reference:** With a valid reason you can freeze your registration for at most two semesters at a time and at most four semesters in total during your studies; applications are made within four weeks of the start of the semester, and the frozen period does not count toward the maximum study period. https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf

Flags: no correct link

### en-017 (double_major_minor, easy) score 10.0

**Q:** What minimum GPA do I need to apply for a minor (yandal) programme?

**Answer:** To apply for a yandal program with a minor, you need at least 3.00 GPA along with the required 60 ECTS credits from your major's courses. For detailed eligibility requirements and application procedures, visit this link: https://bmb.cu.edu.tr/storage/belgeler/mufredat/19-20Yandal.pdf. This document contains all the specific criteria you'll need to know about. Good luck with your application!

**Reference:** Your GPA must be at least 2.50 at the time of application; you can apply between the beginning of the 3rd and the beginning of the 6th semester. http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/%C3%87ift%20Anadal%20Yandal%20Y%C3%B6nerge%2016.04.2024%20son%20%C5%9Fekli.pdf

Flags: no correct link, wrong link

### en-023 (registration, medium) score 10.0

**Q:** After registering, can I still change or drop some of my courses?

**Answer:** Yes, you can modify your course registration through the student portal. Check this link for details: https://bmb.cu.edu.tr/cu/student/forms-and-regulations - it has all the specific requirements and deadlines you need to know about. The regulations section will help you understand what's allowed and when changes need to be completed. Good luck with your enrollment!

**Reference:** Yes, in the first week of the semester you can change or drop courses of your highest registered semester, except lower-semester courses you haven't taken yet and courses you must repeat. https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf

Flags: no correct link, wrong link

### en-025 (student_affairs, medium) score 10.0

**Q:** How many courses can I take at another university's summer school in one summer, and in total during my bachelor's?

**Answer:** You can take maximum 3 courses at the summer school you decide on, but there's no limit on the number of hours or credits for those courses. The only restriction is that they must be different courses from each other, not the same course multiple times. For your bachelor's degree, this policy applies to all summer programs including our own department's summer school and any external universities. The only thing that changes is the credit transfer process - once you complete your course registration at the external institution, our registrar will determine equivalency based on your program requirements. You can find detailed information about the summer school course list by following this link: https://bmb.cu.edu.tr/storage/belgeler/mufredat/20-21_dersler.pdf.

**Reference:** At most three courses in one summer term and at most six courses in total during your bachelor's studies. https://bmb.cu.edu.tr/storage/belgeler/%C3%87.%C3%9C%20Yaz%20%C3%96%C4%9Fretimi%20Y%C3%B6nergesi.pdf

Flags: no correct link, wrong link
