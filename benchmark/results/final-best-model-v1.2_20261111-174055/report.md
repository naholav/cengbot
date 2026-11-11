# Benchmark run `final-best-model-v1.2_20261111-174055`

- Model: **meta-llama/Llama-3.2-3B + final-best-model-v1.2** (`models/final-best-model-v1.2`)
- Questions: 100 of 100 (`benchmark/questions.jsonl`, sha256 41a1a2a59b2f)
- Scoring: **deterministic** (LLM judge off: authentication failed (AuthenticationError, HTTP 401))
- Git: `b5ddce8c64da`
- Generated: 2026-11-11T14:40:55+00:00, graded: 2026-11-11T14:46:46+00:00

## Scores

| metric | value |
|---|---|
| overall (0-100) | 30.2 |
| EN | 30.2 |
| TR | 30.2 |
| key-fact coverage | 20.3% |
| link accuracy (correct link present) | 32.0% |
| wrong official link rate | 56.0% |
| hallucinated URL rate | 0.0% |
| hallucination rate | 0.0% |
| language match | 100.0% |
| length sanity | 100.0% |

## By category

| category | score |
|---|---|
| academic_calendar | 31.7 |
| courses_credits | 26.8 |
| department_general | 42.0 |
| double_major_minor | 18.8 |
| erasmus_exchange | 24.3 |
| exams | 21.7 |
| graduate_programs | 43.2 |
| graduation | 31.3 |
| internship | 40.8 |
| labs_research | 44.7 |
| registration | 10.0 |
| staff_contacts | 59.0 |
| student_affairs | 17.6 |

| difficulty | score |
|---|---|
| easy | 37.2 |
| hard | 23.2 |
| medium | 30.0 |

Generation: mean latency 3.5 s, blocked 0, errors 0, seed 1234.

## 10 worst items

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

**Answer:** The Head of Erasmus exchange programme commission is Assoc.Prof.Dr. Mustafa ORAL, with his official email address available at https://avesis.cu.edu.tr/eefirat. You can directly reach out to him for your Erasmus-related questions. Note that he serves in a supervisory capacity rather than providing direct student services.

**Reference:** Müge Ergin is listed as the International Relations Office general coordinator; e-mail outgoing.erasmus@cu.edu.tr. https://erasmus.cu.edu.tr/cu/erasmus-study/contact

Flags: no correct link, wrong link

### en-017 (double_major_minor, easy) score 10.0

**Q:** What minimum GPA do I need to apply for a minor (yandal) programme?

**Answer:** To apply for a yandal program with a minor, you need at least 3.00 GPA along with the required 60 ECTS credits from your major's courses. For detailed eligibility requirements and application procedures, visit this link: https://bmb.cu.edu.tr/storage/belgeler/mufredat/19-20Yandal.pdf. This document contains all the specific criteria you'll need to know about. Good luck with your application!

**Reference:** Your GPA must be at least 2.50 at the time of application; you can apply between the beginning of the 3rd and the beginning of the 6th semester. http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/%C3%87ift%20Anadal%20Yandal%20Y%C3%B6nerge%2016.04.2024%20son%20%C5%9Fekli.pdf

Flags: no correct link, wrong link

### en-018 (erasmus_exchange, medium) score 10.0

**Q:** How many months in total can a bachelor's student spend on Erasmus study and traineeship mobility?

**Answer:** A bachelor's student can allocate 3 months maximum between Erasmus study and traineeship mobility, with the condition that the duration of each component must not exceed 20 working days. For detailed information about the application terms and procedures, please refer to this comprehensive guide: https://erasmus.cu.edu.tr/cu/erasmus-placement/erasmus-placement-mobility/terms-of-application. This resource contains all the specific requirements and deadlines you'll need to know. It also outlines how these durations work within the overall program timeline.

**Reference:** Up to 12 months in total per study cycle (bachelor's is one cycle), combining study and traineeship mobility, with or without a grant. https://erasmus.cu.edu.tr/cu/erasmus-study/terms-of-application

Flags: no correct link, wrong link

### en-023 (registration, medium) score 10.0

**Q:** After registering, can I still change or drop some of my courses?

**Answer:** Yes, you can modify your course registration through the student portal. Check this link for details: https://bmb.cu.edu.tr/cu/student/forms-and-regulations - it has all the specific requirements and deadlines you need to know about. The regulations section will help you understand what's allowed and when changes need to be completed. Good luck with your enrollment!

**Reference:** Yes, in the first week of the semester you can change or drop courses of your highest registered semester, except lower-semester courses you haven't taken yet and courses you must repeat. https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf

Flags: no correct link, wrong link

### en-025 (student_affairs, medium) score 10.0

**Q:** How many courses can I take at another university's summer school in one summer, and in total during my bachelor's?

**Answer:** You can take maximum 3 courses at the summer school you decide on, but there's no limit on the number of hours or credits for those courses. The only restriction is that they must be different courses from each other, not the same course multiple times. For your bachelor's degree, this policy applies to all summer programs including our own department's summer school and any external universities. The only thing that changes is the credit transfer process - once you complete your course registration at the external institution, our registrar will determine equivalency based on your program requirements.

**Reference:** At most three courses in one summer term and at most six courses in total during your bachelor's studies. https://bmb.cu.edu.tr/storage/belgeler/%C3%87.%C3%9C%20Yaz%20%C3%96%C4%9Fretimi%20Y%C3%B6nergesi.pdf

Flags: no correct link

### en-027 (courses_credits, medium) score 10.0

**Q:** How many ECTS of compulsory and elective courses does the 2026-2027 Computer Engineering plan require?

**Answer:** The 2026-2027 Computer Engineering curriculum requires 176 ECTS of total credits with distribution between compulsory (69) and elective (107) courses. For detailed breakdowns and specific course codes, refer to this document: https://bmb.cu.edu.tr/storage/programlar/ders/Ders_program2024-2025_Bahar.pdf. This comprehensive guide includes all necessary details about credit distributions across different programs. Additionally, you can find information about the total number of elective courses, their corresponding code numbers, and semester-by-semester scheduling arrangements in this resource as well.

**Reference:** 240 ECTS in total: 182 ECTS compulsory (75.83%) and 58 ECTS elective (24.17%). https://ebs.cu.edu.tr/En/Program/CourseCatalog/555/2026

Flags: no correct link, wrong link
