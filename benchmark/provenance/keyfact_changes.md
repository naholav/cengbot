# Key-fact changes for rubric det-v3

Made on 2026-11-12 together with the `det-v3` grader (`benchmark/grader.py`). Questions, reference answers,
links, evidence and categories are unchanged; only `key_facts` were edited and some facts moved to the new,
unscored `extra_facts` list (context that is true and on the source page, but not asked by the question).

Summary: 45 changes in 35 items: 19 facts reworded, 24 facts moved to `extra_facts`,
2 split-off facts added to `extra_facts`. 25 items now have `extra_facts`.

Rules followed:

* **Not asked → `extra_facts`.** A scored fact must answer part of the question or be a condition that changes
  the answer for the asker. Background, other rules and consequences beyond the question were moved.
  Items keep 2-4 scored facts; when moving would leave one, the closest fact stays scored (list below).
  The one exception is `tr-047`, named in the det-v3 review: its question asks a single yes/no thing.
* **Yes/no questions carry the reference answer's yes/no** ("Hayır, ...", "No, ...", "Evet, ...") in the
  fact that answers them, so the polarity rule can reject the opposite answer.
* **Incidental negation removed.** A negation inside a condition or relative clause ("aşmamak koşuluyla",
  "kaybetmemiş olmak", "could not finish", "cannot be changed" for an exception) was restated positively
  with the same meaning, so a correct answer that phrases it positively is not rejected by the polarity rule.
* Every fact is still supported by the item's source page: the reworded facts were checked against the
  extracted page text, and `build_questions.py verify` on the updated candidates (all evidence quotes found)
  rebuilds this `questions.jsonl` byte for byte. Every reference answer still scores 100.

## Changes

| item | change | before | after | reason |
|---|---|---|---|---|
| tr-003 | fact 3 -> extra_facts | İkisinin geçerlilik süresi aynıdır (30 Eylül 2028) | extra_facts: İkisinin geçerlilik süresi aynıdır (30 Eylül 2028) | Not asked: the question asks why the EUR-ACE label comes automatically, not until when it is valid. |
| tr-004 | fact 3 -> extra_facts | Başvuru dönemine kadarki tüm dersleri başarmış olmak | extra_facts: Başvuru dönemine kadarki tüm dersleri başarmış olmak | Not asked: the question asks only for the GNO and class-ranking conditions. |
| tr-007 | fact 4 -> extra_facts | İki yıllık, Bilgisayar Yüksek Mühendisi unvanı | extra_facts: İki yıllık, Bilgisayar Yüksek Mühendisi unvanı | Not asked: the question asks for the number of courses and seminars (programme length and title are background). |
| tr-008 | fact 2 edited | Toplam 40 AKTS'yi aşmamak koşuluyla | En fazla 40 AKTS | Same limit without the conditional negation ("aşmamak koşuluyla"): under det-v3 a negated fact needs a negated answer, so the equivalent "en fazla 40 AKTS" would have been rejected. Source: "40 AKTS'yi aşmamak koşuluyla". |
| tr-008 | fact 3 -> extra_facts | Düşük alınan not dikkate alınmaz | extra_facts: Düşük alınan not dikkate alınmaz | Not asked: the question asks which courses qualify, not how the new grade is used. |
| tr-009 | fact 3 -> extra_facts | Stajlar ayrı dönemlerde yapılır | extra_facts: Stajlar ayrı dönemlerde yapılır | Not asked: the question asks for the number of internships and their length. |
| tr-013 | fact 1 edited | ÇÜ'de yaz öğretimi açılmaz | Hayır, ÇÜ'de yaz öğretimi açılmaz | Yes/no question: the fact now carries the reference answer's "Hayır", so "Evet ... açılır" scores 0 on it. |
| tr-018 | fact 2 edited | Başka dil sınavı belgesi kabul edilmez | Hayır, başka dil sınavı belgesi kabul edilmez | Yes/no question: the fact carries the reference answer's "Hayır". |
| tr-018 | fact 3 -> extra_facts | Sınavlar B2 seviyesindedir | extra_facts: Sınavlar B2 seviyesindedir | Not asked: the question asks whether TOEFL/IELTS certificates are accepted, not the exam level. |
| tr-026 | fact 2 -> extra_facts | Hiç almadığın dersleri de alabilirsin | extra_facts: Hiç almadığın dersleri de alabilirsin | Not asked: the question is about a course with FF or DC; never-taken courses are a different case (the other-grades fact is kept so the item keeps two scored facts). |
| tr-028 | fact 3 -> extra_facts | 3 yerel kredi | extra_facts: 3 yerel kredi | Not asked: the question asks for the semester and the ECTS, not the local credits. |
| tr-030 | fact 1 edited | 2 yarıyıldan fazla kaybetmemiş olmak | En fazla 2 yarıyıl kaybetmiş olmak | Same rule stated without the negated verb ("kaybetmemiş"), so the equivalent "en fazla 2 yarıyıl" is not rejected by the polarity rule. Source: "2 yarıyıldan fazla yarıyıl kaybetmemiş olması gerekmektedir". |
| tr-033 | fact 2 -> extra_facts | Konu danışmanla birlikte belirlenir | extra_facts: Konu danışmanla birlikte belirlenir | Not asked: the question asks when (deadlines), not with whom the topic is chosen. |
| tr-034 | fact 1 edited | Dijital rapor intihal taramasından geçirilir | Evet, dijital rapor intihal taramasından geçirilir | Yes/no question: the fact carries the reference answer's "Evet". |
| tr-036 | fact 1 edited | Öğretim dili kayıtlı olunan programla aynı olmalı | Evet, öğretim dili kayıtlı olunan programla aynı olmalı | Yes/no question: the fact carries the reference answer's "Evet". |
| tr-043 | fact 1 edited | Bilgisayar mühendisi çalışması şart değil, öncelik taşır | Bilgisayar mühendisi çalışması şart değil | Split: the question asks whether it is required ("şart değil"); "öncelik taşır" moves to extra_facts, so a correct "şart değil" answer is not failed on a synonym of "öncelik". |
| tr-043 | extra fact added | - | extra_facts: Kurumda bilgisayar mühendisi çalışması öncelik taşır | Split off scored fact 1 (see above). Source: "kurumda Bilgisayar Mühendisi çalışması öncelik taşımaktadır". |
| tr-043 | fact 2 -> extra_facts | Orta veya büyük ölçekli kurum tavsiye edilir | extra_facts: Orta veya büyük ölçekli kurum tavsiye edilir | Not a condition: the source only recommends a medium or large institution; the question asks which conditions it must meet. |
| tr-044 | fact 2 edited | 0322 338 60 84 /2780 veya 0322 338 61 50 | 0322 338 60 84 (/2780 veya 0322 338 61 50) | One phone number answers the question: the extension and the second number are parenthesised alternatives (optional under det-v3). |
| tr-044 | fact 3 -> extra_facts | Rektörlük Binası, Balcalı | extra_facts: Rektörlük Binası, Balcalı | Not asked: the question asks for e-mail and phone, not the address. |
| tr-046 | fact 1 edited | CEN316, 6. yarıyıl | 6. yarıyıl | The question names the course; its code is not asked (moved to extra_facts). |
| tr-046 | extra fact added | - | extra_facts: Ders kodu CEN316 | Split off scored fact 1 (see above). |
| tr-047 | fact 1 edited | Devam koşulu aranmaz | Hayır, devam koşulu aranmaz | Yes/no question: the fact carries the reference answer's "Hayır". |
| tr-047 | fact 2 -> extra_facts | Ara sınav ve yarıyıl sonu sınav hakları saklıdır | extra_facts: Ara sınav ve yarıyıl sonu sınav hakları saklıdır | Not asked (named in the det-v3 review): the question asks only whether attendance is required. The only item left with one scored fact, because the question asks exactly one thing. |
| en-002 | fact 1 edited | Students cannot choose their own teams | No, students cannot choose their own teams | Yes/no question: the fact carries the reference answer's "No". |
| en-002 | fact 2 -> extra_facts | Teams are announced before the information meeting | extra_facts: Teams are announced before the information meeting | Not asked: the question asks whether teams can be chosen and topics changed, not when teams are announced. |
| en-006 | fact 3 -> extra_facts | Distance-learning courses use 20% / 80% | extra_facts: Distance-learning courses use 20% / 80% | Not asked: the question is about face-to-face courses. |
| en-007 | fact 3 -> extra_facts | Plus the qualification exam and the thesis | extra_facts: Plus the qualification exam and the thesis | Not asked: the question asks for the number of elective courses (the seminar fact is kept so the item keeps two scored facts). |
| en-008 | fact 3 -> extra_facts | Counted whether or not you registered each semester | extra_facts: Counted whether or not you registered each semester | Not asked: the question asks for the maximum duration. |
| en-010 | fact 3 -> extra_facts | Used for programming/data structures lab hours | extra_facts: Used for programming/data structures lab hours | Not asked: the question asks for the number of PCs and the operating systems. |
| en-013 | fact 4 -> extra_facts | The frozen period does not count toward the maximum study period | extra_facts: The frozen period does not count toward the maximum study period | Not asked: the question asks for the limits and the application period. |
| en-015 | fact 3 -> extra_facts | Compulsory course (3+0 hours) | extra_facts: Compulsory course (3+0 hours) | Not asked: the question asks for the ECTS and the semester. |
| en-019 | fact 3 -> extra_facts | Nor the make-up (bütünleme) exam | extra_facts: A student with DZ cannot take the make-up (bütünleme) exam either | Not asked: the question asks about the final exam (reworded so it reads on its own). |
| en-021 | fact 1 edited | Every Graduation Thesis student must complete it | Yes, every Graduation Thesis student must complete it | Yes/no question: the fact carries the reference answer's "Yes" ("optional, not mandatory" now fails it). |
| en-021 | fact 3 -> extra_facts | Not completing it means failing the course | extra_facts: Not completing it means failing the course | Not asked: the question asks whether it is mandatory and its weight, not the consequence of skipping it. |
| en-023 | fact 2 edited | Lower-semester courses not yet taken cannot be changed | Except lower-semester courses not yet taken | Stated as an exception to the "Yes", like the reference answer ("except ..."); the old "cannot be changed" made a correct "except ..." answer fail the polarity rule. |
| en-023 | fact 3 edited | Courses that must be repeated cannot be changed | Except courses that must be repeated | Same as fact 2: the reference answer says "except ... courses you must repeat", with no negation word. |
| en-030 | fact 3 -> extra_facts | The Learning Agreement during-mobility part must be signed | extra_facts: The Learning Agreement during-mobility part must be signed | Not asked: the question asks until when a change can be requested. |
| en-031 | fact 3 -> extra_facts | Only a single make-up exam per course | extra_facts: Only a single make-up exam per course | Not asked: the question asks for the application deadline. |
| en-033 | fact 2 edited | Also opened in fall for students who could not finish in normal time | Also opened in the fall semester for students beyond the normal study time | The negation was inside the relative clause ("could not finish"), so an answer saying "there is no fall section" passed the polarity rule; stated positively. Source: "normal süresinde tamamlayamayan öğrenciler için ... güz yarıyılında da ... açılmaktadır". |
| en-037 | fact 3 -> extra_facts | The Summer School Commission decides on content and hours | extra_facts: The Summer School Commission decides on content and hours | Not asked: the question asks for the ECTS rule and whether the names must match. |
| en-040 | fact 1 edited | A minor gives a certificate, not a separate degree | No, a minor gives a certificate, not a separate degree | Yes/no question: the fact carries the reference answer's "No". |
| en-041 | fact 2 edited | 3.50-4.00: high honor student | 3.50 or higher (up to 4.00): high honor student | 4.00 is the top of the scale, so "3.50 or above" states the same range; the upper bound is a parenthesised (optional) number now. |
| en-042 | fact 1 edited | A literature study is not accepted for multidisciplinary work | No, a literature study is not accepted for the multidisciplinary part | Yes/no question: the fact carries the reference answer's "No"; "part" instead of "work" (the question's word), so a correct answer is not failed on "work". |
| en-044 | fact 1 edited | Failed summer school grades are not counted | No, failed summer school grades are not counted | Yes/no question: the fact carries the reference answer's "No". |

## Not-asked facts kept scored (two-fact minimum)

These items ask one thing; their second fact is related but not asked, and stays scored only so the item
keeps two scored facts. A complete, correct answer that skips it scores 65 in deterministic mode.

* `tr-022` fact 2: Değerlendirmede staj danışmanıyla yüz yüze görüşme yapılır
* `tr-026` fact 2: UB, DZ, FG ve DD notlu dersler de kapsamdadır
* `tr-031` fact 2: FG'nin katsayısı 0.00'dır
* `tr-038` fact 2: Ortak zorunlu üniversite dersleri 8 AKTS
* `en-007` fact 2: Plus 1 seminar
* `en-017` fact 2: Apply between the beginning of the 3rd and the 6th semester
* `en-020` fact 2: Below 70 cannot take the oral exam
* `en-035` fact 2: Registration must be renewed each semester within the calendar period
* `en-042` fact 2: The multidisciplinary project involves students from at least two different departments
* `en-044` fact 2: Passed grades are converted by the Summer School Commission

## Keeping the builder in sync

`provenance/candidates.jsonl` still holds the det-v2 key facts. Before re-running `build_questions.py verify`,
copy the facts above into it (same `key_facts` / `extra_facts`, mapped through `build_report.json` `id_map`),
otherwise the rebuild restores the det-v2 facts.
