# Benchmark sources

Official pages fetched on 2026-11-11 (UTC times below) with an ordinary browser user agent at <= 1 request/s,
converted to text (`benchmark/build_questions.py extract`) and used to write and verify `questions.jsonl`.
Pages of `bmb.cu.edu.tr` keep the language in the session, so the department pages were fetched twice:
once in Turkish and once after `https://bmb.cu.edu.tr/lang/en` (variant `en`). Raw downloads are not committed
(re-create them with `build_questions.py fetch`, see README).

## Pages behind the 100 benchmark items

| source URL | items | candidates | fetched (UTC) |
|---|---|---|---|
| https://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/Onlisans_Lisans_Sinav_Yonetmelik.pdf | 18 | 29 | 2026-11-11 13:59:49 |
| https://bmb.cu.edu.tr/storage/belgeler/formlar/Staj_Yonergesi.pdf | 9 | 14 | 2026-11-11 13:56:50 |
| https://www.cu.edu.tr/media/212daahw/2026genel.pdf | 9 | 11 | 2026-11-11 13:59:28 |
| http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/%C3%87ift%20Anadal%20Yandal%20Y%C3%B6nerge%2016.04.2024%20son%20%C5%9Fekli.pdf | 8 | 11 | 2026-11-11 13:59:49 |
| https://bmb.cu.edu.tr/storage/belgeler/%C3%87.%C3%9C%20Yaz%20%C3%96%C4%9Fretimi%20Y%C3%B6nergesi.pdf | 5 | 7 | 2026-11-11 13:56:51 |
| https://ebs.cu.edu.tr/En/Program/CourseCatalog/555/2026 | 5 | 9 | 2026-11-11 14:03:05 |
| https://bmb.cu.edu.tr/cu/education/doktora/doktora-yeterlik | 4 | 4 | 2026-11-11 13:56:48, 13:57:19 (en) |
| https://erasmus.cu.edu.tr/cu/erasmus-study/terms-of-application | 4 | 7 | 2026-11-11 13:53:46 |
| https://bmb.cu.edu.tr/cu/education/lisans/bitirme-projesi | 3 | 3 | 2026-11-11 13:53:38, 13:57:48 (en) |
| https://bmb.cu.edu.tr/cu/education/lisans/staj | 3 | 3 | 2026-11-11 13:53:37, 13:57:47 (en) |
| https://bmb.cu.edu.tr/storage/engineering%20design/CEN317_Engineering_desing_rubicle__turkish_final.pdf | 3 | 3 | 2026-11-11 13:56:57 |
| https://ebs.cu.edu.tr/Program/DersPlan/555/2026 | 3 | 6 | 2026-11-11 13:57:02 |
| http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/KayitYenilemeYeniSenato.pdf | 2 | 2 | 2026-11-11 13:59:50 |
| https://bmb.cu.edu.tr/cu/bolumumuz/about | 2 | 3 | 2026-11-11 13:53:12, 13:57:21 (en) |
| https://bmb.cu.edu.tr/cu/kalite/mudek-ve-eur-ace-akreditasyonlarimiz-hakkinda | 2 | 2 | 2026-11-11 13:53:33, 13:57:43 (en) |
| https://bmb.cu.edu.tr/cu/personel/akademik-personel | 2 | 3 | 2026-11-11 13:53:17, 13:57:27 (en) |
| https://bmb.cu.edu.tr/cu/research/laboratuvarlar | 2 | 4 | 2026-11-11 13:53:30, 13:57:40 (en) |
| https://bmb.cu.edu.tr/cu/student/bitirme-tezi | 2 | 2 | 2026-11-11 13:53:26, 13:57:36 (en) |
| https://bmb.cu.edu.tr/storage/belgeler/erasmus/Erasmus_Bilgiler.pdf | 2 | 2 | 2026-11-11 13:56:53 |
| https://bmb.cu.edu.tr/cu/bolumumuz/administrative-structure | 1 | 3 | 2026-11-11 13:53:14, 13:57:23 (en) |
| https://bmb.cu.edu.tr/cu/education/doktora | 1 | 2 | 2026-11-11 13:53:21, 13:57:31 (en) |
| https://bmb.cu.edu.tr/cu/education/lisans | 1 | 1 | 2026-11-11 13:53:19, 13:57:29 (en) |
| https://bmb.cu.edu.tr/cu/education/yuksek-lisans | 1 | 2 | 2026-11-11 13:53:20, 13:57:30 (en) |
| https://bmb.cu.edu.tr/cu/research/calisma-gruplari | 1 | 1 | 2026-11-11 13:53:31, 13:57:41 (en) |
| https://bmb.cu.edu.tr/cu/student/erasmus | 1 | 3 | 2026-11-11 13:53:24, 13:57:33 (en) |
| https://bmb.cu.edu.tr/storage/B%C3%B6l%C3%BCmlerden%20g%C3%B6r%C3%BC%C5%9Flere%20g%C3%B6re%20d%C3%BCzenlenen%20Bitirme_Odevi_Ilkeleri-12.01.2022.pdf | 1 | 2 | 2026-11-11 13:56:56 |
| https://bmb.cu.edu.tr/storage/belgeler/yaz_okulu/yazokulu_sunumG%C3%BCncel1.pdf | 1 | 1 | 2026-11-11 13:56:52 |
| https://ebs.cu.edu.tr/En/Program/Information/555/2026 | 1 | 1 | 2026-11-11 14:03:05 |
| https://ebs.cu.edu.tr/Program/GenelBilgi/555/2026 | 1 | 2 | 2026-11-11 13:57:01 |
| https://erasmus.cu.edu.tr/cu/erasmus-study/contact | 1 | 1 | 2026-11-11 13:59:53 |
| https://oidb.cu.edu.tr/%C4%B1letisim/ | 1 | 1 | 2026-11-11 13:53:47 |

## Other pages fetched (context, link discovery, reserve candidates)

| URL | status | fetched (UTC) |
|---|---|---|
| http://genelsekreterlik.cu.edu.tr/storage/MEVZUAT/%C3%96%C4%9ERENC%C4%B0/%C3%87.%C3%9C%20Yaz%20%C3%96%C4%9Fretimi%20Y%C3%B6nergesi.pdf | 200 | 2026-11-11T13:59:51+00:00 |
| https://bmb.cu.edu.tr/cu/Programlar | 200 | 2026-11-11T13:53:29+00:00 |
| https://bmb.cu.edu.tr/cu/Programlar (variant en) | 200 | 2026-11-11T13:57:39+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/committees | 200 | 2026-11-11T13:53:15+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/committees (variant en) | 200 | 2026-11-11T13:57:25+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/oryantasyon | 200 | 2026-11-11T13:53:16+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/oryantasyon (variant en) | 200 | 2026-11-11T13:57:26+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/ozgorev-ve-ozgorus | 200 | 2026-11-11T13:53:13+00:00 |
| https://bmb.cu.edu.tr/cu/bolumumuz/ozgorev-ve-ozgorus (variant en) | 200 | 2026-11-11T13:57:22+00:00 |
| https://bmb.cu.edu.tr/cu/education/doktora/doktora-yeterlilik | 404 | 2026-11-11T13:53:40+00:00 |
| https://bmb.cu.edu.tr/cu/education/lisans/ders-icerikleri | 200 | 2026-11-11T13:56:49+00:00 |
| https://bmb.cu.edu.tr/cu/education/lisans/ders-icerikleri (variant en) | 200 | 2026-11-11T13:57:20+00:00 |
| https://bmb.cu.edu.tr/cu/education/lisans/ders-listesi | 200 | 2026-11-11T13:56:48+00:00 |
| https://bmb.cu.edu.tr/cu/education/lisans/ders-listesi (variant en) | 200 | 2026-11-11T13:57:19+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudek | 200 | 2026-11-11T13:53:35+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudek (variant en) | 200 | 2026-11-11T13:57:44+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudek-hakkinda | 200 | 2026-11-11T13:53:32+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudek-hakkinda (variant en) | 200 | 2026-11-11T13:57:42+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudekk | 200 | 2026-11-11T13:53:36+00:00 |
| https://bmb.cu.edu.tr/cu/kalite/mudekk (variant en) | 200 | 2026-11-11T13:57:46+00:00 |
| https://bmb.cu.edu.tr/cu/personel/staff | 200 | 2026-11-11T13:53:18+00:00 |
| https://bmb.cu.edu.tr/cu/personel/staff (variant en) | 200 | 2026-11-11T13:57:28+00:00 |
| https://bmb.cu.edu.tr/cu/student/cen317-engineering-design | 200 | 2026-11-11T13:53:27+00:00 |
| https://bmb.cu.edu.tr/cu/student/cen317-engineering-design (variant en) | 200 | 2026-11-11T13:57:37+00:00 |
| https://bmb.cu.edu.tr/cu/student/e-services | 200 | 2026-11-11T13:53:28+00:00 |
| https://bmb.cu.edu.tr/cu/student/e-services (variant en) | 200 | 2026-11-11T13:57:38+00:00 |
| https://bmb.cu.edu.tr/cu/student/forms-and-regulations | 200 | 2026-11-11T13:53:22+00:00 |
| https://bmb.cu.edu.tr/cu/student/forms-and-regulations (variant en) | 200 | 2026-11-11T13:57:32+00:00 |
| https://bmb.cu.edu.tr/cu/student/internship | 404 | 2026-11-11T13:53:39+00:00 |
| https://bmb.cu.edu.tr/cu/student/yaz-okulu | 200 | 2026-11-11T13:53:25+00:00 |
| https://bmb.cu.edu.tr/cu/student/yaz-okulu (variant en) | 200 | 2026-11-11T13:57:34+00:00 |
| https://bmb.cu.edu.tr/storage/belgeler/mufredat/19-20CiftAnadal.pdf | 200 | 2026-11-11T13:56:54+00:00 |
| https://bmb.cu.edu.tr/storage/belgeler/mufredat/19-20Yandal.pdf | 200 | 2026-11-11T13:56:55+00:00 |
| https://bmb.cu.edu.tr/storage/belgeler/oryantasyon/egitimOgretimYonetmelik.pdf | 200 | 2026-11-11T13:56:59+00:00 |
| https://ceng.cu.edu.tr/ | 200 | 2026-11-11T13:53:41+00:00 |
| https://ebs.cu.edu.tr/En/Program/CourseCatalog/555 | 200 | 2026-11-11T13:57:04+00:00 |
| https://ebs.cu.edu.tr/En/Program/Information/555 | 200 | 2026-11-11T13:57:03+00:00 |
| https://ebs.cu.edu.tr/Program/DersPlan/555 | 200 | 2026-11-11T13:53:43+00:00 |
| https://ebs.cu.edu.tr/Program/DersPlan/555/2025 | 200 | 2026-11-11T14:03:06+00:00 |
| https://ebs.cu.edu.tr/Program/GenelBilgi/555 | 200 | 2026-11-11T13:53:42+00:00 |
| https://ebs.cu.edu.tr/Program/TemelOgrenmeCiktilari/555/2026 | 200 | 2026-11-11T13:57:05+00:00 |
| https://erasmus.cu.edu.tr/cu/erasmus-placement/erasmus-placement-mobility | 200 | 2026-11-11T13:59:55+00:00 |
| https://erasmus.cu.edu.tr/cu/erasmus-study/dokumanlar-ve-formlar | 200 | 2026-11-11T13:57:06+00:00 |
| https://genelsekreterlik.cu.edu.tr/cu/mevzuat/universitemiz-mevzuati/ogrenci-mevzuati | 200 | 2026-11-11T13:59:29+00:00 |
| https://ogrenciisleri.cu.edu.tr/cu/iletisim | 404 | 2026-11-11T13:53:48+00:00 |
| https://oidb.cu.edu.tr | 200 | 2026-11-11T13:57:08+00:00 |
| https://oidb.cu.edu.tr/basvurular/cift-anadal-yandal/ | 200 | 2026-11-11T13:59:31+00:00 |
| https://oidb.cu.edu.tr/basvurular/cift-anadal-yandal/cift-anadal/ | 200 | 2026-11-11T13:59:32+00:00 |
| https://oidb.cu.edu.tr/basvurular/cift-anadal-yandal/yan-dal/ | 200 | 2026-11-11T13:59:33+00:00 |
| https://oidb.cu.edu.tr/basvurular/yaz-okulu/ | 200 | 2026-11-11T13:59:52+00:00 |
| https://oidb.cu.edu.tr/mevzuat/ | 200 | 2026-11-11T13:59:34+00:00 |
| https://www.cu.edu.tr/cu/students/links/academic-calendar | 200 | 2026-11-11T13:53:44+00:00 |
| https://www.cu.edu.tr/cu/students/links/yonetmelik-ve-yonergeler | 200 | 2026-11-11T13:57:00+00:00 |
| https://www.cu.edu.tr/eng/students/academic-calendar/ | 200 | 2026-11-11T13:59:28+00:00 |

Notes:

- `https://bmb.cu.edu.tr/cu/student/internship` and `https://bmb.cu.edu.tr/cu/education/doktora/doktora-yeterlilik`,
  which appear in the training data, answered HTTP 404 on 2026-11-11 (the current pages are
  `/cu/education/lisans/staj` and `/cu/education/doktora/doktora-yeterlik`).
- The EBS answers `HEAD` requests with 404; link checks therefore use a streamed `GET`.
