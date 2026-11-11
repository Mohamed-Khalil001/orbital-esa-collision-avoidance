# ORBITAL-AI — هندسة تفادي الاصطدام الفضائي والذكاء الاصطناعي المداري

🚀 **التطبيق التفاعلي المباشر:** [افتح ORBITAL-AI على Streamlit](https://orbital-esa-collision-avoidance-3ewp7cecpuqb3kh6kmmbse.streamlit.app/)

مرجع هندسي من ستة فصول يأخذ القارئ من فهم بيئة الحطام في المدار الأرضي المنخفض، إلى إنتاج بيانات الاقتران CDM، ثم بناء نموذج إنذار مبكر، وأخيرًا تحويل التنبؤ إلى تحليل فيزيائي ومناورة تفادي قابلة للمراجعة.

> **فكرة الكتاب:** نرصد البيئة → ننتج المعرفة المدارية → نفهم سجل الاقتران → نصوغ قرار الخطر → نبني نموذجًا موثوقًا → نتحقق فيزيائيًا من خيارات المناورة.

## خريطة الفصول والمواد المصاحبة

| الفصل | الموضوع | قراءة الفصل | عروض PDF المصاحبة |
|---|---|---|---|
| 1 | مسرح العمليات: بيئة الحطام المداري | [Chapter 01](chapters/ch01_leo_debris_environment/chapter_01.md) | [Orbital Debris Crisis 1](chapters/ch01_leo_debris_environment/resources/Orbital_Debris_Crisis_1.pdf) · [Part 2](chapters/ch01_leo_debris_environment/resources/Orbital_Debris_Crisis_2.pdf) |
| 2 | عيون الأرض في السماء: الرصد وفهرسة الأجسام | [Chapter 02](chapters/ch02_space_surveillance_and_sp_catalog/chapter_02.md) | [Orbital Debris Tracking and CDM](chapters/ch02_space_surveillance_and_sp_catalog/resources/Orbital_Debris_Tracking_and_CDM.pdf) |
| 3 | قصة البيانات: CDM و103 أعمدة | [Chapter 03](chapters/ch03_cdm_data_lineage_103_columns/chapter_03.md) | [CDM Layer 2 Orbital Dynamics](chapters/ch03_cdm_data_lineage_103_columns/resources/CDM_Layer_2_Orbital_Dynamics.pdf) · [Architecture of Orbital Uncertainty](chapters/ch03_cdm_data_lineage_103_columns/resources/The_Architecture_of_Orbital_Uncertainty.pdf) · [CDM Mission Control Blueprint](chapters/ch03_cdm_data_lineage_103_columns/resources/CDM_Mission_Control_Blueprint.pdf) |
| 4 | من التحذير إلى المسألة: ESOC وصياغة ML | [Chapter 04](chapters/ch04_esoc_operations_and_ml_formulation/chapter_04.md) | [Machine Learning for Space Collision Avoidance](chapters/ch04_esoc_operations_and_ml_formulation/resources/Machine_Learning_for_Space_Collision_Avoidance.pdf) |
| 5 | بناء عقل القرار: Physics-Guided Two-Stage M6 | [Chapter 05](chapters/ch05_physics_guided_two_stage_m6/chapter_05.md) | [Orbital Safety and Collision Avoidance](chapters/ch05_physics_guided_two_stage_m6/resources/Orbital_Safety_and_Collision_Avoidance.pdf) |
| 6 | من التنبؤ إلى الفعل: المحاكاة وCAM | [Chapter 06](chapters/ch06_simulation_and_cam_decision_support/chapter_06.md) | [LEO Collision Warning Lifecycle](chapters/ch06_simulation_and_cam_decision_support/resources/LEO_Collision_Warning_Lifecycle.pdf) |

## ابدأ القراءة

1. اقرأ الفصول بالترتيب؛ فالمفاهيم والاختصارات تتراكم عمدًا من فصل إلى التالي.
2. استخدم روابط التنقل أعلى وأسفل كل فصل للانتقال بين أجزاء الكتاب.
3. الرسومات المستخدمة داخل الفصول موجودة في [مجلد Figures](Figures/)، والرسومات المستخرجة من المستندات الأصلية في [Figures/embedded](Figures/embedded/).
4. ملفات PDF روابط مباشرة داخل المستودع ويمكن فتحها أو تنزيلها من GitHub.

## بنية الكتاب

```text
book/
├── README.md
├── Figures/
│   ├── figure_*.png
│   └── embedded/
└── chapters/
    ├── ch01_leo_debris_environment/
    │   ├── chapter_01.md
    │   └── resources/*.pdf
    ├── ch02_space_surveillance_and_sp_catalog/
    ├── ch03_cdm_data_lineage_103_columns/
    ├── ch04_esoc_operations_and_ml_formulation/
    ├── ch05_physics_guided_two_stage_m6/
    └── ch06_simulation_and_cam_decision_support/
```

## ملاحظات تحريرية وعلمية

- المحتوى عربي مع الإبقاء على المصطلحات الإنجليزية والرموز الرياضية المتعارف عليها.
- تم الحفاظ على الرسومات المتاحة وربطها بمسارات نسبية داخل المستودع.
- عروض PDF مواد مساعدة وليست بديلًا عن نص الفصل.
- راجع القيم الرقمية والمراجع العلمية مقابل المصادر الأصلية قبل اعتماد الكتاب كوثيقة تشغيلية أو مرجع سلامة رسمي.
- ORBITAL-AI نموذج بحثي وليس نظامًا معتمدًا للتحكم في مركبات فضائية أو إصدار أوامر مناورة فعلية.
