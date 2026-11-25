# Example outputs (retrieval-based explanations)

Selected as the most confident predictions in each outcome cell: TP = correctly flagged, TN = correctly passed, FP = real claim flagged, FN = fake claim missed. These come from the held-out test split. Neighbours are labelled *training* claims. No generative model was used.

## TP - en

> SHOCKING secret about the traffic fine system that doctors and officials do not want you to know, 100% proven!

- True label: **Potentially Fake**; predicted: **Potentially Fake** (score 1.00)
- Prediction: Potentially Fake (score 1.00; decision threshold 0.50). 3 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [en, Potentially Fake, similarity 0.94] SHOCKING secret about the new vaccine that doctors and officials do not want you to know, 100% proven!
  - [en, Potentially Fake, similarity 0.94] SHOCKING secret about the property tax system that doctors and officials do not want you to know, 100% proven!
  - [en, Potentially Fake, similarity 0.92] SHOCKING secret about the local train service that doctors and officials do not want you to know, 100% proven!

## TP - hi-latn

> yaataayaat jurmaana pranaali ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!

- True label: **Potentially Fake**; predicted: **Potentially Fake** (score 1.00)
- Prediction: Potentially Fake (score 1.00; decision threshold 0.50). 3 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi-latn, Potentially Fake, similarity 0.92] sampatti kar pranaali ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!
  - [hi-latn, Potentially Fake, similarity 0.90] gaon ke aspataal ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!
  - [hi-latn, Potentially Fake, similarity 0.90] bank rin yojana ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!

## TP - hi-latn

> ration card yojana ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!

- True label: **Potentially Fake**; predicted: **Potentially Fake** (score 1.00)
- Prediction: Potentially Fake (score 1.00; decision threshold 0.50). 3 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi-latn, Potentially Fake, similarity 0.96] bank rin yojana ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!
  - [hi-latn, Potentially Fake, similarity 0.95] metro pariyojana ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!
  - [hi-latn, Potentially Fake, similarity 0.94] krishi subsidy yojana ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!

## TN - hi

> अधिकारियों ने प्रेस ब्रीफिंग में बिजली कनेक्शन के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।

- True label: **Real**; predicted: **Real** (score 0.00)
- Prediction: Real (score 0.00; decision threshold 0.50). 0 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi, Real, similarity 0.96] अधिकारियों ने प्रेस ब्रीफिंग में नए टीके के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।
  - [hi, Real, similarity 0.96] अधिकारियों ने प्रेस ब्रीफिंग में पेंशन योजना के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।
  - [hi, Real, similarity 0.96] अधिकारियों ने प्रेस ब्रीफिंग में बाढ़ राहत कोष के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।

## TN - hi-latn

> Adhikariyon ne press briefing mein bank rin yojana ke baare mein kaaryakram ki pushti ki aur taareekhein saarvajanik record mein prakaashit keen.

- True label: **Real**; predicted: **Real** (score 0.00)
- Prediction: Real (score 0.00; decision threshold 0.50). 0 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi-latn, Real, similarity 0.98] Adhikariyon ne press briefing mein pension yojana ke baare mein kaaryakram ki pushti ki aur taareekhein saarvajanik record mein prakaashit …
  - [hi-latn, Real, similarity 0.98] Adhikariyon ne press briefing mein ration card yojana ke baare mein kaaryakram ki pushti ki aur taareekhein saarvajanik record mein prakaas…
  - [hi-latn, Real, similarity 0.98] Adhikariyon ne press briefing mein metro pariyojana ke baare mein kaaryakram ki pushti ki aur taareekhein saarvajanik record mein prakaashi…

## TN - hi

> अधिकारियों ने प्रेस ब्रीफिंग में बस पास प्रणाली के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।

- True label: **Real**; predicted: **Real** (score 0.00)
- Prediction: Real (score 0.00; decision threshold 0.50). 0 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi, Real, similarity 0.98] अधिकारियों ने प्रेस ब्रीफिंग में संपत्ति कर प्रणाली के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।
  - [hi, Real, similarity 0.97] अधिकारियों ने प्रेस ब्रीफिंग में नए टीके के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।
  - [hi, Real, similarity 0.96] अधिकारियों ने प्रेस ब्रीफिंग में यातायात जुर्माना प्रणाली के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।

## FP - en

> Anonymous source says that the public library will change overnight and everyone will lose money, no official confirmation needed.

- True label: **Real**; predicted: **Potentially Fake** (score 0.95)
- Prediction: Potentially Fake (score 0.95; decision threshold 0.50). 2 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [en, Potentially Fake, similarity 0.95] Anonymous source says that the city water supply will change overnight and everyone will lose money, no official confirmation needed.
  - [en, Potentially Fake, similarity 0.95] Anonymous source says that the new vaccine will change overnight and everyone will lose money, no official confirmation needed.
  - [en, Real, similarity 0.95] Anonymous source says that the pension scheme will change overnight and everyone will lose money, no official confirmation needed.

## FP - hi-latn

> Agyaat sutra ka kehna hai ki saarvajanik pustakaalay ke baare mein raaton-raat sab badal jayega aur sabka paisa doobega, kisi aadhikarik pushti ki zaroorat nahi.

- True label: **Real**; predicted: **Potentially Fake** (score 0.92)
- Prediction: Potentially Fake (score 0.92; decision threshold 0.50). 3 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi-latn, Potentially Fake, similarity 0.95] Agyaat sutra ka kehna hai ki gaon ke aspataal ke baare mein raaton-raat sab badal jayega aur sabka paisa doobega, kisi aadhikarik pushti ki…
  - [hi-latn, Potentially Fake, similarity 0.95] Agyaat sutra ka kehna hai ki naye teeke ke baare mein raaton-raat sab badal jayega aur sabka paisa doobega, kisi aadhikarik pushti ki zaroo…
  - [hi-latn, Potentially Fake, similarity 0.95] Agyaat sutra ka kehna hai ki sampatti kar pranaali ke baare mein raaton-raat sab badal jayega aur sabka paisa doobega, kisi aadhikarik push…

## FP - hi

> अज्ञात सूत्र का कहना है कि सार्वजनिक पुस्तकालय के बारे में रातों-रात सब बदल जाएगा और सबका पैसा डूबेगा, किसी आधिकारिक पुष्टि की ज़रूरत नहीं।

- True label: **Real**; predicted: **Potentially Fake** (score 0.91)
- Prediction: Potentially Fake (score 0.91; decision threshold 0.50). 2 of the 3 most similar labelled training claims are 'Potentially Fake'. This only describes similarity inside the model's embedding space; it is not evidence about whether the claim is true. Verify with reliable external sources.
- Most similar labelled training claims:
  - [hi, Potentially Fake, similarity 0.94] अज्ञात सूत्र का कहना है कि नए टीके के बारे में रातों-रात सब बदल जाएगा और सबका पैसा डूबेगा, किसी आधिकारिक पुष्टि की ज़रूरत नहीं।
  - [hi, Potentially Fake, similarity 0.94] अज्ञात सूत्र का कहना है कि राशन कार्ड योजना के बारे में रातों-रात सब बदल जाएगा और सबका पैसा डूबेगा, किसी आधिकारिक पुष्टि की ज़रूरत नहीं।
  - [hi, Real, similarity 0.94] अज्ञात सूत्र का कहना है कि पेंशन योजना के बारे में रातों-रात सब बदल जाएगा और सबका पैसा डूबेगा, किसी आधिकारिक पुष्टि की ज़रूरत नहीं।
