# Viva preparation questions

Short answer outlines tied to this project. Replace the bracketed parts with your own measured results. Do not quote demo numbers.

## The problem and idea

1. **What problem does this project address?**
   Detecting potentially false claims when the same misinformation circulates in several languages, where methods built around the exact words of one language may not carry over.

2. **What is an embedding, and what does "cross-lingual" add?**
   An embedding turns text into a vector so that similar meanings are close together. A cross-lingual (multilingual) encoder is trained on many languages so that a claim and its translation should land near each other, even though they share no words. It is a goal, not a guarantee.

3. **Why not just translate everything to English first?**
   Translation adds a step that can change meaning, struggles with informal or code-mixed text, and adds cost. Shared embeddings are one way to avoid it. Whether they work better is an empirical question this project measures; do not assume the answer.

4. **Why is the encoder frozen rather than fine-tuned?**
   Simpler, cheaper and more reproducible on a student machine, and it isolates what the pretrained representation provides. Fine-tuning is listed as future work. It might score higher, which is a limitation.

## Data and labels

5. **Where do the labels come from, and how reliable are they?**
   [From the dataset's own annotation process, e.g. fact-checking websites.] The model learns that labelling, not ground truth. Different sources can use different verdict names, which is why a label map is printed and can be overridden.

6. **The dataset is heavily imbalanced. Why does that matter?**
   A model that always predicts the common class gets high accuracy but is useless. That is why the project reports macro-F1, fake-class precision / recall / F1 and balanced accuracy, includes a majority-class baseline, uses balanced class weights and tunes the decision threshold on validation data.

7. **What is the text in your dataset: original posts or something else?**
   [State it.] If it is fact-checkers' article text, a model may partly learn the style of such articles, so results may not carry over to raw social-media posts.

## Methodology and leakage

8. **How did you stop test data leaking into training?**
   One fixed split for all models. Claims that share a `claim_id` (translations or duplicates) stay in the same split; otherwise a chronological split when dates parse, else a stratified one with a printed warning. Duplicates and conflicting labels are removed. Vocabularies, scalers, `C` and thresholds are fitted or chosen on training / validation rows only.

9. **What does the label-flipping test prove?**
   Flipping every test label does not change a single test prediction, so test labels cannot be influencing training, tuning or thresholds. It checks one kind of leakage; it does not prove the data have no near-duplicates.

10. **Why a grouped split for translations?**
    If a claim is in training in English and in test in Hindi, the model has effectively seen the test claim. Scores would then overstate cross-lingual ability.

11. **You used one split and one seed. What is the risk?**
    Small differences between models may be noise. [Say whether you repeated runs, or report it as a limitation.]

## Results and interpretation

12. **Which model won, and did the embedding approach beat TF-IDF?**
    [Your measured answer.] Either outcome is valid. If TF-IDF char n-grams won, discuss why (domain vocabulary, text type, language mix, frozen encoder).

13. **What does the cross-lingual transfer test show?**
    [Train on one language, test on others; compare with training on all.] A large drop means little transfer; a small drop suggests the representation carries over. Languages with few test rows are noisy.

14. **How did you measure embedding alignment (research question 1)?**
    With parallel data, compare similarity of true translation pairs with mismatched pairs and the top-1 retrieval rate against chance. Without parallel data this cannot be measured; say so.

15. **Is the stand-in "hashing" encoder in the demo cross-lingual?**
    No. It only checks that the software runs. Demo numbers are not results.

## Explanations and generative AI

16. **What is the "Generative AI" part?**
    The explanations in this code are retrieval-based: similar labelled training claims in any language. No generative model is called. A prompt is generated that tells a model to use only the given information and not to declare a claim true or false. If you try a model, check its output by hand and say how you evaluated it.

17. **Can a generative model verify a claim?**
    No. It can summarise or rephrase, but it is not a fact-checking source. Reliable external evidence is still required.

## Limitations and ethics

18. **What are the main limitations?**
    Dataset-specific labels, domain and period; frozen encoder; single split and seed; language coverage; no evaluation on live posts or on code-mixed text unless your data has it; explanation usefulness not measured.

19. **What could go wrong if this were deployed?**
    False positives could suppress true content; false negatives could give false reassurance; performance may differ across languages and communities. Output should be advice to a human, not a verdict.

20. **What would you do next?**
    [E.g. an Indian-language dataset, encoder fine-tuning, repeated splits, evidence retrieval, a user study of explanations.]
